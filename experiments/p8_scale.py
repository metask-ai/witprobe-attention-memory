"""P8 规模化:witness/tanh/oracle 证书对比,参数化模型/序列长/精度(7B 用 bf16)。
python p8_scale.py --model <path> --seqlen 2048 --dtype bfloat16 [--skip-ppl]
"""
import argparse, json, math, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rsc.operator import rope_freqs, complexify
from p0_premises import load_corpus, capture_qk
from p5_cert_quant import quantize, head_logits

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
OUT = os.path.join(os.path.dirname(__file__), "out")
BITS = [8, 6, 4, 3, 2]
TAUS = [0.02, 0.05, 0.1, 0.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--tag", default="p8")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).to(DEV).eval()
    cfg = model.config
    D = cfg.hidden_size // cfg.num_attention_heads
    G = cfg.num_attention_heads // cfg.num_key_value_heads
    theta = rope_freqs(D, cfg.rope_theta)
    _, valid_ids = load_corpus(args.seqlen, tok)
    qk_v = capture_qk(model, valid_ids)   # 已 .float().cpu()

    rows = []
    for li in range(len(qk_v)):
        q_r, k_r = qk_v[li]
        for hk in range(cfg.num_key_value_heads):
            k_head = k_r[:, hk, :]
            qc = complexify(q_r[:, hk * G, :], D)
            kc = complexify(k_head, D)
            ex, mask = head_logits(kc, qc, theta, D)
            p = ex.masked_fill(~mask, -1e30).softmax(-1)
            qn = q_r[:, hk * G, :][-128:].norm(dim=-1)
            for bits in BITS:
                kq, sigma = quantize(k_head, bits)
                rho = (k_head - kq).norm(dim=-1)
                aq, _ = head_logits(complexify(kq, D), qc, theta, D)
                pq = aq.masked_fill(~mask, -1e30).softmax(-1)
                eps = (aq - ex).masked_fill(~mask, 0.0)
                tv_true = float((0.5 * (p - pq).abs().sum(-1)).mean())
                tb = float(torch.tanh((qn * sigma.norm() / (2 * math.sqrt(D))).clamp(max=20)).mean())
                c = (qn[:, None] * rho[None, :] / math.sqrt(D)).masked_fill(~mask, 0.0)
                cbar = (pq * c).sum(-1, keepdim=True)
                cmax = c.amax(-1).clamp(max=8)
                devB = (pq * (c - cbar).abs()).sum(-1) + cbar.squeeze(-1) * (torch.exp(cmax) - 1)
                wb = float((0.5 * torch.exp(2 * cmax) * devB).mean())
                ebar = (p * eps).sum(-1, keepdim=True)
                dev = (p * (eps - ebar).abs()).sum(-1)
                emax = eps.abs().amax(-1).clamp(max=8)
                ob = float((0.5 * torch.exp(2 * emax) * (dev + 0.5 * dev.pow(2) * torch.exp(emax))).mean())
                rows.append({"layer": li, "kv_head": hk, "bits": bits, "tv_true": tv_true,
                             "tanh": tb, "witness": wb, "oracle": ob})
        if li % 7 == 0:
            print(f"layer {li}/{len(qk_v)} done", flush=True)

    import statistics as st
    heads = sorted({(r["layer"], r["kv_head"]) for r in rows})
    print(f"\nmodel={args.model} seqlen={args.seqlen} dtype={args.dtype}")
    print("tau  | tanh平均bits | witness平均bits | oracle平均bits | witness违约")
    summary = {}
    choice_cache = {}
    for tau in TAUS:
        res = {}
        for cert in ("tanh", "witness", "oracle"):
            bc = {}
            for h in heads:
                ok = [r["bits"] for r in rows if (r["layer"], r["kv_head"]) == h and r[cert] <= tau]
                bc[h] = min(ok, default=16)
            res[cert] = bc
        choice_cache[tau] = res
        viol = sum(1 for r in rows if r["witness"] <= tau and r["tv_true"] > r["witness"] + 1e-3)
        tot = sum(1 for r in rows if r["witness"] <= tau)
        print(f"{tau:4.2f} | {st.mean(res['tanh'].values()):11.2f} | {st.mean(res['witness'].values()):14.2f} "
              f"| {st.mean(res['oracle'].values()):13.2f} | {viol}/{tot}")
        summary[str(tau)] = {c: st.mean(res[c].values()) for c in res} | {"viol": f"{viol}/{tot}"}

    result = {"model": args.model, "seqlen": args.seqlen, "dtype": args.dtype, "summary": summary}

    if not args.skip_ppl:
        tau = 0.05
        bits_w = choice_cache[tau]["witness"]

        def install(choice_map):
            hooks = []
            for li, layer in enumerate(model.model.layers):
                bs = [choice_map.get((li, hk), 16) for hk in range(cfg.num_key_value_heads)]
                def mk(bs_):
                    def hook(mod, inp, out):
                        o = out.view(out.shape[0], out.shape[1], cfg.num_key_value_heads, D)
                        parts = []
                        for hh, b in enumerate(bs_):
                            x = o[:, :, hh, :]
                            if b >= 16:
                                parts.append(x)
                            else:
                                xf = x.reshape(-1, D).float()
                                parts.append(quantize(xf, b)[0].to(x.dtype).view_as(x))
                        return torch.stack(parts, 2).view_as(out)
                    return hook
                hooks.append(layer.self_attn.k_proj.register_forward_hook(mk(bs)))
            return lambda: [h.remove() for h in hooks]

        with torch.no_grad():
            base = float(torch.exp(model(valid_ids.to(DEV), labels=valid_ids.to(DEV)).loss))
            rm = install(bits_w)
            pw = float(torch.exp(model(valid_ids.to(DEV), labels=valid_ids.to(DEV)).loss))
            rm()
        print(f"PPL @tau=0.05 witness选择: baseline={base:.3f} -> {pw:.3f}  平均bits={st.mean(bits_w.values()):.2f}")
        result["ppl"] = {"base": base, "witness": pw, "avg_bits": st.mean(bits_w.values())}

    fn = os.path.join(OUT, f"{args.tag}_{os.path.basename(args.model)}_{args.seqlen}.json")
    json.dump(result, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
