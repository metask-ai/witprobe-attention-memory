"""P9 混合档:高残差 token 保留 FP16(写时按 rho_t 判定,query 无关),截断 cmax。
有效 bits = bits*(1-frac) + 16*frac(+witness 开销另计)。
python p9_mixed.py --model <path> --seqlen 8192 --dtype bfloat16 --exact-fracs 0 0.005 0.02
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
TAUS = [0.05, 0.1, 0.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--exact-fracs", type=float, nargs="*", default=[0.0, 0.005, 0.02])
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
    qk_v = capture_qk(model, valid_ids)
    del model

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
                kq0, _ = quantize(k_head, bits)
                rho0 = (k_head - kq0).norm(dim=-1)          # [S]
                for frac in args.exact_fracs:
                    m = int(round(frac * k_head.shape[0]))
                    kq = kq0.clone()
                    rho = rho0.clone()
                    if m > 0:
                        top = rho0.topk(m).indices           # 写时可判:残差最大的 m 个 token 存 FP
                        kq[top] = k_head[top]
                        rho[top] = 0.0
                    aq, _ = head_logits(complexify(kq, D), qc, theta, D)
                    pq = aq.masked_fill(~mask, -1e30).softmax(-1)
                    tv_true = float((0.5 * (p - pq).abs().sum(-1)).mean())
                    c = (qn[:, None] * rho[None, :] / math.sqrt(D)).masked_fill(~mask, 0.0)
                    cbar = (pq * c).sum(-1, keepdim=True)
                    cmax = c.amax(-1).clamp(max=8)
                    devB = (pq * (c - cbar).abs()).sum(-1) + cbar.squeeze(-1) * (torch.exp(cmax) - 1)
                    wb = float((0.5 * torch.exp(2 * cmax) * devB).mean())
                    eff_bits = bits * (1 - frac) + 16 * frac
                    rows.append({"layer": li, "kv_head": hk, "bits": bits, "frac": frac,
                                 "eff_bits": eff_bits, "tv_true": tv_true, "witness": wb})
        if li % 7 == 0:
            print(f"layer {li}/{len(qk_v)} done", flush=True)

    import statistics as st
    heads = sorted({(r["layer"], r["kv_head"]) for r in rows})
    print(f"\nmodel={os.path.basename(args.model)} seqlen={args.seqlen}")
    print("tau  | frac  | witness平均有效bits | 违约")
    summary = {}
    for tau in TAUS:
        for frac in args.exact_fracs:
            bc = {}
            for h in heads:
                ok = [r["eff_bits"] for r in rows
                      if (r["layer"], r["kv_head"]) == h and r["frac"] == frac and r["witness"] <= tau]
                bc[h] = min(ok, default=16)
            viol = sum(1 for r in rows if r["frac"] == frac and r["witness"] <= tau
                       and r["tv_true"] > r["witness"] + 1e-3)
            tot = sum(1 for r in rows if r["frac"] == frac and r["witness"] <= tau)
            print(f"{tau:4.2f} | {frac:.3f} | {st.mean(bc.values()):17.2f} | {viol}/{tot}")
            summary[f"{tau}_{frac}"] = {"bits": st.mean(bc.values()), "viol": f"{viol}/{tot}"}
    fn = os.path.join(OUT, f"p9_{os.path.basename(args.model)}_{args.seqlen}.json")
    json.dump({"summary": summary}, open(fn, "w"), indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
