"""P5: margin-证书 vs worst-case(tanh)证书 驱动的自适应 key 量化。
对每 (层,kv头):per-channel 对称量化 key 到 b∈{8,6,4,3,2} bits。
证书给出该头在该 bits 下的 TV 上界;目标 TV<=tau 时各证书允许的最低 bits。
指标:同 tau 下的平均 bits(证书效率)+ 实际 TV 是否守约 + 端到端 PPL。
python3 p5_cert_quant.py --tau 0.05
"""
import argparse, json, math, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rsc.operator import rope_freqs, complexify
from p0_premises import load_corpus, capture_qk

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(__file__), "out")
BITS = [8, 6, 4, 3, 2]


def quantize(k, bits):
    """per-channel 对称量化。k:[S,D] -> 反量化后 [S,D] 与 步长 sigma:[D]"""
    amax = k.abs().amax(0).clamp_min(1e-8)
    q = 2 ** (bits - 1) - 1
    sigma = amax / q
    kq = (k / sigma).round().clamp(-q, q) * sigma
    return kq, sigma


def head_logits(kc, qc, theta, D, n_q=128):
    S = kc.shape[0]
    pos = torch.arange(S, dtype=torch.float32)
    rk = kc * torch.exp(1j * pos[:, None] * theta[None, :])
    rq = qc[-n_q:] * torch.exp(1j * pos[-n_q:, None] * theta[None, :])
    lg = (rq @ rk.mH).real / math.sqrt(D)
    mask = pos[None, :] <= pos[-n_q:, None]
    return lg, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--seqlen", type=int, default=2048)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(DEV).eval()
    cfg = model.config
    D = cfg.hidden_size // cfg.num_attention_heads
    G = cfg.num_attention_heads // cfg.num_key_value_heads
    theta = rope_freqs(D, cfg.rope_theta)
    calib_ids, valid_ids = load_corpus(args.seqlen, tok)
    qk_v = capture_qk(model, valid_ids)

    rows = []
    choice = {"margin": {}, "tanh": {}}
    for li in range(len(qk_v)):
        q_r, k_r = qk_v[li]
        for hk in range(cfg.num_key_value_heads):
            k_head = k_r[:, hk, :]                       # [S,D] pre-RoPE
            qc = complexify(q_r[:, hk * G, :], D)
            kc = complexify(k_head, D)
            ex, mask = head_logits(kc, qc, theta, D)
            p = ex.masked_fill(~mask, -1e30).softmax(-1)
            for bits in BITS:
                kq, sigma = quantize(k_head, bits)
                kcq = complexify(kq, D)
                aq, _ = head_logits(kcq, qc, theta, D)
                eps = (aq - ex).masked_fill(~mask, 0.0)
                # 真实 TV
                pq = aq.masked_fill(~mask, -1e30).softmax(-1)
                tv_true = float((0.5 * (p - pq).abs().sum(-1)).mean())
                # margin 证书(可在线计算:用量化侧分布加权也可,这里用精确侧成立性检验)
                ebar = (p * eps).sum(-1, keepdim=True)
                dev = (p * (eps - ebar).abs()).sum(-1)
                emax = eps.abs().amax(-1)
                mb = float((0.5 * torch.exp(2 * emax.clamp(max=8)) * (dev + 0.5 * dev.pow(2) * torch.exp(emax.clamp(max=8)))).mean())
                # worst-case 证书:tanh(||q||·||sigma||/(2 sqrt(D))) —— 2605.20868 形式
                qn = q_r[:, hk * G, :][-128:].norm(dim=-1)
                delta = float((qn * sigma.norm() / (2 * math.sqrt(D))).mean())
                tb = math.tanh(min(delta, 20.0))
                rows.append({"layer": li, "kv_head": hk, "bits": bits,
                             "tv_true": tv_true, "margin_bound": mb, "tanh_bound": tb})
            # 各证书允许的最低 bits
            for cert, key in (("margin", "margin_bound"), ("tanh", "tanh_bound")):
                ok = [r for r in rows if r["layer"] == li and r["kv_head"] == hk and r[key] <= args.tau]
                choice[cert][(li, hk)] = min((r["bits"] for r in ok), default=16)
        if li % 7 == 0:
            print(f"layer {li} done", flush=True)

    import statistics as st
    mb_bits = list(choice["margin"].values())
    tb_bits = list(choice["tanh"].values())
    viol = [r for r in rows if r["margin_bound"] <= args.tau and r["tv_true"] > r["margin_bound"] + 1e-3]
    print(f"\ntau={args.tau}")
    print(f"margin证书: 平均bits={st.mean(mb_bits):.2f} (中位{st.median(mb_bits)}) 16=回退FP")
    print(f"tanh证书:   平均bits={st.mean(tb_bits):.2f} (中位{st.median(tb_bits)})")
    print(f"margin证书违约率: {len(viol)}/{sum(1 for r in rows if r['margin_bound']<=args.tau)}")

    # 端到端:按 margin 证书的逐头 bits 真实量化 key_proj 输出,测 PPL
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
                        parts.append(x if b >= 16 else quantize(x.reshape(-1, D), b)[0].view_as(x))
                    return torch.stack(parts, 2).view_as(out)
                return hook
            hooks.append(layer.self_attn.k_proj.register_forward_hook(mk(bs)))
        return lambda: [h.remove() for h in hooks]

    with torch.no_grad():
        base = float(torch.exp(model(valid_ids.to(DEV), labels=valid_ids.to(DEV)).loss))
        rm = install(choice["margin"]); pm = float(torch.exp(model(valid_ids.to(DEV), labels=valid_ids.to(DEV)).loss)); rm()
        rm = install(choice["tanh"]); pt = float(torch.exp(model(valid_ids.to(DEV), labels=valid_ids.to(DEV)).loss)); rm()
    print(f"PPL: baseline={base:.3f} margin-cert量化={pm:.3f} tanh-cert量化={pt:.3f}")
    json.dump({"rows": rows, "margin_bits": st.mean(mb_bits), "tanh_bits": st.mean(tb_bits),
               "ppl": {"base": base, "margin": pm, "tanh": pt}},
              open(os.path.join(OUT, "p5_cert_quant.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
