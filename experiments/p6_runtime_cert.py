"""P6 决定性实验:运行时可算的残差见证 margin 证书。
证书族(对每头每 bits):
  A tanh-wc:   tanh(||q||·||sigma||/(2 sqrt(D)))            全局worst-case(2605.20868式)
  B witness:   逐token |eps_t|<=c_t=||q||·rho_t/sqrt(D),rho_t=写cache时存的残差范数
               TV界 = 0.5·e^{2 max c}·( E_p~[|c - E_p~ c|] + E_p~[c] · (e^{max c}-1) 修正 )
               —— 全部量运行时可得(p~ 为压缩侧注意力,c_t 为缓存标量×当前||q||)
  C oracle:    真实eps的margin界(上参考,不可运行时获得)
指标:tau∈{0.02,0.05,0.1,0.2} 下各证书允许的最低bits、违约率、B选择下的端到端PPL。
python3 p6_runtime_cert.py
"""
import argparse, json, math, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rsc.operator import rope_freqs, complexify
from p0_premises import load_corpus, capture_qk
from p5_cert_quant import quantize, head_logits

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(__file__), "out")
BITS = [8, 6, 4, 3, 2]
TAUS = [0.02, 0.05, 0.1, 0.2]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(DEV).eval()
    cfg = model.config
    D = cfg.hidden_size // cfg.num_attention_heads
    G = cfg.num_attention_heads // cfg.num_key_value_heads
    theta = rope_freqs(D, cfg.rope_theta)
    _, valid_ids = load_corpus(2048, tok)
    qk_v = capture_qk(model, valid_ids)

    rows = []
    for li in range(len(qk_v)):
        q_r, k_r = qk_v[li]
        for hk in range(cfg.num_key_value_heads):
            k_head = k_r[:, hk, :]
            qc = complexify(q_r[:, hk * G, :], D)
            kc = complexify(k_head, D)
            ex, mask = head_logits(kc, qc, theta, D)
            p = ex.masked_fill(~mask, -1e30).softmax(-1)
            qn = q_r[:, hk * G, :][-128:].norm(dim=-1)          # ||q|| per query
            for bits in BITS:
                kq, sigma = quantize(k_head, bits)
                rho = (k_head - kq).norm(dim=-1)                 # [S] 写时可得的残差范数
                kcq = complexify(kq, D)
                aq, _ = head_logits(kcq, qc, theta, D)
                pq = aq.masked_fill(~mask, -1e30).softmax(-1)    # 压缩侧注意力(运行时可得)
                eps = (aq - ex).masked_fill(~mask, 0.0)
                tv_true = float((0.5 * (p - pq).abs().sum(-1)).mean())
                # A: worst-case
                tb = float(torch.tanh((qn * sigma.norm() / (2 * math.sqrt(D))).clamp(max=20)).mean())
                # B: witness (全部运行时量)
                c = (qn[:, None] * rho[None, :] / math.sqrt(D)).masked_fill(~mask, 0.0)  # [N,S]
                cbar = (pq * c).sum(-1, keepdim=True)
                cmax = c.amax(-1)
                devB = (pq * (c - cbar).abs()).sum(-1) + cbar.squeeze(-1) * (torch.exp(cmax.clamp(max=8)) - 1)
                wb = float((0.5 * torch.exp(2 * cmax.clamp(max=8)) * devB).mean())
                # C: oracle margin
                ebar = (p * eps).sum(-1, keepdim=True)
                dev = (p * (eps - ebar).abs()).sum(-1)
                emax = eps.abs().amax(-1)
                ob = float((0.5 * torch.exp(2 * emax.clamp(max=8)) * (dev + 0.5 * dev.pow(2) * torch.exp(emax.clamp(max=8)))).mean())
                rows.append({"layer": li, "kv_head": hk, "bits": bits, "tv_true": tv_true,
                             "tanh": tb, "witness": wb, "oracle": ob})
        if li % 7 == 0:
            print(f"layer {li} done", flush=True)

    import statistics as st
    heads = sorted({(r["layer"], r["kv_head"]) for r in rows})
    print("\ntau  | tanh平均bits | witness平均bits | oracle平均bits | witness违约")
    summary = {}
    for tau in TAUS:
        res = {}
        for cert in ("tanh", "witness", "oracle"):
            bits_choice = {}
            for h in heads:
                ok = [r["bits"] for r in rows if (r["layer"], r["kv_head"]) == h and r[cert] <= tau]
                bits_choice[h] = min(ok, default=16)
            res[cert] = bits_choice
        viol = sum(1 for r in rows if r["witness"] <= tau and r["tv_true"] > r["witness"] + 1e-3)
        tot = sum(1 for r in rows if r["witness"] <= tau)
        print(f"{tau:4.2f} | {st.mean(res['tanh'].values()):11.2f} | {st.mean(res['witness'].values()):14.2f} "
              f"| {st.mean(res['oracle'].values()):13.2f} | {viol}/{tot}")
        summary[tau] = {c: st.mean(res[c].values()) for c in res} | {"viol": f"{viol}/{tot}"}

    # 端到端 PPL @ tau=0.05,witness 选择
    tau = 0.05
    bits_w = {}
    for h in heads:
        ok = [r["bits"] for r in rows if (r["layer"], r["kv_head"]) == h and r["witness"] <= tau]
        bits_w[h] = min(ok, default=16)

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
        rm = install(bits_w); pw = float(torch.exp(model(valid_ids.to(DEV), labels=valid_ids.to(DEV)).loss)); rm()
    print(f"\nPPL @tau=0.05 witness选择: baseline={base:.3f} -> {pw:.3f}  平均bits={st.mean(bits_w.values()):.2f}")
    json.dump({"summary": {str(k): v for k, v in summary.items()},
               "ppl": {"base": base, "witness": pw}},
              open(os.path.join(OUT, "p6_runtime_cert.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
