"""P7: 分频带残差见证。B个频带,每token存B个带内残差范数(RoPE酉性=>位置不变)。
|eps_t| <= sum_b ||q_b||·rho_{t,b}/sqrt(D)。B=1退化为P6全局CS。
python3 p7_banded_witness.py
"""
import json, math, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rsc.operator import rope_freqs, complexify
from p0_premises import load_corpus, capture_qk
from p5_cert_quant import quantize, head_logits

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(__file__), "out")
BITS = [8, 6, 4, 3, 2]
TAUS = [0.05, 0.1, 0.2]
BANDS = [1, 4, 8, 16]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(DEV).eval()
    cfg = model.config
    D = cfg.hidden_size // cfg.num_attention_heads
    P = D // 2
    G = cfg.num_attention_heads // cfg.num_key_value_heads
    theta = rope_freqs(D, cfg.rope_theta)
    _, valid_ids = load_corpus(2048, tok)
    qk_v = capture_qk(model, valid_ids)
    del model

    rows = []
    for li in range(0, len(qk_v), 2):          # 隔层采样,14层x2头足够
        q_r, k_r = qk_v[li]
        for hk in range(cfg.num_key_value_heads):
            k_head = k_r[:, hk, :]
            qc = complexify(q_r[:, hk * G, :], D)
            kc = complexify(k_head, D)
            ex, mask = head_logits(kc, qc, theta, D)
            p = ex.masked_fill(~mask, -1e30).softmax(-1)
            qcp = qc[-128:]
            for bits in BITS:
                kq, _ = quantize(k_head, bits)
                rc = kc - complexify(kq, D)                    # 复残差 [S,P]
                aq, _ = head_logits(complexify(kq, D), qc, theta, D)
                pq = aq.masked_fill(~mask, -1e30).softmax(-1)
                eps = (aq - ex).masked_fill(~mask, 0.0)
                tv_true = float((0.5 * (p - pq).abs().sum(-1)).mean())
                row = {"layer": li, "kv_head": hk, "bits": bits, "tv_true": tv_true}
                for B in BANDS:
                    bw = P // B
                    # c[n,t] = sum_b ||q_b[n]||·||r_b[t]|| / sqrt(D)  (复模长)
                    qb = qcp.abs().pow(2).view(128, B, bw).sum(-1).sqrt()   # [N,B]
                    rb = rc.abs().pow(2).view(-1, B, bw).sum(-1).sqrt()     # [S,B]
                    c = (qb @ rb.T / math.sqrt(D)).masked_fill(~mask, 0.0)
                    cbar = (pq * c).sum(-1, keepdim=True)
                    cmax = c.amax(-1).clamp(max=8)
                    dev = (pq * (c - cbar).abs()).sum(-1) + cbar.squeeze(-1) * (torch.exp(cmax) - 1)
                    row[f"wit{B}"] = float((0.5 * torch.exp(2 * cmax) * dev).mean())
                    if B == BANDS[-1]:
                        viol = bool((eps.abs().amax(-1) > c.amax(-1) + 1e-4).any())
                        row["cs_violated"] = viol               # 逐token CS 界成立性抽查
                rows.append(row)
        print(f"layer {li} done", flush=True)

    import statistics as st
    heads = sorted({(r["layer"], r["kv_head"]) for r in rows})
    print("\ntau  | " + " | ".join(f"B={B}平均bits" for B in BANDS))
    out = {}
    for tau in TAUS:
        line = []
        for B in BANDS:
            ch = []
            for h in heads:
                ok = [r["bits"] for r in rows if (r["layer"], r["kv_head"]) == h and r[f"wit{B}"] <= tau]
                ch.append(min(ok, default=16))
            line.append(st.mean(ch))
        out[tau] = dict(zip(map(str, BANDS), line))
        print(f"{tau:4.2f} | " + " | ".join(f"{x:10.2f}" for x in line))
    nv = sum(1 for r in rows if r.get("cs_violated"))
    print(f"逐token CS界违约头x档: {nv}/{len([r for r in rows if 'cs_violated' in r])}")
    json.dump(out, open(os.path.join(OUT, "p7_banded.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
