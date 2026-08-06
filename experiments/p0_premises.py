"""P0 前提验证:在真实模型上检验 RSC 的四个数学前提。

P1 频率交叉能量集中:少数频率对承载绝大部分 E[|k_j||q_j|]
P2 簇内低秩:相邻频率簇的跨协方差 E[k_C q_C^*] 近 rank-1
P3 漂移可控:逐头有效注意力距离 L_h 可测,漂移项 = L_h * 簇宽
P4 margin 界紧度:数据依赖 TV 界 vs worst-case tanh 界 vs 真实 TV

用法: python3 p0_premises.py [--model Qwen/Qwen2.5-1.5B-Instruct] [--seqlen 2048]
输出: rsc/experiments/out/p0_summary.json + stdout 摘要
"""
import argparse, json, math, os, sys
import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(__file__), "out")


def load_corpus(seqlen, tok):
    base = os.path.join(os.path.dirname(__file__), "data")
    text = ""
    for f in ("pride.txt", "code_sample.py"):
        with open(os.path.join(base, f), encoding="utf-8", errors="ignore") as fh:
            text += fh.read() + "\n\n"
    ids = tok(text, return_tensors="pt").input_ids[0]
    # 两段不重叠窗口:校准窗 + 验证窗
    assert ids.numel() >= 2 * seqlen, "语料不足"
    return ids[:seqlen].unsqueeze(0), ids[seqlen : 2 * seqlen].unsqueeze(0)


def capture_qk(model, input_ids):
    """抓每层 q_proj/k_proj 输出(pre-RoPE),返回 [(q, k)] per layer,q:[S,Hq,D] k:[S,Hk,D]"""
    caps = []
    hooks = []

    def mk(layer_caps, name):
        def hook(mod, inp, out):
            layer_caps[name] = out.detach().float().cpu()
        return hook

    for layer in model.model.layers:
        lc = {}
        caps.append(lc)
        hooks.append(layer.self_attn.q_proj.register_forward_hook(mk(lc, "q")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(mk(lc, "k")))
    with torch.no_grad():
        model(input_ids.to(model.device))
    for h in hooks:
        h.remove()
    cfg = model.config
    D = cfg.hidden_size // cfg.num_attention_heads
    out = []
    for lc in caps:
        q = lc["q"][0].view(-1, cfg.num_attention_heads, D)
        k = lc["k"][0].view(-1, cfg.num_key_value_heads, D)
        out.append((q, k))
    return out


def complexify(x, D):
    """rotate_half 约定:pair (i, i+D/2) 共享频率 theta_i。x:[S,H,D] -> [S,H,D/2] complex"""
    return torch.complex(x[..., : D // 2], x[..., D // 2 :])


def rope_freqs(D, base):
    j = torch.arange(D // 2, dtype=torch.float64)
    return (base ** (-2 * j / D)).float()  # theta_j,降序


def softmax_tv(p, q_):
    return 0.5 * (p - q_).abs().sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--probe-layers", type=int, nargs="*", default=None,
                    help="只分析这些层(默认全部)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(DEV).eval()
    cfg = model.config
    D = cfg.hidden_size // cfg.num_attention_heads
    Hq, Hk = cfg.num_attention_heads, cfg.num_key_value_heads
    G = Hq // Hk  # GQA 组大小
    theta = rope_freqs(D, cfg.rope_theta)
    P = D // 2

    calib_ids, valid_ids = load_corpus(args.seqlen, tok)
    qk_calib = capture_qk(model, calib_ids)
    qk_valid = capture_qk(model, valid_ids)
    layers = args.probe_layers or list(range(len(qk_calib)))

    S = args.seqlen
    pos = torch.arange(S)
    summary = {"model": args.model, "seqlen": S, "D": D, "Hq": Hq, "Hk": Hk,
               "rope_theta": cfg.rope_theta, "P": P, "per_head": []}

    for li in layers:
        q_r, k_r = qk_calib[li]
        qc = complexify(q_r, D)          # [S,Hq,P]
        kc = complexify(k_r, D)          # [S,Hk,P]
        for hk in range(Hk):
            # GQA:该 kv 头对应的所有 q 头合并统计
            qg = qc[:, hk * G:(hk + 1) * G, :].reshape(-1, P)   # [S*G,P]
            kg = kc[:, hk, :]                                    # [S,P]
            # P1: 逐频率交叉能量
            eq = qg.abs().mean(0)
            ek = kg.abs().mean(0)
            xe = (eq * ek)                                       # E|q_j| * E|k_j| 代理
            xe_sorted, order = xe.sort(descending=True)
            cum = xe_sorted.cumsum(0) / xe.sum()
            n90 = int((cum < 0.90).sum()) + 1
            n99 = int((cum < 0.99).sum()) + 1
            # P2: 连续频率簇 (宽度4) 的跨协方差 rank-1 占比
            r1 = []
            for c0 in range(0, P, 4):
                kC = kg[:, c0:c0 + 4]
                qC = qg[:, c0:c0 + 4]
                # 跨协方差 (P2 用校准窗)
                M = (qC.mH @ kC.to(qC.dtype)) / kC.shape[0] if qC.shape[0] == kC.shape[0] \
                    else (qC[: kC.shape[0]].mH @ kC) / kC.shape[0]
                sv = torch.linalg.svdvals(M)
                r1.append(float(sv[0] / sv.sum().clamp_min(1e-12)))
            # P3: 有效注意力距离(用真实 RoPE 后 logits 的注意力质量加权 |Δ|)
            # 取校准窗后半段 256 个 query,全长 key,单 q 头代表(组内第一个)
            qs = qc[S - 256:, hk * G, :]                          # [256,P]
            rot_k = kg * torch.exp(1j * pos[:, None] * theta[None, :])
            rot_q = qs * torch.exp(1j * pos[S - 256:, None] * theta[None, :])
            logits = (rot_q @ rot_k.mH).real / math.sqrt(D)       # [256,S]
            mask = pos[None, :] <= pos[S - 256:, None]
            logits = logits.masked_fill(~mask, -1e30)
            attn = logits.softmax(-1)
            dist = (pos[S - 256:, None] - pos[None, :]).clamp_min(0)
            L_eff = float((attn * dist).sum(-1).mean())
            ent = float((-attn.clamp_min(1e-12).log() * attn).sum(-1).mean())
            summary["per_head"].append({
                "layer": li, "kv_head": hk, "n90": n90, "n99": n99,
                "rank1_ratio_w4": r1, "L_eff": L_eff, "entropy": ent,
                "xe_top8_frac": float(cum[7]) if P >= 8 else 1.0,
            })
        print(f"layer {li} done", flush=True)

    # 聚合打印
    import statistics as st
    n90s = [h["n90"] for h in summary["per_head"]]
    r1m = [st.mean(h["rank1_ratio_w4"]) for h in summary["per_head"]]
    print(f"\nP1: n90 median={st.median(n90s)}/{P} pairs (min={min(n90s)}, max={max(n90s)})")
    print(f"P2: cluster(w=4) rank-1 ratio mean={st.mean(r1m):.3f} min={min(r1m):.3f}")
    print(f"P3: L_eff range: {min(h['L_eff'] for h in summary['per_head']):.0f}"
          f" .. {max(h['L_eff'] for h in summary['per_head']):.0f}")

    with open(os.path.join(OUT, "p0_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("saved", os.path.join(OUT, "p0_summary.json"))


if __name__ == "__main__":
    main()
