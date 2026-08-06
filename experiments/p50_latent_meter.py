# -*- coding: utf-8 -*-
"""P50(R8-E1.1,里程碑 M1):MLA latent 尺子首验(DeepSeek-V2-Lite 代理)。

理论要点(R8_ROADMAP E1):MLA 缓存对象是 latent c_t(512,无 RoPE)+ 共享 k_pe(64,
post-RoPE)。量化 latent 的 logit 误差经 k_nope = W_UK c 线性传播:
    Δs_t = sm·(q_nopeᵀ W_UK Δc_t + q_peᵀ Δk_pe,t)
有效 query u_h = W_UK^{(h)T} q_nope ∈ R^512。Tier A 见证 = Δc 的分带范数
(**无 RoPE ⇒ 无需带酉性,位置不变性免费**)+ k_pe 侧沿用 64 维分带;
Tier B = 减性抖动 scale 的 ς² 代理经同一 u_h 传播。

工程正确性(proof–kernel contract 文化):
  · q/k 用 SDPA monkeypatch 捕获(post-RoPE/post-mscale,与前向逐位同源,
    YaRN 约定零风险);
  · 内建契约自检:W_UK·c 必须与捕获 k 的 nope 段逐位一致(bf16 容差),
    否则 latent 映射就是错的,直接 assert;
  · A 的计算走 log 域不截断(p35 口径,溢出→平凡界)。

python p50_latent_meter.py [--model ~/witcert/models/DeepSeek-V2-Lite]
                           [--seqlen 4096] [--docs 2] [--tag ""]
"""
import argparse, glob, json, math, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.join(os.path.dirname(__file__), "out")
NBAND = 8
DELTA_REQ = 1e-2
TAUS = (0.2, 0.5)
BLK = 256


def q_rtn(X, bits):
    qm = 2 ** (bits - 1) - 1
    sc = (X.abs().amax(0, keepdim=True) / qm).clamp_min(1e-8)
    return (X / sc).round().clamp(-qm, qm) * sc, sc.expand_as(X)


def q_dither(X, gen):
    qm = 126.5
    sc = (X.abs().amax(0, keepdim=True) / qm).clamp_min(1e-8)
    u = (torch.rand(X.shape, generator=gen) - 0.5).to(X.device)
    return ((X / sc + u).round() - u) * sc, sc.expand_as(X)


def blockwise(qfn, X, *a):
    """逐 256-token 块独立定标(部署口径),返回 (X̂, 逐元素 scale)。"""
    outs, scs = [], []
    for i in range(0, X.shape[0], BLK):
        xb = X[i:i + BLK]
        xhat, sc = qfn(xb, *a)
        outs.append(xhat); scs.append(sc)
    return torch.cat(outs, 0), torch.cat(scs, 0)


QUANT = {
    "latent-rtn8": lambda C, g: blockwise(q_rtn, C, 8),
    "latent-dither8": lambda C, g: blockwise(q_dither, C, g),
    "latent-rtn4": lambda C, g: blockwise(q_rtn, C, 4),
}


def band_norms(X, nb):
    S, D = X.shape
    return X.view(S, nb, D // nb).norm(dim=-1)          # [S, nb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/witcert/models/DeepSeek-V2-Lite"))
    ap.add_argument("--corpus", default=os.path.expanduser("~/witcert/corpus"))
    ap.add_argument("--domains", nargs="*", default=["natural", "needle"])
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    DEV = "cuda"
    S = args.seqlen

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=False).to(DEV).eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    H = cfg.num_attention_heads
    DL = cfg.kv_lora_rank                      # 512
    DR = cfg.qk_rope_head_dim                  # 64
    DN = cfg.qk_nope_head_dim                  # 128
    QD = DN + DR                               # 192
    SAMPLE = list(range(0, L, 4))
    NQ = 8

    # softmax scale 不猜属性:从 SDPA 的 scale kwarg 捕获真值(含 YaRN mscale)
    sm_box = [None]
    print(f"L={L} H={H} latent={DL} rope={DR} nope={DN} 采样层={SAMPLE}", flush=True)

    # W_UK:kv_b_proj.weight [H*(DN+v_head), DL] → 每头前 DN 行是 K 部
    W_UK = {}
    for li in SAMPLE:
        w = model.model.layers[li].self_attn.kv_b_proj.weight.detach()   # [H*(DN+DV), DL]
        DV = w.shape[0] // H - DN
        W_UK[li] = w.view(H, DN + DV, DL)[:, :DN, :].float()             # [H, DN, DL]

    # ---- eager 注意力捕获(post-RoPE q/k 与 scaling,逐位同源)+ latent 捕获 ----
    import transformers.models.deepseek_v2.modeling_deepseek_v2 as _m2
    eager_orig = _m2.eager_attention_forward
    cap_qk = {}
    cap_c = {}
    li_counter = [0]

    def eager_patched(module, query, key, value, attention_mask, scaling, **kw):
        i = li_counter[0]; li_counter[0] += 1
        if i in SAMPLE:
            cap_qk[i] = (query.detach()[0], key.detach()[0])   # [H,S,QD]
            sm_box[0] = float(scaling)
        return eager_orig(module, query, key, value, attention_mask, scaling, **kw)

    hooks = []
    for li in SAMPLE:
        ln = model.model.layers[li].self_attn.kv_a_layernorm
        hooks.append(ln.register_forward_hook(
            (lambda li_: lambda m, i, o: cap_c.__setitem__(li_, o.detach()[0]))(li)))

    gen = torch.Generator().manual_seed(7)
    files = []
    for dom in args.domains:
        files += sorted(glob.glob(os.path.join(args.corpus, dom, "*")))[: args.docs]

    agg = {m: {"tv": [], "bA": [], "bB": [], "viol_A": 0, "viol_B": 0, "n": 0,
               "covB": {str(t): 0 for t in TAUS}, "tanh": [], "viol_tanh": 0}
           for m in QUANT}
    n_docs = 0
    for fp in files:
        ids = tok(open(fp, encoding="utf-8", errors="ignore").read(),
                  return_tensors="pt").input_ids[:, :S]
        if ids.shape[1] < S:
            continue
        cap_qk.clear(); cap_c.clear(); li_counter[0] = 0
        _m2.eager_attention_forward = eager_patched
        try:
            with torch.no_grad():
                model.model(ids.to(DEV))
        finally:
            _m2.eager_attention_forward = eager_orig
        n_docs += 1
        sm = sm_box[0] if sm_box[0] is not None else 1.0 / math.sqrt(QD)
        if n_docs == 1:
            print(f"sm(实测 eager scaling)={sm:.6f} vs 1/√192={1/math.sqrt(QD):.6f}", flush=True)

        qpos = torch.linspace(S // 2, S - 1, NQ).long()
        lnf = math.log(2 * S * len(SAMPLE) * H * NQ / DELTA_REQ)    # δ_loc 联合分配
        for li in SAMPLE:
            q_all, k_all = cap_qk[li]                     # [H,S,QD] bf16
            c = cap_c[li].float()                         # [S,DL]
            k_pe_true = k_all[0, :, DN:].float()          # MQA 共享,取头 0 [S,DR]
            # —— 契约自检:W_UK·c 必须重构出捕获 k 的 nope 段 ——
            k_nope_rec = torch.einsum("hnd,sd->hsn", W_UK[li], c)   # [H,S,DN]
            err = (k_nope_rec - k_all[:, :, :DN].float()).abs().max()
            assert float(err) < 5e-2, f"latent 映射自检失败 layer{li}: max|Δ|={float(err):.4f}"

            for mname, qz in QUANT.items():
                chat, csc = qz(c, gen)
                R = (c - chat)                            # [S,DL] latent 残差
                kpe_hat, kpe_sc = blockwise(q_rtn, k_pe_true, 8)
                Rpe = k_pe_true - kpe_hat
                rb = band_norms(R, NBAND).double()        # [S,B] 无 RoPE,直接可存
                rpb = band_norms(Rpe, NBAND).double()
                k_nope_hat = torch.einsum("hnd,sd->hsn", W_UK[li], chat)
                for hi in range(0, H, 2):                 # 头下采样 ×2
                    q_h = q_all[hi].float()               # [S,QD]
                    u_h = q_h[:, :DN] @ W_UK[li][hi]      # [S,DL] 有效 query
                    qg = q_h[qpos]                        # [NQ,QD]
                    ug = u_h[qpos].double()               # [NQ,DL]
                    lg_ex = (torch.einsum("nd,sd->ns", qg[:, :DN].double(),
                                          k_all[hi, :, :DN].double())
                             + qg[:, DN:].double() @ k_pe_true.double().T) * sm
                    lg_c = (torch.einsum("nd,sd->ns", qg[:, :DN].double(),
                                         k_nope_hat[hi].double())
                            + qg[:, DN:].double() @ kpe_hat.double().T) * sm
                    mask = (torch.arange(S, device=DEV)[None, :]
                            <= qpos.to(DEV)[:, None])
                    tv = 0.5 * (lg_ex.masked_fill(~mask, -1e30).softmax(-1)
                                - lg_c.masked_fill(~mask, -1e30).softmax(-1)).abs().sum(-1)
                    # Tier A:c_t = sm·(Σ_b‖u_b‖‖Δc_b‖ + Σ_b‖q_pe,b‖‖Δk_pe,b‖),log 域 A
                    ub = ug.view(NQ, NBAND, DL // NBAND).norm(dim=-1)     # [NQ,B]
                    qpb = qg[:, DN:].double().view(NQ, NBAND, DR // NBAND).norm(dim=-1)
                    cA = (ub @ rb.T + qpb @ rpb.T) * sm                    # [NQ,S]
                    lgm = lg_c.masked_fill(~mask, float("-inf"))
                    lp = lgm - torch.logsumexp(lgm, -1, keepdim=True)
                    logA = torch.logsumexp(lp + cA, -1).clamp_min(0.0)
                    bA = 0.5 * torch.expm1(2 * logA)
                    # tanh 参考(确定性 δ=0):tanh(max_t cA)
                    btanh = torch.tanh(cA.max(-1).values)
                    # Tier B(仅 dither 档有概率语义;其余记 NaN)
                    if "dither" in mname:
                        var = (csc.double() ** 2) / 12.0                   # [S,DL]
                        varpe = (kpe_sc.double() ** 2) / 12.0
                        sig2 = ((ug ** 2) @ var.T + (qg[:, DN:].double() ** 2) @ varpe.T) \
                               * (sm ** 2)                                  # [NQ,S]
                        uB = (2.0 * sig2 * lnf).sqrt()
                        logA_B = torch.logsumexp(lp + uB, -1).clamp_min(0.0)
                        bB = 0.5 * torch.expm1(2 * logA_B)
                    else:
                        bB = torch.full_like(bA, float("nan"))
                    a = agg[mname]
                    a["n"] += tv.numel()
                    a["tv"] += tv.tolist()
                    a["bA"] += bA.clamp(max=1.0).tolist()
                    a["tanh"] += btanh.tolist()
                    a["viol_A"] += int((tv > bA + 1e-9).sum())
                    a["viol_tanh"] += int((tv > btanh.to(tv.device) + 1e-9).sum())
                    if "dither" in mname:
                        a["bB"] += bB.clamp(max=1.0).tolist()
                        a["viol_B"] += int((tv > bB + 1e-9).sum())
                        for t in TAUS:
                            a["covB"][str(t)] += int((bB <= t).sum())
            del R, Rpe, k_nope_hat
            torch.cuda.empty_cache()
        print(f"{os.path.basename(fp)} 完成", flush=True)

    for h_ in hooks:
        h_.remove()
    import statistics as st
    res = {"model": os.path.basename(args.model), "S": S, "nband": NBAND,
           "delta_req": DELTA_REQ, "sm": sm_box[0], "docs": n_docs,
           "sampled_layers": SAMPLE, "head_stride": 2, "schemes": {}}
    print(f"\n=== P50 latent 尺子(V2-Lite,S={S},{n_docs} 文档,契约自检通过)===")
    for m, a in agg.items():
        fin = [x for x in a["bA"] if x < 1.0]
        row = {"n": a["n"], "viol_A": a["viol_A"], "viol_tanh": a["viol_tanh"],
               "tv_median": st.median(a["tv"]) if a["tv"] else None,
               "bA_median": st.median(a["bA"]) if a["bA"] else None,
               "bA_informative_frac": len(fin) / max(1, a["n"])}
        if a["bB"]:
            row["viol_B"] = a["viol_B"]
            row["bB_median"] = st.median(a["bB"])
            row["covB"] = {t: v / a["n"] for t, v in a["covB"].items()}
        res["schemes"][m] = row
        print(m, json.dumps(row, ensure_ascii=False))
    fn = f"p50_latent_v2lite{args.tag}.json"
    json.dump(res, open(os.path.join(OUT, fn), "w"), ensure_ascii=False, indent=1)
    print("saved", fn)


if __name__ == "__main__":
    main()
