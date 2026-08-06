# -*- coding: utf-8 -*-
"""尺子的数学:纯张量函数,不知道任何 hook / 模型 / serving 栈的事。

三类记忆对象对应三个 meter,都只吃张量、吐标量字典:

  · `band_witness`   —— 带范数见证(LatentKV / KV):W = Σ_b ‖Δ_b‖ ≥ ‖Δ‖,
    Tier A 的 sound 上界。返回真实残差、见证、保守度 W/‖Δ‖。
  · `topk_margin`    —— 稀疏选择边界(SparseSelector):m = s_(r) − s_(r+1);
    m > 2ε ⇒ 该 top-r 集合在 |Δs| ≤ ε 的扰动下不可能改变。
  · `gate_contraction` —— 递归状态收缩(RecurrentState):a_t 分布与半衰期。

**为什么单独成文件**:这些数学在 GQA / MLA latent / V4 压缩条目 / DSA / C4 / KDA
上是同一套;此前每个 apply_*_patch.py 各自抄一遍,抄错一次就是一次假结果。
"""
from __future__ import annotations

import math
import os

import torch


def _acc(dst: dict, key: str, v: torch.Tensor) -> None:
    """把一批标量并进累加槽(均值/极值/计数)。

    **累加量留在设备上**:`float(t)` / `int(t)` 每次都是一次 GPU→CPU 同步,
    把流水线串行化。实测每次调用十几个这样的同步会让 serving 慢一倍,且与采样
    行数无关(2026-07-30 p60 首测 +100%)。同步一律推迟到 finalize/落盘时,
    每 FLUSH 次才发生一次。
    """
    # **计数也必须在设备上**(2026-07-31,B3):CUDA graph 只重放 kernel 序列,
    # Python 语句只在 capture 时执行一次。若 n 是 python int,开图后它永远停在
    # capture 时的值,而 sum/min/max 却在继续累加 —— 均值会被系统性地算大。
    s = dst.get(key)
    if s is None:
        s = dst[key] = {"n": v.new_zeros((), dtype=torch.long),
                        "sum": v.new_zeros(()),
                        "min": v.new_full((), float("inf")),
                        "max": v.new_full((), float("-inf"))}
    elif not isinstance(s["n"], torch.Tensor):          # 兼容旧结构
        s["n"] = v.new_full((), int(s["n"]), dtype=torch.long)
    s["n"] += v.numel()
    s["sum"] += v.sum()
    # **每次执行 +1 的设备侧调用计数**(2026-07-31,B3 收尾):`n` 数的是元素数,
    # 单张快照分不清"1 次调用采 128 行"和"1 次调用 + 127 次重放各采 1 行" ——
    # 我用 n/_n_calls 当活性判据时就被这个比值骗过(120× 看着很像有重放,其实没有)。
    # 这个计数器是张量自增,会被捕进图并在重放时执行,故
    # **dev_calls > python 侧 _n_calls 严格等价于"重放期确实执行过探针"**。
    if "calls" not in s:
        s["calls"] = v.new_zeros((), dtype=torch.long)
    s["calls"] += 1
    torch.minimum(s["min"], v.min(), out=s["min"])
    torch.maximum(s["max"], v.max(), out=s["max"])


def _bump(dst: dict, key: str, v: torch.Tensor) -> None:
    """整数计数器同样留在设备上(v 为 bool 张量,累加其 sum)。"""
    cur = dst.get(key)
    dst[key] = v.sum() if cur is None else cur + v.sum()


def _num(x):
    """落盘时才做的一次性同步。"""
    return x.item() if isinstance(x, torch.Tensor) else x


#: 见证量化档位。**带图常开时这是主要成本旋钮**(B6,2026-07-31 实测):
#: 开图后成本与采样率无关而与**每次执行的 kernel 数**成正比 —— 每个插桩点约 3.5% 吞吐,
#: 与被插桩的层数近似成正比。少算一个量化档就少一串 kernel。
#: 用 WITCERT_PROBE_QUANTIZERS="int8" 只留一档;留空则两档全上。
_QUANT_BITS = {"int8": 8, "int4": 4}


def _quantizers():
    raw = os.environ.get("WITCERT_PROBE_QUANTIZERS", "").strip()
    if not raw:
        return (("int8", 8), ("int4", 4))
    picked = [(n, _QUANT_BITS[n]) for n in raw.replace(" ", "").split(",")
              if n in _QUANT_BITS]
    return tuple(picked) or (("int8", 8), ("int4", 4))


def band_witness(x: torch.Tensor, bands: int, quantizers=None,
                 out: dict | None = None) -> dict:
    """带范数见证:对 x[n, D] 逐带量化,记 ‖Δ‖ / W=Σ_b‖Δ_b‖ / W/‖Δ‖ / 相对残差。

    W ≥ ‖Δ‖ 是三角不等式的直接推论,故 `viol_*` 恒应为 0 —— 它不是科学结论,
    是**实现自检**:非零即说明分带或范数算错了。

    quantizers=None 时按 WITCERT_PROBE_QUANTIZERS 取(默认两档)。
    """
    if quantizers is None:
        quantizers = _quantizers()
    out = {} if out is None else out
    n, D = x.shape
    xn = x.norm(dim=-1)
    _acc(out, "x_norm", xn)
    if D % bands:
        out["bands_skipped_dim"] = out.get("bands_skipped_dim", 0) + 1
        return out
    xb = x.view(n, bands, D // bands)
    for name, bits in quantizers:
        qmax = 2 ** (bits - 1) - 1
        scale = xb.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        qb = torch.round(xb / scale).clamp(-qmax - 1, qmax) * scale
        db = xb - qb
        wit = db.norm(dim=-1).sum(-1)
        true = db.reshape(n, -1).norm(dim=-1)
        _acc(out, f"wit_{name}", wit)
        _acc(out, f"true_{name}", true)
        _acc(out, f"rel_{name}", true / xn.clamp_min(1e-12))
        _acc(out, f"tight_{name}", wit / true.clamp_min(1e-12))
        _bump(out, f"viol_{name}", wit < true * (1 - 1e-5))
    return out


def qnorm(x: torch.Tensor, scale: float, out: dict | None = None) -> dict:
    """注意力查询的**缩放范数** scale·‖q_head‖ —— selector→attention 桥的运行时系数。

    为什么是它:score 对存储扰动的敏感度由 Cauchy–Schwarz 给出
        |⟨q, Δk⟩|·scale ≤ scale·‖q‖·‖Δk‖
    右边第一个因子就是本函数采的分布;第二个因子是 band_witness 已采的绝对见证 W。
    两者相乘即 selector_score:linf 的 ε —— 桥的系数从此**运行时可算**,
    不再是论文里"折 TV 需逐请求 q 范数"那句欠账。
    """
    out = {} if out is None else out
    _acc(out, "q_scaled_norm", x.norm(dim=-1).reshape(-1) * scale)
    return out


def sel_bridge(m_attn: torch.Tensor, m_sel: torch.Tensor, out: dict | None = None) -> dict:
    """selector→attention 经验桥的逐行观测:同一批 (token, 选择集) 上,
    被选择**遗漏**的质量分别按两种分布计:

      m_attn = Σ_{loc∉选中} softmax(scale·q·k 全候选)     —— 真实注意力质量口径
      m_sel  = Σ_{loc∉选中} softmax(索引 logits 全候选)   —— 选择器分数质量口径

    关键恒等式:对"限制到选中集并重归一化"的分布,TV(全集, 选中集) **恰等于**遗漏质量
    (两侧各贡献 m/2)。故 m_attn 就是选择段在 attn_dist:tv 里的**直接测量**,
    不再需要那座"不能由形式化凭空补出"的换算桥 —— 桥被测量取代,档位如实标 empirical。
    m_sel 同时记下,用于回答"选择器分数质量是不是注意力质量的好代理"。
    """
    out = {} if out is None else out
    _acc(out, "m_out_attn", m_attn)
    _acc(out, "m_out_sel", m_sel)
    _acc(out, "bridge_ratio", m_attn / m_sel.clamp_min(1e-9))
    return out


def topk_margin(logits: torch.Tensor, valid: torch.Tensor, topk: int,
                ranks=(1, 8, 64), etas=(1e-3, 1e-2), out: dict | None = None) -> dict:
    """稀疏选择边界余量与翻转认证。

    `valid` 是**必需**参数而非可选:上游索引 logits 一律以 clean_logits=False 产出,
    有效区间外是未初始化显存;不屏蔽就会把 1e38 量级的垃圾选进 top-k。
    只统计有效列数 > topk 的行(否则 top-k 即全选,margin 无定义)。
    """
    out = {} if out is None else out
    n, S = logits.shape
    if S <= topk + 1:
        out["skipped_short"] = out.get("skipped_short", 0) + 1
        return out
    # 掩码必须逐行对齐 logits。[1,S] 之类可广播的显式 expand;不兼容则让 torch 抛错 ——
    # 形状对不上却静默按错行屏蔽,正是这里要防的那类 bug。
    if valid.shape != logits.shape:
        valid = valid.expand_as(logits)
    keep = valid.sum(-1) > (topk + 1)
    _bump(out, "rows_trivial", ~keep)
    # keep.any() 会同步;这里用行数上界代替提前返回,空批时后续 topk 自然是空张量
    lg, valid = logits[keep], valid[keep]
    n = lg.shape[0]
    if n == 0:
        return out
    lg = torch.where(valid, lg, torch.full_like(lg, -1e30))
    vals, idx = lg.topk(topk + 1, dim=-1)
    margin = (vals[:, topk - 1] - vals[:, topk]).clamp_min(0)
    out["rows"] = out.get("rows", 0) + n
    _acc(out, "margin", margin)
    _acc(out, "score_abs", lg.masked_fill(~valid, 0).abs())
    topk_set = idx[:, :topk]
    for r in ranks:
        mr = (vals[:, r - 1] - vals[:, r]).clamp_min(0)
        _acc(out.setdefault("margin_r", {}), str(r), mr)
    for eta in etas:
        eps = eta * lg.gather(1, topk_set).abs().amax(-1, keepdim=True)
        cert = (margin[:, None] > 2 * eps).squeeze(-1)
        # 扰动也不能用 RNG(capture 会失效)。用与位置相关的确定性伪随机:
        # sin 的高频相位在 [-1,1] 上足够铺开,且逐位置固定 —— 重复测量可复现。
        _ph = torch.arange(lg.numel(), device=lg.device, dtype=lg.dtype).reshape(lg.shape)
        _u = torch.sin(_ph * 12.9898 + 78.233) * 43758.5453
        _u = (_u - _u.floor()) * 2 - 1                     # ∈ [-1,1)
        pert = torch.where(valid, lg + _u * eps, torch.full_like(lg, -1e30))
        a = torch.zeros_like(lg, dtype=torch.bool).scatter_(1, topk_set, True)
        b = torch.zeros_like(lg, dtype=torch.bool).scatter_(1, pert.topk(topk, dim=-1).indices, True)
        flip = (a ^ b).any(-1)
        k_ = f"{eta:g}"
        for r in ranks:
            mr = (vals[:, r - 1] - vals[:, r]).clamp_min(0)
            cr = (mr[:, None] > 2 * eps).squeeze(-1)
            ar = torch.zeros_like(lg, dtype=torch.bool).scatter_(1, idx[:, :r], True)
            br = torch.zeros_like(lg, dtype=torch.bool).scatter_(1, pert.topk(r, dim=-1).indices, True)
            fr = (ar ^ br).any(-1)
            kk = f"{eta:g}/r{r}"
            for fld, val in (("cert_r", cr), ("flip_r", fr), ("flip_in_cert_r", fr & cr)):
                _bump(out.setdefault(fld, {}), kk, val)
        # Lemma S2 的两项:被**换出**条目的注意力质量,与被**换入**条目的质量。
        # TV(p,p̂) ≤ ½(m_out + m_in) + 公共集上的扰动项 —— 故选择段的契约是
        # **加性**的 (a_S,b_S)=(1, ½(m_out+m_in)),它不放大上游误差,只添一项。
        # 只量 m_out 不够:换入项同样进上界(2026-07-31 推 S2 时才发现)。
        idx2 = pert.topk(topk, dim=-1)
        p = torch.softmax(vals[:, :topk], dim=-1)
        lost = (p * (a & ~b).gather(1, topk_set).float()).sum(-1)
        p_hat = torch.softmax(idx2.values, dim=-1)
        gain = (p_hat * (b & ~a).gather(1, idx2.indices).float()).sum(-1)
        _acc(out.setdefault("lost_mass", {}), k_, lost)
        _acc(out.setdefault("gain_mass", {}), k_, gain)
        _acc(out.setdefault("sel_bS", {}), k_, 0.5 * (lost + gain))   # b_S 本身
        for fld, val in (("cert", cert), ("flip", flip), ("flip_in_cert", flip & cert)):
            _bump(out.setdefault(fld, {}), k_, val)
    return out


def gate_contraction(a_t: torch.Tensor, beta: torch.Tensor | None = None,
                     edges=(0.0, 0.5, 0.9, 0.99, 0.999, 1.01),
                     out: dict | None = None) -> dict:
    """递归状态收缩因子分布。

    关键不是均值而是**近 1 尾部**:‖ΔS_t‖ ≤ a_t‖ΔS_{t−1}‖ + b_t‖Δx_t‖ 的
    worst-case 界由 max a_t 支配,a_t→1 的通道使 uniform 界空洞(1289× 教训的状态版)。
    故直方图最后一桶 P(a≥0.999) 与 max 必须与均值一起报。
    """
    out = {} if out is None else out
    a = a_t.reshape(-1).clamp(0.0, 1.0)
    _acc(out, "a_t", a)
    # 直方图也留在设备上:一次 bucketize + bincount,替代逐桶 int(...) 同步
    e = torch.tensor(edges[1:-1], device=a.device, dtype=a.dtype)
    cnt = torch.bincount(torch.bucketize(a, e), minlength=len(edges) - 1)[:len(edges) - 1]
    out["a_hist"] = cnt if out.get("a_hist") is None else out["a_hist"] + cnt
    if beta is not None:
        _acc(out, "beta", beta.reshape(-1).abs())
    # **长期稳定性的判据是 Σ log a_t < 0(乘积),不是 ā < 1(均值)**:
    # e_t ≤ (Π a_i)·e_0 + Σ_s (Π_{i>s} a_i)·b_s,决定衰减的是对数和。
    # 均值口径会把"少数通道 a→1"这件事平均掉 —— 那正是界会空洞的地方。
    # 故直接累加 log a(下截断防 -inf;a=0 的通道本就完全遗忘,不影响稳定性判据)。
    _acc(out, "log_a", a.clamp_min(1e-12).log())
    return out


def pool_dominance(scores: torch.Tensor, block: int, out: dict | None = None) -> dict:
    """池化归因(**经验观测,不是证书**)。

    压缩类方案把连续 `block` 个 token 按门控分数池成一条条目。这里回答:
    这次池化里**信息是均摊还是被单点主导**?

    **所有指标必须跨 block 可比** —— V4 同一个模型里就有 block=4 与 block=128 两种
    粒度(compress_ratio 是 Literal[0,4,128]),裸的 top1_share 在两者间差一个数量级,
    混在一起比会得出"某些层极度均摊"的假结论(2026-07-31 实测踩到)。故:

      · top1_norm  (top1 − 1/block) / (1 − 1/block) ∈ [0,1]:0=完全均摊,1=完全主导;
      · entropy_n  块内熵 / log(block) ∈ [0,1];
      · eff_frac   有效块大小 exp(H)/block ∈ (0,1]:这条压缩条目"实际装了几个 token"
        占名义块大小的比例。
      · **w_l2**    ‖w‖_2 ∈ [1/√block, 1] —— **池化契约的系数本身**:
        块内 token 各带误差 Δ_j 时 ‖Σ_j w_j Δ_j‖ ≤ ‖w‖_2·(Σ_j‖Δ_j‖²)^{1/2}(Cauchy–Schwarz)。
        即池化不是误差源而是**衰减器**,衰减多少正好由这个量给出(见 papers/p2/math_spec.md §1.2)。
        top1_share 只能给出它的下界,区间太宽,故必须直接量。

    不用 max/mean 之类的比值:门控分数是带符号 logit,均值过零时比值会爆到 1e4 量级,
    该指标本身没有定义域保证。

    自检不变量:softmax 后 top1 ≥ 1/block。违反即分带或 reshape 算错,记 viol_pool。
    """
    out = {} if out is None else out
    s = scores.detach().float()
    s = s.reshape(-1, s.shape[-1]) if s.dim() > 2 else s
    n, D = s.shape
    if block <= 1 or n < block:
        out["pool_skipped"] = out.get("pool_skipped", 0) + 1
        return out
    nb = n // block
    b = s[: nb * block].reshape(nb, block, D)
    w = torch.softmax(b, dim=1)
    top1 = w.max(dim=1).values                      # [nb, D],数学下界 1/block
    lo = 1.0 / block
    ent_nats = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(1)
    _acc(out, "top1_norm", (top1 - lo) / (1.0 - lo))
    _acc(out, "w_l2", w.pow(2).sum(1).sqrt())      # 池化契约系数 a_pool
    _acc(out, "entropy_n", ent_nats / math.log(block))
    _acc(out, "eff_frac", torch.exp(ent_nats) / block)
    _bump(out, "viol_pool", top1 < lo - 1e-4)
    prev = out.get("block")
    if prev is not None and prev != block:
        # 同一 slot 里块大小变了 —— 分层键出问题或上游动态改 ratio,必须显式暴露
        out["block_mixed"] = out.get("block_mixed", 0) + 1
    out["block"] = block
    return out


def half_life(a_mean: float) -> float | None:
    """按均值口径的误差半衰期(步);ā≥1 时无收缩,返回 None。"""
    return math.log(2) / (-math.log(a_mean)) if 0 < a_mean < 1 else None


def finalize(slot: dict) -> dict:
    """把累加槽换算成可读均值,直方图归一化。**唯一发生 GPU→CPU 同步的地方。**"""
    def _slotify(v):
        n = int(_num(v["n"]))
        o = {"mean": _num(v["sum"]) / max(1, n), "min": _num(v["min"]),
             "max": _num(v["max"]), "n": n}
        # **设备侧执行计数必须落进产物**:验收门判图内活性只认它。
        # 序列化只输出固定字段,新加的累加量不显式带出来就等于没加(踩过一轮)。
        if "calls" in v:
            o["calls"] = int(_num(v["calls"]))
        return o
    o = {}
    for k, v in slot.items():
        # `_` 开头且是 dict 的是**内部状态**(如融合算子的裸累加缓冲),不是指标 ——
        # 走进下面的分支会当成计数器字典去 .item(),对多元素张量直接报错
        if k.startswith("_") and isinstance(v, dict):
            continue
        if isinstance(v, dict) and "sum" in v and "n" in v and "min" in v:
            o[k] = _slotify(v)
        elif isinstance(v, dict) and v and all(isinstance(x, dict) and "sum" in x
                                               for x in v.values()):
            o[k] = {kk: _slotify(x) for kk, x in v.items()}
            o[k + "_mean"] = {kk: o[k][kk]["mean"] for kk in o[k]}
        elif isinstance(v, dict):                      # 计数器字典(cert_r / flip_r …)
            o[k] = {kk: _num(x) for kk, x in v.items()}
        elif k == "a_hist":
            h = [_num(x) for x in v]
            tot = max(1, sum(h)); o[k] = [x / tot for x in h]
        else:
            o[k] = _num(v)
    return o
