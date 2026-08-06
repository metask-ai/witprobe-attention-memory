# -*- coding: utf-8 -*-
"""补桥后的合法组合(p76):存储见证 → 选择器分数 → 注意力 TV,全程同度量对接。

**背景**:度量类型检查拒绝了 p64 的链(相对见证 + TV 相加不属任何度量,p64 已 superseded)。
一审重建把三段接成一条链;**二审又拆下来一段** —— b_S 的 m_out/m_in 来自
softmax(索引 logits),度量是 selector_dist:tv 而非 attn_dist:tv,一审的 attn_dist
标签是手写错的。现在的结构:

  窗口存储链(合法,进链):
    ① 打分桥  |Δscore| ≤ (scale·‖q‖)·‖Δk‖      score_bridge(Cauchy–Schwarz)
    ② softmax TV ≤ ½(e^{2ε}−1)                   softmax_tv_bridge(tv_le_eform 实例化)
  选择段(**单独报,不相加**):
    b_S,selector_dist:tv 度量;折注意力质量需 selector→attention 经验桥(尚不存在)

口径(随数字引用,一条都不能省):
  · **只覆盖窗口条目(swa 池)的分数扰动**:q 与窗口 K 共享同一注意力,配对合法;
    被选中的压缩页条目自身的量化误差未见证,不在界内(A6 待测)。
  · scale·‖q‖ 与 W 都是**采样分布统计量**(43 层 × 8 rank),非逐请求上确界。
  · ‖Δk‖ 用绝对见证 W(二次量化口径):真实存储误差无运行时参照,W 是可算代理。
  · Cauchy–Schwarz 取 Δk 与 q 对齐的最坏情形 —— sound 但悲观,这是证书的代价。
  · 逐层 TV 是该层注意力分布上的界;跨层求和是聚合预算,不是端到端输出距离。

python3 experiments/p76_bridged_composition.py
"""
import glob
import json
import math
import os
import statistics
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from witcert.probe import contracts as C          # noqa: E402


def collect(ranks):
    """全 rank 聚合:每层取 max(soundness 要求)与 mean(典型值)。"""
    qn_max, qn_mean, wt_max, wt_mean = {}, {}, {}, {}
    for f in ranks:
        d = json.load(open(f))
        for k, v in d["slots"].items():
            layer = int(k.split("|")[1][1:])
            if k.endswith("|qn"):
                s = v["q_scaled_norm"]
                qn_max[layer] = max(qn_max.get(layer, 0.0), s["max"])
                qn_mean.setdefault(layer, []).append(s["mean"])
            elif "swa_norm_rope" in k and "wit_int8" in v:
                s = v["wit_int8"]
                wt_max[layer] = max(wt_max.get(layer, 0.0), s["max"])
                wt_mean.setdefault(layer, []).append(s["mean"])
    qn_mean = {L: statistics.mean(v) for L, v in qn_mean.items()}
    wt_mean = {L: statistics.mean(v) for L, v in wt_mean.items()}
    return qn_max, qn_mean, wt_max, wt_mean


def main():
    ranks = sorted(glob.glob(os.path.join(OUT, "wc_qnorm.json.rank*")))
    assert len(ranks) == 8, f"预期 8 个 rank,实得 {len(ranks)}"
    qn_max, qn_mean, wt_max, wt_mean = collect(ranks)
    att = json.load(open(os.path.join(OUT, "p55_v4flash_attribution.json")))
    b_S = att["select_path"]["b_S"]["0.001"]
    cert = att["select_path"]["cert_rate_set"]

    # 最大值口径的链(sound):类型检查在 Chain.then 里执行
    ch = C.v4_bridged_chain(qn_max, wt_max)
    tot = ch.compose()
    sel = C.selection_contract(b_S, cert)
    # **自检**:选择段必须被类型系统拒绝 —— 若有一天它能接上,说明有人偷改了度量标签
    try:
        C.v4_bridged_chain(qn_max, wt_max).then(sel)
        raise SystemExit("选择段接上了注意力链 —— 度量标签被改动,本产物作废")
    except C.MetricMismatch:
        pass
    # 逐层数字用**精确形** ½(e^{2ε}−1)(softmax_tv_bridge 直接给出,已证):
    # 仿射松弛 e^{2ε₀}·ε 只在"必须以仿射系数进链"时才需要 —— 逐层标量报告
    # 不受此限,用松弛形是白白放大(全局 ε₀ 一项就放大 ~4×,踩过)。
    per_layer = {}
    eps0 = max(qn_max[L] * wt_max[L] for L in qn_max)
    tv_exact = lambda e: 0.5 * (math.exp(2 * e) - 1)      # 不加 b_S:度量不同,不相加
    for L in sorted(qn_max):
        e_w = qn_max[L] * wt_max[L]
        e_t = qn_mean[L] * wt_mean[L]
        per_layer[L] = {
            "qn_max": qn_max[L], "wit_max": wt_max[L],
            "eps_worst": e_w, "eps_typ": e_t,
            "tv_worst": tv_exact(e_w),
            "tv_typ": tv_exact(e_t),
        }
    tvs_w = [v["tv_worst"] for v in per_layer.values()]
    tvs_t = [v["tv_typ"] for v in per_layer.values()]
    nv_w = sum(1 for t in tvs_w if t < 1.0)
    nv_t = sum(1 for t in tvs_t if t < 1.0)

    rep = {
        "what": "补桥后的合法组合:存储见证 -> 打分桥 -> softmax 桥 -> 选择段(全部 attn_dist:tv)",
        "machine": "hgx",
        "stack": att.get("stack", "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8"),
        "caliber": [
            "scale·‖q‖ 为采样分布最大值(43 层 × 8 rank),非逐请求上确界",
            "‖Δk‖ 用绝对见证 W(二次量化口径),真实存储误差无运行时参照",
            "Cauchy–Schwarz 取 Δk 与 q 对齐的最坏情形 —— sound 但悲观,这是证书的代价",
            "逐层 TV 是该层注意力分布上的界;跨层求和是聚合预算,不是端到端输出距离",
        ],
        "chain": {
            "stages": [{"name": c.name, "a": c.a, "b": c.b, "tier": c.tier,
                        "m": f"{c.m_in} -> {c.m_out}", "proof": c.proof}
                       for c in ch.stages],
            "composed": {"a": tot.a, "b": tot.b, "tier": tot.tier,
                         "m": f"{tot.m_in} -> {tot.m_out}"},
            "eps0": eps0,
        },
        "selection_separate": {
            "name": sel.name, "b_S": b_S, "m": f"{sel.m_in} -> {sel.m_out}",
            "tier": sel.tier,
            "note": ("**不与窗口存储链相加**:b_S 的质量是 softmax(索引 logits) 上的,"
                     "折注意力质量需 selector→attention 经验桥(尚不存在);"
                     "类型系统拒绝该组合已由本脚本自检"),
        },
        "per_layer": per_layer,
        "summary": {
            "n_layers": len(per_layer),
            "tv_worst_median": statistics.median(tvs_w),
            "tv_worst_max": max(tvs_w),
            "tv_typ_median": statistics.median(tvs_t),
            "b_S": b_S,
            "qn_max_all": max(qn_max.values()), "wit_max_all": max(wt_max.values()),
            "request_budget_sum_worst": sum(tvs_w),
            "request_budget_sum_typ": sum(tvs_t),
            "n_nonvacuous_worst": nv_w, "n_nonvacuous_typ": nv_t,
        },
    }
    s = rep["summary"]
    # **头条按数据写,不预设结论**(报告文案硬编码过相反方向,踩过)
    if nv_t == len(per_layer):
        _shape = ("typical 口径 43/43 层非空洞(<1),worst 口径 %d/43 层非空洞" % nv_w)
    else:
        _shape = ("typical 口径仅 %d/43 层非空洞,worst 口径 %d/43 层" % (nv_t, nv_w))
    rep["findings"] = {
        "0_headline": (
            "**桥接后的界(窗口存储项),如实报**:逐层注意力 TV 界(精确形 ½(e^{2ε}−1))"
            "中位 %.3f(worst)/ %.3f(typical);%s。"
            "43 层求和 %.1f(worst)>> 1 —— 请求级聚合预算空洞。"
            "**这是有证书的界的真实代价** —— 被类型检查废掉的 0.0231/0.982 连界都不是。"
            "选择段 b_S=%.5f 在 selector_dist:tv 度量,**单独报不相加**(经验桥待建);"
            "压缩页条目的量化项未见证(A6)"
            % (s["tv_worst_median"], s["tv_typ_median"], _shape,
               s["request_budget_sum_worst"], b_S)),
        "1_bridge_coeff": (
            "桥接系数 scale·max‖q‖ = %.4f —— q 过 rmsnorm 后 scale·‖q‖ ≈ 1,"
            "打分桥**几乎不放大**;界的主导项是绝对见证 W(全层最大 %.4f),"
            "即二次量化的逐条目误差本身" % (s["qn_max_all"], s["wit_max_all"])),
        "2_typed_chain": (
            "链条 %s,tier=%s;每段 proof 字段指向已证 Lean 定理 —— "
            "这是类型检查拒绝原链之后的合法重建,不是原数字的换皮"
            % (rep["chain"]["composed"]["m"], rep["chain"]["composed"]["tier"])),
        "3_open_terms": (
            "**界外的两项,缺一不可地要说**:①压缩页条目的量化误差未见证 —— "
            "softmax 桥要求全体 logits 的 ℓ∞,本界只覆盖窗口侧,页侧视为精确(A6 补测);"
            "②b_S(%.5f)在 selector_dist:tv,折注意力质量需 selector→attention "
            "**经验桥** —— 那是相关性测量,不能由形式化凭空补出" % b_S),
        "4_where_to_tighten": (
            "要把窗口项收紧一个量级:①W(更紧的逐带见证或降带宽)是主导项;"
            "②Cauchy–Schwarz 对齐最坏情形(改逐请求实采 |q·Δk| 分布可换经验档)"),
    }
    dst = os.path.join(OUT, "p76_bridged_composition.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print("组合:", rep["chain"]["composed"])
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
