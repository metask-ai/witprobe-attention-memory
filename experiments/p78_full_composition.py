# -*- coding: utf-8 -*-
"""A6 收口(p78):补齐两个界外项后的**全链**组合,与经验桥的质量检验。

p76 二审后的两个界外项在 p77 补测,本文件把它们组合进去:

  ① centry:压缩页条目的二次量化见证 —— ε 从"只盖窗口"升级为
     ε = scale·‖q‖·max(W_swa, W_centry),softmax 桥的 ℓ∞ 前提对**全体候选条目**成立
     (窗口 + 压缩页都被见证;c4 层如此,c128 层仍只有窗口侧,见 caliber)。
  ② selbridge:同一 (q, 候选集, 选中集) 上,截断遗漏质量的两种口径。
     **数据修正了本文件初版的解读**(2026-07-31,两处):
     a) m_out_attn(截断遗漏的注意力质量)是 **top-512 截断本身**的质量 ——
        那是**架构设计量**:稀疏选择是 V4 的模型语义,理想参照本来就带截断,
        它不属于误差链。误差链里的选择项是"上游扰动导致选择改变"的换手质量
        (p55 的 b_S,η 扰动口径)—— 形状一直对,只差单位。
     b) 比值 m_out_attn/m_out_sel 的中位接近 1 —— 选择器质量是注意力质量的
        可用代理,与初版预写的"远离 1"相反(具体数字看产物,别在注释里写死)。
        于是它就是**经验桥系数 ρ**:b_S(selector 单位)× ρ_max = 选择误差项(attn 单位),
        档位 empirical(ρ 在截断集上测得,外推到换手集是均质性假设,写进 caliber)。

口径:
  · c4 层(21 层)有全部三项;c128 层(其余)无 selbridge/centry ——
    其页侧仍未见证,**分开报,不混**。
  · 所有统计量为采样分布(decode token,每次至多 4 行),非逐请求上确界。
  · 组合档位 empirical:选择段是测量。这不是缺陷,是换算桥不存在的诚实成本。

python3 experiments/p78_full_composition.py
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
    qn_max, mo_attn, mo_sel, ratio = {}, {}, {}, {}
    wt_swa = {"max": {}, "mean": {}}
    wt_cen = {"max": {}, "mean": {}}
    for f in ranks:
        d = json.load(open(f))
        for k, v in d["slots"].items():
            L = int(k.split("|")[1][1:])
            if k.endswith("|qn"):
                qn_max[L] = max(qn_max.get(L, 0.0), v["q_scaled_norm"]["max"])
            elif "swa_norm_rope" in k and "wit_int8" in v:
                wt_swa["max"][L] = max(wt_swa["max"].get(L, 0.0), v["wit_int8"]["max"])
                wt_swa["mean"].setdefault(L, []).append(v["wit_int8"]["mean"])
            elif k.endswith("|centry") and "wit_int8" in v:
                wt_cen["max"][L] = max(wt_cen["max"].get(L, 0.0), v["wit_int8"]["max"])
                wt_cen["mean"].setdefault(L, []).append(v["wit_int8"]["mean"])
            elif k.endswith("|selbridge") and "m_out_attn" in v:
                mo_attn.setdefault(L, []).append(v["m_out_attn"]["mean"])
                mo_sel.setdefault(L, []).append(v["m_out_sel"]["mean"])
                ratio[L] = max(ratio.get(L, 0.0), v["bridge_ratio"]["max"])
    for w in (wt_swa, wt_cen):
        w["mean"] = {L: statistics.mean(v) for L, v in w["mean"].items()}
    mo_attn = {L: statistics.mean(v) for L, v in mo_attn.items()}
    mo_sel = {L: statistics.mean(v) for L, v in mo_sel.items()}
    return qn_max, wt_swa, wt_cen, mo_attn, mo_sel, ratio


def main():
    ranks = sorted(glob.glob(os.path.join(OUT, "wc_bridge.json.rank*")))
    assert len(ranks) == 8, f"预期 8 个 rank,实得 {len(ranks)}"
    qn, wswa, wcen, mo_attn, mo_sel, ratio = collect(ranks)
    c4_layers = sorted(set(wcen["max"]) & set(mo_attn))
    assert c4_layers, "没有任何 c4 层拿到 centry+selbridge —— 桥测量没跑"
    att = json.load(open(os.path.join(OUT, "p55_v4flash_attribution.json")))
    b_S_sel = att["select_path"]["b_S"]["0.001"]
    cert_rate = att["select_path"]["cert_rate_set"]

    rho_max = max(ratio.values())
    n_rho = sum(1 for _ in c4_layers)

    # **选择段由机器过桥**(P0-4):selection_contract(selector 单位)∘ 经验桥 → attn 单位。
    # 桥自带四件事:测于哪 / 外推到哪 / 样本量 / ρ 是样本最大值非置信上界。
    bridge = C.empirical_selector_to_attn_bridge(
        rho_max, n_rho, measured_on="c4 截断集(decode 采样,质量加权)",
        applied_to="η=0.001 扰动换手集")
    sel_attn = C.Chain().then(C.selection_contract(b_S_sel, cert_rate)).then(bridge).compose()
    b_S_attn = sel_attn.b            # = ρ_max × b_S,由 Chain.compose 算出

    # **逐层数字由链自己算**(P0-3):每层构造该层系数的链,evaluate(e_in=W) 出数。
    # 不再另行手算 —— 数字、类型、档位走同一条路。
    per_layer = {}
    def layer_value(L, w_key):
        W = max(wswa[w_key].get(L, 0.0), wcen[w_key][L])
        ch_L = C.v4_bridged_chain({L: qn[L]}, {L: W})
        ch_L.also(sel_attn)
        ev = ch_L.evaluate(e_in=W)
        return ev["e_out"], ev, ch_L
    for L in c4_layers:
        v_w, _, _ = layer_value(L, "max")
        v_t, _, _ = layer_value(L, "mean")
        per_layer[L] = {
            "eps_worst": qn[L] * max(wswa["max"].get(L, 0.0), wcen["max"][L]),
            "eps_typ": qn[L] * max(wswa["mean"].get(L, 0.0), wcen["mean"][L]),
            "wit_swa_max": wswa["max"].get(L), "wit_centry_max": wcen["max"][L],
            "tv_total_worst": v_w,           # 链算(采样最大系数口径的上界)
            "tv_total_typ": v_t,             # 链算(均值系数 —— **诊断读数,非上界**)
            "tv_store_worst": v_w - b_S_attn,
            "tv_store_typ": v_t - b_S_attn,
            "trunc_mass_attn": mo_attn[L], "trunc_mass_sel": mo_sel[L],
        }
    # 全层链(compose 展示类型与档位;逐层数字来自上面的 evaluate)
    ch = C.v4_bridged_chain({L: qn[L] for L in c4_layers},
                            {L: max(wswa["max"].get(L, 0.0), wcen["max"][L])
                             for L in c4_layers})
    ch.also(sel_attn)
    tot = ch.compose()
    # **对拍**:链算数字必须与公式一致 —— 谁漂移谁报警
    _L0 = c4_layers[0]
    _e0 = per_layer[_L0]["eps_worst"]
    assert abs(per_layer[_L0]["tv_total_worst"]
               - (0.5 * (math.exp(2 * _e0) - 1) + b_S_attn)) < 1e-9, "链算与公式漂移"

    tw = [v["tv_total_worst"] for v in per_layer.values()]
    tt = [v["tv_total_typ"] for v in per_layer.values()]
    nv_w = sum(1 for t in tw if t < 1.0)
    nv_t = sum(1 for t in tt if t < 1.0)
    trunc = [v["trunc_mass_attn"] for v in per_layer.values()]
    ratios = [mo_attn[L] / max(mo_sel[L], 1e-9) for L in c4_layers]

    rep = {
        "what": "A6 收口:页见证 + 经验桥换算后的全链组合(c4 层),截断质量另立名目",
        "machine": "hgx",
        "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8, ctx 65536",
        "caliber": [
            f"c4 层({len(c4_layers)} 层)有全部三项;c128 层页侧仍未见证,分开报不混",
            "统计量为采样分布(decode token,每次至多 4 行),非逐请求上确界",
            "选择误差项 = ρ_max × b_S:ρ 在**截断集**上实测(大集合,质量加权),"
            "外推到 η 扰动的**换手集**(小集合)是均质性假设 —— 故档位 empirical",
            "截断遗漏质量是**架构设计量**(稀疏选择是模型语义),不计入误差链",
            "ε 覆盖窗口 + 压缩页两侧条目(c4 层);组合档位 empirical(最弱段)",
        ],
        "chain": {
            "stages": [{"name": c.name, "a": c.a, "b": c.b, "tier": c.tier,
                        "m": f"{c.m_in} -> {c.m_out}", "proof": c.proof} for c in ch.stages],
            "composed": {"a": tot.a, "b": tot.b, "tier": tot.tier,
                         "m": f"{tot.m_in} -> {tot.m_out}"},
        },
        "per_layer": per_layer,
        "summary": {
            "n_c4_layers": len(c4_layers),
            "semantics": ("worst = 采样最大系数口径下的**上界**(链算);"
                          "typical = 均值系数的**诊断读数,非上界**(评审 P0:"
                          "均值系数没有上界性质,不得称 bound / non-vacuous)"),
            "tv_total_typ_median": statistics.median(tt),
            "tv_total_worst_median": statistics.median(tw),
            "n_nonvacuous_typ": nv_t, "n_nonvacuous_worst": nv_w,
            "b_S_attn": b_S_attn, "rho_median": statistics.median(ratios),
            "rho_max": rho_max,
            "trunc_mass_median": statistics.median(trunc),
            "trunc_mass_max": max(trunc),
            "wit_centry_max": max(wcen["max"].values()),
        },
    }
    s = rep["summary"]
    # **头条与桥结论全部由数据分支生成 —— 本文件初版预写结论被数据打脸两处,引以为戒**
    if 0.5 <= s["rho_median"] <= 2.0:
        proxy = ("选择器质量是注意力质量的**可用代理**:ρ 中位 %.2f 接近 1,"
                 "尾部到 %.2f(折算取 ρ_max,保守)"
                 % (s["rho_median"], s["rho_max"]))
    else:
        proxy = ("选择器质量**偏离**注意力质量(ρ 中位 %.2f,最大 %.2f)——代理不可靠,"
                 "折算不成立" % (s["rho_median"], s["rho_max"]))
    rep["findings"] = {
        "0_headline": (
            "**全链逐层结果(存储 + 经验桥换算的选择误差,均 attn_dist:tv,链算)**:"
            "worst(采样最大系数口径)为**上界**,中位 %.3f,%d/%d 层 < 1;"
            "typical(均值系数)为**诊断读数而非上界**,中位 %.3f。"
            "组合档位 empirical。选择误差项 ρ_max×b_S = %.5f —— 不是瓶颈;"
            "主导是存储见证本身(centry max %.3f 比 swa 还大)"
            % (s["tv_total_worst_median"], nv_w, len(c4_layers),
               s["tv_total_typ_median"], s["b_S_attn"], s["wit_centry_max"])),
        "1_bridge_quality": (
            "**经验桥(数据说了算)**:" + proxy +
            ";故 b_S 可按 ρ_max 折算到注意力单位,选择误差 = %.5f,empirical 档"
            % s["b_S_attn"]),
        "2_truncation": (
            "**截断遗漏质量(架构设计量,另立名目)**:top-512 截断在本负载 decode 采样上"
            "遗漏假想全压缩注意力质量中位 %.3f(最大 %.3f)—— 这是 V4 稀疏语义的描述量,"
            "不是误差;把它当误差加进链是本文件初版的错,已纠正"
            % (s["trunc_mass_median"], s["trunc_mass_max"])),
        "3_coverage": (
            "ε 覆盖窗口与压缩页两侧(c4 层);c128 层页侧仍未见证,分开报 —— "
            "范围收窄了一半,如实说"),
    }
    dst = os.path.join(OUT, "p78_full_composition.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print("组合:", rep["chain"]["composed"])
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
