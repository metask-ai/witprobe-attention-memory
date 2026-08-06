# -*- coding: utf-8 -*-
"""A7 收口(p86):页侧见证补齐 c128 后,存储项组合覆盖**全部 43 层**。

结构差异如实分开(43 = 2 dense + 21 c4 + 20 c128,layer_mapping 实证):
  · dense 层(2,L0/L1):ratio=0,**无压缩页**(compress_kv_pool=None)——
    存储项只有窗口侧,ε = qn·W_swa;
  · c4 层(21):存储项(窗口+页,ε = qn·max(W_swa, W_centry_w))+ 经验桥换算的
    选择误差项(ρ_max×b_S,empirical)—— 与 p78 同构;
  · c128 层(20):**无索引器,注意力稠密读全部压缩条目** —— 组合只有存储项。

口径:
  · W_centry_w 为**写后读回**的二次量化见证(采样条目,分布统计非逐请求上确界);
  · worst = 全 rank 逐层最大系数口径的**上界**;typical = 均值系数**诊断读数,非上界**;
  · 逐层 TV 是该层注意力分布上的界;跨层不组合(无深度桥)。

python3 experiments/p86_full43_composition.py
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
    qn = {}
    w = {"swa": {"max": {}, "mean": {}}, "cw": {"max": {}, "mean": {}}}
    for f in ranks:
        d = json.load(open(f))
        for k, v in d["slots"].items():
            L = int(k.split("|")[1][1:])
            if k.endswith("|qn"):
                qn[L] = max(qn.get(L, 0.0), v["q_scaled_norm"]["max"])
            elif "swa_norm_rope" in k and "wit_int8" in v:
                w["swa"]["max"][L] = max(w["swa"]["max"].get(L, 0.0), v["wit_int8"]["max"])
                w["swa"]["mean"].setdefault(L, []).append(v["wit_int8"]["mean"])
            elif k.endswith("|centry_w") and "wit_int8" in v:
                w["cw"]["max"][L] = max(w["cw"]["max"].get(L, 0.0), v["wit_int8"]["max"])
                w["cw"]["mean"].setdefault(L, []).append(v["wit_int8"]["mean"])
    for kk in w.values():
        kk["mean"] = {L: statistics.mean(v) for L, v in kk["mean"].items()}
    return qn, w


def main():
    ranks = sorted(glob.glob(os.path.join(OUT, "wc_c128.json.rank*")))
    assert len(ranks) == 8, f"预期 8 个 rank,实得 {len(ranks)}"
    qn, w = collect(ranks)
    att = json.load(open(os.path.join(OUT, "p55_v4flash_attribution.json")))
    b_S = att["select_path"]["b_S"]["0.001"]
    cert = att["select_path"]["cert_rate_set"]
    # c4 层集合以 p77/p78 的 selbridge 覆盖为准(21 层);其余为 c128
    p78 = json.load(open(os.path.join(OUT, "p78_full_composition.json")))
    c4_layers = {int(k) for k in p78["per_layer"]}
    rho_max = p78["summary"]["rho_max"]
    paged = sorted(set(qn) & set(w["cw"]["max"]))
    assert len(paged) == 41, f"页侧见证未覆盖 41 个有页层,实得 {len(paged)}"
    dense = sorted(set(qn) - set(w["cw"]["max"]))
    assert dense == [0, 1], f"稠密层应为 L0/L1,实得 {dense}"
    layers = sorted(set(qn))

    bridge = C.empirical_selector_to_attn_bridge(
        rho_max, len(c4_layers), measured_on="c4 截断集(decode 采样)",
        applied_to="η=0.001 扰动换手集")
    sel_attn = C.Chain().then(C.selection_contract(b_S, cert)).then(bridge).compose()

    per_layer = {}
    for L in layers:
        def wmax(cal):
            return max(w["swa"][cal].get(L, 0.0), w["cw"][cal].get(L, 0.0))
        fam = "dense" if L in dense else ("c4" if L in c4_layers else "c128")
        ch_w = C.v4_bridged_chain({L: qn[L]}, {L: wmax("max")})
        if fam == "c4":
            ch_w.also(sel_attn)
        v_w = ch_w.evaluate(e_in=wmax("max"))["e_out"]
        e_t = qn[L] * wmax("mean")
        v_t = 0.5 * (math.exp(2 * e_t) - 1) + (sel_attn.b if fam == "c4" else 0.0)
        per_layer[L] = {"family": fam,
                        "eps_worst": qn[L] * wmax("max"), "eps_typ": e_t,
                        "tv_worst": v_w, "tv_typ": v_t}
    def med(fam, key):
        return statistics.median([v[key] for v in per_layer.values() if v["family"] == fam])
    tvw = [v["tv_worst"] for v in per_layer.values()]
    nv_w = sum(1 for t in tvw if t < 1.0)

    rep = {
        "what": "A7 收口:页侧见证补齐 c128,存储项组合覆盖全 43 层(结构差异分开报)",
        "machine": "hgx",
        "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8, ctx 65536",
        "caliber": [
            "dense(L0/L1)ratio=0 无压缩页;c128 无索引器稠密读 —— 两者均无选择项(架构事实)",
            "W_centry_w 为写后读回的二次量化见证;采样分布统计,非逐请求上确界",
            "worst=采样最大系数口径上界;typical=均值系数诊断读数非上界;跨层不组合",
        ],
        "n_layers": len(layers),
        "families": {"dense": 2, "c4": len(c4_layers), "c128": 43 - 2 - len(c4_layers)},
        "per_layer": per_layer,
        "summary": {
            "tv_worst_median_c4": med("c4", "tv_worst"),
            "tv_worst_median_c128": med("c128", "tv_worst"),
            "tv_typ_median_c4": med("c4", "tv_typ"),
            "tv_typ_median_c128": med("c128", "tv_typ"),
            "n_nonvacuous_worst_43": nv_w,
            "w_cw_max_c128": max(w["cw"]["max"][L] for L in paged if L not in c4_layers),
            "w_cw_max_c4": max(w["cw"]["max"][L] for L in paged if L in c4_layers),
        },
    }
    s = rep["summary"]
    rep["findings"] = {
        "0_headline": (
            "**页侧空白清零:41 个有页层全部见证,另 2 层(L0/L1)为 ratio=0 稠密层"
            "(无页侧,架构事实)** —— 43 层的存储项结构全部落账。逐层 TV(worst 口径上界)"
            "中位:c4 %.3f / c128 %.3f;typical 诊断读数中位:c4 %.3f / c128 %.3f;"
            "43 层中 %d 层 worst 界 < 1。c128 页见证 max %.3f(c4 侧 %.3f)"
            % (s["tv_worst_median_c4"], s["tv_worst_median_c128"],
               s["tv_typ_median_c4"], s["tv_typ_median_c128"],
               s["n_nonvacuous_worst_43"], s["w_cw_max_c128"], s["w_cw_max_c4"])),
        "1_structure": (
            "结构差异照实分开:c4 层 = 存储 + 经验桥选择项(empirical);"
            "c128 层 = 仅存储项 —— 无索引器是架构事实(p85 探明),不是测量省略"),
    }
    dst = os.path.join(OUT, "p86_full43_composition.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
