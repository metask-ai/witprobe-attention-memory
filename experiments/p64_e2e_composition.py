# -*- coding: utf-8 -*-
"""R9-B5:V4-Flash 的**请求级**端到端组合与非空洞性论证。

p55 给的是逐层中位数;但一条请求要**穿过全部 43 层**,所以"请求级"必须把深度也组合
进来。本文件回答三个问题:

  1. 单层的界是多少?非空洞吗?
  2. 按 (C1) 沿深度组合后,请求级的界是多少?还非空洞吗?
  3. 和"平凡契约"比,我们的契约紧了多少?——不给这个对比,"非空洞"就只是个形容词。

**平凡契约**指不用任何测量能写出的界:选择段最坏情况整个 top-k 集合被换掉(b_S=1),
存储段最坏情况条目被完全破坏(b=1)。它们 sound 但无信息(TV≤1 恒成立)。

python experiments/p64_e2e_composition.py
"""
import json
import os
import statistics as S
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "experiments", "out")
sys.path.insert(0, os.path.join(ROOT, "src"))
from witcert.probe import contracts as C  # noqa: E402


def main():
    att = json.load(open(os.path.join(OUT_DIR, "p55_v4flash_attribution.json")))
    snap_path = os.path.join(OUT_DIR, "wc_attr.json.rank0")
    snap = json.load(open(snap_path))

    st_med = att["store_path"]["rel_int8_median"] * att["store_path"]["tight_int8_median"]
    n_store = att["store_path"]["n_layers"]
    n_sel = att["select_path"]["n_layers"]

    # --- 1. 逐层:有选择器的层 = 存储+选择;其余层只有存储 ---
    per_layer = []
    for k, s in snap["slots"].items():
        if not k.endswith("|swa_norm_rope") or "rel_int8" not in s:
            continue
        li = int(k.split("|L")[1].split("|")[0])
        b_store = s["rel_int8"]["mean"] * s["tight_int8"]["mean"]
        sel = next((v for kk, v in snap["slots"].items()
                    if kk.endswith("|c4") and f"|L{li}|" in kk and "sel_bS" in v), None)
        b_sel = sel["sel_bS"]["0.001"]["mean"] if sel else 0.0
        ch = C.Chain().then(C.Contract(f"store@L{li}", 1.0, b_store, tier=C.CERTIFIED))
        if sel:
            ch.then(C.Contract(f"select@L{li}", 1.0, b_sel, tier=C.PARTIAL))
        per_layer.append({"layer": li, "b_store": b_store, "b_select": b_sel,
                          "bound": ch.compose().b, "tier": ch.compose().tier,
                          "has_selector": sel is not None})
    per_layer.sort(key=lambda r: r["layer"])
    bounds = [r["bound"] for r in per_layer]

    # --- 2. 请求级:按 (C1) 沿深度组合。各层 a=1,故请求级界 = 各层 b 之和 ---
    req = C.Chain()
    for r in per_layer:
        req.then(C.Contract(f"L{r['layer']}", 1.0, r["bound"],
                            tier=C.PARTIAL if r["has_selector"] else C.CERTIFIED))
    req_c = req.compose()

    # --- 3. 与平凡契约对比 ---
    triv = C.Chain()
    for r in per_layer:
        triv.then(C.Contract(f"L{r['layer']}(trivial)", 1.0, 1.0, tier=C.CERTIFIED))
    triv_b = triv.compose().b

    rep = {
        "what": "V4-Flash 请求级端到端组合与非空洞性论证",
        "model": att["model"], "stack": att["stack"],
        "caliber": [
            "单位是**相对见证**(运行时可算),不是 TV —— 折 TV 需逐请求 q 范数",
            "深度组合按 (C1) 且各段 a=1,故请求级界 = 各层界之和;"
            "若某段 a<1(如误差在层间被平均)则该式偏保守",
            f"存储覆盖 {n_store} 层,选择器只在 {n_sel} 层;无选择器的层只计存储",
        ],
        "per_layer": {
            "n": len(per_layer),
            "bound_median": S.median(bounds), "bound_min": min(bounds), "bound_max": max(bounds),
            "worst_layer": max(per_layer, key=lambda r: r["bound"])["layer"],
            "rows": per_layer,
        },
        "request_level": {
            "bound": req_c.b, "tier": req_c.tier, "n_layers": len(per_layer),
            "trivial_bound": triv_b,
            "tightness_vs_trivial": triv_b / max(1e-12, req_c.b),
        },
    }
    pl, rq = rep["per_layer"], rep["request_level"]
    # 非空洞的判据要**留余量**:界只要逼近平凡上界(误差与信号同量级),实用上就已经没有信息。
    # 0.982 < 1.0 说"非空洞"是技术上正确、实质上误导 —— 本项目反复抓这类过度声称,
    # 自己更不能犯。故:<0.5 才算非空洞,[0.5,1) 算**边缘**,>=1 算饱和。
    VACUOUS, BORDER = 1.0, 0.5
    verdict = ("非空洞" if rq["bound"] < BORDER
               else "**边缘(实质已接近平凡)**" if rq["bound"] < VACUOUS else "**饱和**")
    rq["verdict"] = verdict
    rq["headroom_to_trivial"] = 1.0 - rq["bound"]
    rep["findings"] = {
        "1_per_layer_nonvacuous":
            "**单层非空洞**:逐层界中位 %.4f、最大 %.4f(第 %d 层),远小于平凡界 1.0"
            % (pl["bound_median"], pl["bound_max"], pl["worst_layer"]),
        "2_request_level":
            ("请求级判定 = %s:沿 %d 层按 (C1) 组合得 **%.3f**(相对见证单位),"
             "距平凡上界 1.0 只剩 %.3f 的余量。契约比平凡契约紧 %.0f×,"
             "但**朴素的深度组合在请求级实质上已经没有信息** —— "
             "单层结论可用,请求级结论不可用"
             % (verdict, rq["n_layers"], rq["bound"], rq["headroom_to_trivial"],
                rq["tightness_vs_trivial"])),
        "3_where_the_difficulty_is":
            ("难点不在任何单段,而在**深度**:单段贡献 %.4f 量级,乘以 %d 层就顶到 %.2f。"
             "这与论文1 的'跨层累积才是失效机制'是同一件事,现在用契约语言量化了。"
             % (pl["bound_median"], rq["n_layers"], rq["bound"])),
        "4_what_would_fix_it":
            "要让请求级非空洞,需要某段 a<1(层间误差被平均/衰减)。池化的 a_pool≈0.55 正是"
            "这类系数,但它作用在池化**之前**的误差上,不在本链条里 —— 层间是否也有这样的"
            "衰减,是下一个要测的东西(**推测,未验证**)",
        "5_honest_reading":
            "所以本节的正确读法是:**契约演算成功地把难点定位到了深度上**,而不是"
            "'我们给出了请求级证书'。前者是真结论且有用(它指出下一步该测层间衰减),"
            "后者会是过度声称。",
    }
    dst = os.path.join(OUT_DIR, "p64_e2e_composition.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
