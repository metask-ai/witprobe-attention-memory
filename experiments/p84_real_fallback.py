# -*- coding: utf-8 -*-
"""A9 终局(p84):真实回退四臂对照的判读。

**这是第一次策略在数值路径上真实生效**:
  off    基线;all 全掩码无回退(退化对照);policy 真实逐条目回退;dither 概率预算真实入账。

判读标准(数据说了算,先写判据再看数):
  1. all 臂输出应偏离基线(掩码真的在影响模型 —— 若不偏离,说明压缩太轻,结论无信息);
  2. policy 臂偏离度应 ≤ all 臂(策略在保护输出;严格更小才算正结果);
  3. dither 臂 0 < Σδ ≤ δ_req,assumption 留痕;
  4. retention 落在有效区(0.2–0.9):太高说明阈值形同虚设,太低说明全在回退。

偏离度口径:同一 prompt 贪心解码,与基线的**首异位置**(prefix match length)与
完全一致率;逐 prompt 配对。8 条 prompt 为小样本 —— 方向性结论,不做显著性声称。

python3 experiments/p84_real_fallback.py
"""
import glob
import json
import os
import statistics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def load_arm(arm):
    out = json.load(open(os.path.join(OUT, f"p83_out_{arm}.json")))["outputs"]
    cw = None
    fs = sorted(glob.glob(os.path.join(OUT, f"wc_cw_{arm}.json.rank0")))
    if fs:
        cw = (json.load(open(fs[0])) or {}).get("certified_write")
    return out, cw


def main():
    base, _ = load_arm("off")
    rep = {"what": "A9 终局:真实回退四臂对照(策略首次在数值路径生效)",
           "machine": "hgx",
           "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8",
           "caliber": [
               "压缩 = FP8 nope 尾数掩码 2 位(值域真实变粗);scale/rope 不动",
               "回退 = 超阈条目保持原字节 —— 数值路径真实生效,非影子",
               "偏离度 = 与基线贪心输出的首异位置;8 prompt 小样本,方向性结论",
               "写侧无请求 id,账目为全局口径(逐请求账本见 p82 影子层)",
           ],
           "arms": {}}
    for arm in ("all", "policy", "dither"):
        outs, cw = load_arm(arm)
        pls = [prefix_len(a, b) for a, b in zip(base, outs)]
        exact = sum(1 for a, b in zip(base, outs) if a == b)
        rep["arms"][arm] = {
            "prefix_match_median": statistics.median(pls),
            "prefix_match_min": min(pls),
            "exact_match": f"{exact}/{len(base)}",
            "certified_write": cw,
        }
    a_all = rep["arms"]["all"]; a_pol = rep["arms"]["policy"]; a_dit = rep["arms"]["dither"]
    cw_pol = a_pol["certified_write"] or {}
    cw_dit = a_dit["certified_write"] or {}

    checks = {
        "1_all_diverges": a_all["exact_match"] != f"{len(base)}/{len(base)}",
        "2_policy_protects": (a_pol["prefix_match_median"] >= a_all["prefix_match_median"]),
        "3_delta_spent": 0.0 < (cw_dit.get("delta_spent") or 0.0) <= 0.01,
        "4_retention_informative": 0.2 <= (cw_pol.get("retention") or 0.0) <= 0.9,
    }
    rep["checks"] = checks
    rep["findings"] = {
        "0_headline": (
            "**真实回退首次生效**:policy 臂 retention %.3f(真实压缩 %s 条 / 真实回退 %s 条),"
            "首异位置中位 all=%s / policy=%s / dither=%s(基线=完整长度);"
            "dither 臂 Σδ=%.6f(≤ δ_req=0.01),assumption:%s。四判据:%s"
            % (cw_pol.get("retention") or -1,
               f"{cw_pol.get('n_compressed', 0):,}", f"{cw_pol.get('n_fallback_exact', 0):,}",
               a_all["prefix_match_median"], a_pol["prefix_match_median"],
               a_dit["prefix_match_median"],
               cw_dit.get("delta_spent") or 0.0,
               "; ".join(cw_dit.get("assumptions") or []),
               {k: bool(v) for k, v in checks.items()})),
    }
    dst = os.path.join(OUT, "p84_real_fallback.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
