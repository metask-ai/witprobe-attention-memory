# -*- coding: utf-8 -*-
"""R8-E4.1:把 p56 的逐 rank 原始观测汇总成 p56_kda_contraction.json。

输入 experiments/out/wc_kda_probe.json.rank*,输出 experiments/out/p56_kda_contraction.json。

**这份产物的用途是给论文3 定形态**(E4.2 的 go/no-go),故两个口径必须同时出现:
  · 均值口径 ā 与半衰期 —— 决定"平均意义下界收不收敛";
  · 近 1 尾部 P(a_t≥0.999) 与 max —— 决定"最坏口径下界空不空洞"。
只报前者会把一个条件成立的结论说成无条件成立(1289× 空洞化教训的状态版)。

口径限制(引用数字必须一并给出):a_t 由逐层标量 A_log/dt_bias 均值广播近似,
非逐头精确;模型为 Kimi-Linear-48B-A3B 代理,非 Kimi-K3 93L。

python experiments/p56_kda_report.py
"""
import glob
import json
import math
import os
import statistics as S
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "wc_kda_probe.json.rank*")),
                key=lambda s: int(s.split("rank")[-1]))
    if not fs:
        raise SystemExit("找不到 wc_kda_probe.json.rank*")
    ranks = []
    for f in fs:
        d = json.load(open(f))
        rows = []
        for li in sorted(d["layers"], key=int):
            L = d["layers"][li]
            a = L["a_t"]
            rows.append({"layer": int(li), "a_mean": a["mean"], "a_max": a["max"],
                         "a_min": a["min"], "p_ge_999": (L.get("a_hist") or [0])[-1],
                         "A": L.get("A"), "dt_bias": L.get("dt_bias"),
                         "beta_mean": L.get("beta", {}).get("mean"), "n": a["n"]})
        ranks.append({"rank": int(f.split("rank")[-1]), "n_calls": d["n_calls"],
                      "n_layers": len(rows), "per_layer": rows})

    allrows = [r for rk in ranks for r in rk["per_layer"]]
    am = [r["a_mean"] for r in allrows]
    pg = [r["p_ge_999"] for r in allrows]
    hl = [math.log(2) / (-math.log(v)) for v in am if 0 < v < 1]
    rep = {
        "model": "Kimi-Linear-48B-A3B-Instruct(Kimi-K3 同架构代理,KimiLinearForCausalLM)",
        "stack": "sglang 0.5.13.post1, tp2, ctx 32768, disable-cuda-graph + disable-piecewise-cuda-graph",
        "protocol": ("探针注入 KimiDeltaAttention 的两条 qkvbfg 路径;"
                     "a_t = exp(−softplus(forget_gate + dt_bias)·exp(A_log));探针流量 2k/8k/20k"),
        "caliber_limits": ["a_t 用逐层标量 A_log/dt_bias 均值广播近似,非逐头精确",
                           "48B 代理,非 Kimi-K3 93L"],
        "ranks": len(ranks),
        "n_kda_layers": ranks[0]["n_layers"],
        "layers": [r["layer"] for r in ranks[0]["per_layer"]],
        "coverage_note": ("捕获层号 = config.linear_attn_config.kda_layers 的 0-indexed 全集;"
                          "全模型 27 层中另 7 层为 full_attn"),
        "a_mean": {"median": S.median(am), "min": min(am), "max": max(am)},
        "p_ge_0999": {"median": S.median(pg), "min": min(pg), "max": max(pg)},
        "half_life_steps": {"median": S.median(hl), "min": min(hl), "max": max(hl)},
        "a_max_overall": max(r["a_max"] for r in allrows),
        "beta_mean": S.median([r["beta_mean"] for r in allrows if r["beta_mean"]]),
        "per_rank": ranks,
    }
    rep["findings"] = {
        "1_mean_contraction_real":
            "均值口径收缩成立:ā 中位 %.4f(区间 %.4f-%.4f),误差半衰期中位 %.1f 步 —— "
            "递归界的累积项收敛到约 b/(1−ā)≈%.0f·b,**非空洞**"
            % (rep["a_mean"]["median"], rep["a_mean"]["min"], rep["a_mean"]["max"],
               rep["half_life_steps"]["median"], 1 / (1 - rep["a_mean"]["median"])),
        "2_worst_case_vacuous":
            "最坏口径空洞:每层 a_t max 达 %.3f,P(a_t≥0.999) 中位 %.3f 最大 %.3f 且随深度增 —— "
            "逐通道取 max 的 uniform 界无收缩"
            % (rep["a_max_overall"], rep["p_ge_0999"]["median"], rep["p_ge_0999"]["max"]),
        "3_verdict":
            "论文3 形态 = **条件 go**:界须按通道/统计口径,或对近 1 通道单列处理;"
            "纯 worst-case 叙事放弃",
        "4_depth_trend":
            "ā 随深度单调上升(浅层遗忘强、深层近乎保持),这是逐层口径才能看到的结构",
    }
    dst = os.path.join(OUT_DIR, "p56_kda_contraction.json")
    # 防覆盖:本仓库有两条并行工作线都产出 p56,schema 不同。不是本脚本生成的产物
    # 一律不动 —— 静默覆盖别人的实验结果比不生成危险得多。
    if os.path.exists(dst) and "--force" not in sys.argv:
        old = json.load(open(dst))
        if old.get("generated_by") != "experiments/p56_kda_report.py":
            raise SystemExit(
                "拒绝覆盖 %s:它不是本脚本生成的(generated_by=%r)。\n"
                "如确要替换,加 --force;否则请直接引用现有产物。"
                % (dst, old.get("generated_by")))
    rep["generated_by"] = "experiments/p56_kda_report.py"
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
