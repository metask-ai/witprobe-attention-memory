# -*- coding: utf-8 -*-
"""F4(p95):同对象 e-process 判读。

六审 P0-3 的收口:此前 418×/390× 对比的两个对象是**不同失败事件**(已在 p92
如实声明)。本实验把 e-process **接到逐条目授权事件本身**:

  · 因子挂在每个授权条目上:g_e = exp(λ_e(W_e − m_e) − λ_e²C_e/8),
    λ_e = 4κ/√C_e,κ=0.5 预注册;W_e 为该条目的**实现见证**(写后读回),
    m_e 为其**抽签前**先验均值项(均值项+确定项)—— 同一物理条目、同一次授权;
  · E[g_e|历史] ≤ 1 由 McDiarmid MGF + witness_mean_le 保证
    (Lean: McDiarmid.eprocess_factor_mean_le_one,axioms 三条标准);
  · Ville ⟹ P(∃t: log M_t ≥ ln(1/δ_e)) ≤ δ_e = 1%,任意时刻有效;
  · 语义:**模型校验**,不是 union 预算的替代 —— 越阈 = 半径假设(R3-SR
    条件均值零/独立有界)被数据否证,是持续运行下的漂移哨兵;逐条目授权
    仍由望远镜 δ 预算承担。

python3 experiments/p95_eprocess_wired.py
"""
import glob
import json
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def seq_of(rid):
    return int(rid.split("#")[1]) if "#" in rid else -1


def main():
    fs = sorted(glob.glob(os.path.join(OUT, "p95_dither_r7.rank0")))
    if not fs:
        raise SystemExit("缺 p95_dither_r7.rank0")
    d = json.load(open(fs[0]))
    by_rid = d.get("cwrite_by_rid") or {}
    rows = []
    for rid in sorted(by_rid, key=seq_of):
        ep = by_rid[rid].get("eprocess")
        if ep is None:
            continue
        thr = math.log(1.0 / ep["delta_e"])
        rows.append({"rid": rid, "n_factors": ep["n_factors"],
                     "log_M_final": ep["log_M"], "log_M_max": ep["log_M_max"],
                     "threshold": thr, "crossed": ep["crossed"],
                     "kappa": ep["kappa"]})
    n_crossed = sum(1 for r in rows if r["crossed"])
    n_fac = sum(r["n_factors"] for r in rows)
    max_lm = max((r["log_M_max"] for r in rows), default=0.0)
    rep = {
        "what": "F4:同对象 e-process —— 接到逐条目授权事件的任意时刻模型校验",
        "machine": "hgx",
        "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8",
        "caliber": [
            "对象 = 压缩页池授权条目的实现见证 W_e 对抽签前先验均值 m_e —— 与逐条目"
            "授权**同一物理条目、同一事件时刻**(六审 P0-3 的对象错位由此关闭)",
            "语义 = 模型校验(半径假设的漂移哨兵),δ_e=1% 独立于 δ_req;"
            "**不替代**逐条目 union 预算 —— 两者互补,p92 的量级对比仍按不同对象报",
            "任意时刻性在写批粒度(批内因子一次乘入);κ=0.5 预注册,逐请求账户",
            "E[g|历史]≤1 的形式依据:Lean eprocess_factor_mean_le_one"
            "(McDiarmid MGF + witness_mean_le 组合,axioms=[propext,choice,quot])",
        ],
        "per_request": rows,
        "summary": {"n_requests": len(rows), "n_factors_total": n_fac,
                    "log_M_max_global": max_lm,
                    "threshold": math.log(100.0), "n_crossed": n_crossed},
    }
    rep["findings"] = {"0_headline": (
        "**同对象 e-process 上线**:%d 个请求账户、%s 个授权条目因子,"
        "log M 峰值 %.2f vs 阈值 ln(100)=%.2f,越阈 %d 次 —— 半径假设"
        "(SR 条件均值零/独立有界)在全部真实流量上未被否证;任意时刻有效"
        "(写批粒度),κ=0.5 预注册,δ_e=1%% 独立校验预算。事件族 = 实现见证"
        "对先验均值的累计超越 —— 与逐条目授权同对象同时刻,互补不替代"
        % (len(rows), f"{n_fac:,}", max_lm, math.log(100.0), n_crossed))}
    dst = os.path.join(OUT, "p95_eprocess_wired.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print(" ", rep["findings"]["0_headline"])


if __name__ == "__main__":
    main()
