# -*- coding: utf-8 -*-
"""请求级风险账本(p80):第一档保证的**可执行版本**与覆盖率/回退率曲线。

**保证的定义**(第一档,与 GPT 讨论一致):
    P(请求内任一入账事件越界) ≤ Σ_i δ_i ≤ δ_req
账本三铁律:empirical 绝不入 certified 账本;无法认证 → 局部回退(精确读取);
未知长度用望远镜权重 δ_i = w_i·δ_req(Σw_i ≤ 1,Lean: Ledger.telescope_sum)。
**它保证"局部操作不越界",不保证最终输出相同** —— 后者需层间传播或重置点(下一阶段)。

**本文件做的事**:用 p79 的**条目级**见证生存计数,离线重放账本 ——
  事件 = (c4 层, 采样 decode 步, 条目) 的存储侧读取;
  证书 = 确定性判定 W_entry ≤ W_thr(逐条目;运行时可算);
  过 → certified,该条目走压缩;不过 → **局部回退**(仅该条目精确读取)。
  于是整步的 TV 界**构造性成立**:被读条目全部 ≤ W_thr ⟹
  TV_step ≤ ½(e^{2·qn·W_thr}−1)(回退条目精确,不贡献误差)。
  证书粒度是本文件第一版的教训:按步取条目最大值判定,一个大见证条目就把整步
  打回退,覆盖率 0% —— **局部回退的"局部"必须落在条目上**。
  选择侧事件当前一律 fallback(索引条目见证未建,A8)—— **empirical 不入账**。

统计口径(三审收紧):
  · 条目级计数是**同请求内相关条目 + 确定性等距抽样**,不满足独立性 ——
    Hoeffding/rule-of-three 在条目粒度**不构成严格置信上界**,一律标 diagnostic。
  · 可辩护的置信在**请求粒度**:24 条请求视为可交换单元,若全部请求的账本重放
    无超阈事件,rule-of-three 给 P(新请求出现超阈) 的单侧 95% 上界 3/24 = 12.5%
    (Lean: coverage_confidence;可交换性假设显式写出)。
  · **"0 次违约"必须写成区间,不写 0%。**

口径:
  · ε_step = scale·max_head‖q‖ × max_采样条目 W —— 条目是等距采样 cap 个,
    **采样最大值口径**,非该步全体条目的真最大;生产化需逐读取检查(gating 补丁的职责)。
  · 负载 24 请求 × 3 档长度 × 96 token decode;分层(8k/30k/60k)未拆 —— 单一混合口径。
  · c128 层不在账本(页侧未见证);账本覆盖 c4 21 层。

python3 experiments/p80_request_ledger.py
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

GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50)


def collect(ranks):
    """每层:条目数与各 W 阈值下的超越计数,另收每步 ε 计数(诊断用)。
    全 rank 合并(rank 是同一步的不同分片,计数直接相加)。"""
    wsurv, wn, esurv, en = {}, {}, {}, {}
    for f in ranks:
        d = json.load(open(f))
        for k, v in d["slots"].items():
            if not k.endswith("|selbridge"):
                continue
            L = int(k.split("|")[1][1:])
            if "wit_n" in v:
                wn[L] = wn.get(L, 0) + int(v["wit_n"])
                for t, c in (v.get("wit_surv") or {}).items():
                    wsurv.setdefault(L, {})[t] = wsurv.get(L, {}).get(t, 0) + int(c)
            if "eps_n" in v:
                en[L] = en.get(L, 0) + int(v["eps_n"])
                for t, c in (v.get("eps_surv") or {}).items():
                    esurv.setdefault(L, {})[t] = esurv.get(L, {}).get(t, 0) + int(c)
    return wsurv, wn, esurv, en


def p_upper(k, n, delta=0.05):
    """单事件超阈率的单侧 1−δ 上界。k=0 用 (1−p)^n ≤ δ ⟹ p ≤ ln(1/δ)/n
    (Lean: coverage_confidence);k>0 用 Hoeffding 单侧(保守)。"""
    if n <= 0:
        return 1.0
    if k == 0:
        return min(1.0, math.log(1 / delta) / n)
    return min(1.0, k / n + math.sqrt(math.log(1 / delta) / (2 * n)))


def main():
    ranks = sorted(glob.glob(os.path.join(OUT, "wc_ledger.json.rank*")))
    assert len(ranks) == 8, f"预期 8 个 rank,实得 {len(ranks)}"
    wsurv, wn, esurv, en = collect(ranks)
    layers = sorted(wn)
    assert layers, "没有条目级见证计数 —— p79 没跑到账本路径"
    # qn:桥系数(逐层最大),从同一批产物读
    qn = {}
    for f in ranks:
        d = json.load(open(f))
        for k, v in d["slots"].items():
            if k.endswith("|qn"):
                L = int(k.split("|")[1][1:])
                qn[L] = max(qn.get(L, 0.0), v["q_scaled_norm"]["max"])

    curve = []
    for w_thr in GRID:
        led = C.RequestLedger(0.01)
        per_layer = {}
        for L in layers:
            n = wn[L]
            k = wsurv.get(L, {}).get(f"{w_thr:g}", 0)
            led.n_certified += n - k
            led.n_fallback += k
            per_layer[L] = {"n": n, "k": k,
                            "p_diag": p_upper(k, n),   # **诊断值**:条目相关+等距抽样,非严格置信
                            "tv_bound": 0.5 * (math.exp(2 * qn.get(L, 1.0) * w_thr) - 1)}
        rep_l = led.report()
        curve.append({
            "W_thr": w_thr,
            "tv_bound_per_step_max": max(v["tv_bound"] for v in per_layer.values()),
            "entry_retention": rep_l["certified_coverage"],   # 命名:压缩条目保留率
            "fallback_rate": rep_l["fallback_rate"],
            "p_diag_worst_layer": max(v["p_diag"] for v in per_layer.values()),
            "n_entries": rep_l["n_events"],
        })

    n_total = sum(wn.values())
    # **双工作点,数据说了算**:覆盖优先点(≥80% 中界最紧)与界优先点(界<1 中覆盖最高)。
    # 若同一点两者兼得,曲线自会显示;不兼得就如实分开报,不挑好看的那个。
    ok_pts = [c for c in curve if c["entry_retention"] >= 0.80]
    main_pt = min(ok_pts, key=lambda c: c["tv_bound_per_step_max"]) if ok_pts else curve[-1]
    nv_pts = [c for c in curve if c["tv_bound_per_step_max"] < 1.0]
    bound_pt = max(nv_pts, key=lambda c: c["entry_retention"]) if nv_pts else None
    rep = {
        "what": ("请求级风险账本**原型 + 离线可行性重放**(三审定位):"
                 "条目级确定性证书 + 假想局部回退;保留率/TV 界曲线(c4 层)。"
                 "**不是在线逐请求闭环** —— 无请求边界/逐读取检查/真实回退读路径,"
                 "那些是被本重放定价的实现工作"),
        "machine": "hgx",
        "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8, ctx 65536",
        "guarantee": ("P(请求内任一入账事件越界) ≤ Σδ_i ≤ δ_req;本轮证书全部为确定性"
                      "逐条目判定(δ=0),Σδ=0。构造性保证:被读条目全 ≤ W_thr ⟹ "
                      "TV_step ≤ ½(e^{2·qn·W_thr}−1)。"
                      "**保证局部操作不越界,不保证最终输出相同**"),
        "lean": ["Ledger.ledger_sound", "Ledger.telescope_sum", "Ledger.coverage_confidence"],
        "caliber": [
            "证书粒度=条目;回退粒度=条目(局部回退),覆盖率=压缩收益保留率",
            "探针为采样审计口径;生产化=逐读取检查,由 gating 补丁执行(C2 边界)",
            "选择侧事件一律 fallback(索引条目见证未建,A8);empirical 不入账",
            "24 请求 × 3 档长度混合口径;c128 层不在账本;窗口条目见证由 swa 路覆盖,"
            "其分布未入本账本(W_swa ≤ W_centry,实测)",
        ],
        "n_layers": len(layers), "n_entries_total": n_total,
        "coverage_curve": curve,
        "working_point_coverage": main_pt,
        "working_point_bound": bound_pt,
        "findings": {},
    }
    _dual = ("保留优先点 W=%.2f:压缩条目保留率 %.1f%%,但每步 TV 构造界 %.2f(空洞);"
             % (main_pt["W_thr"], 100 * main_pt["entry_retention"],
                main_pt["tv_bound_per_step_max"]))
    if bound_pt is not None:
        _dual += ("界优先点 W=%.2f:TV ≤ %.2f(非空洞),保留率仅 %.1f%%"
                  % (bound_pt["W_thr"], bound_pt["tv_bound_per_step_max"],
                     100 * bound_pt["entry_retention"]))
    else:
        _dual += "无任何档位给出非空洞界"
    rep["findings"]["0_headline"] = (
        "**账本原型 + 离线可行性重放(非在线闭环),数字如实**:" + _dual +
        " —— 在 int8 二次量化代理的见证分布上(集中于 0.5–0.7,无重尾),"
        "**非空洞每步界与 ≥80%% 覆盖不能同时成立**。杠杆与 p78 一致:见证 W 本身;"
        "更细的带宽/贴合 fp8 的见证是收紧方向。Σδ=0(全确定性证书),δ_req 预算未动")
    rep["findings"]["1_curve"] = "; ".join(
        "W=%.2f: 保留 %.1f%%/TV≤%.2f" % (c["W_thr"], 100 * c["entry_retention"],
                                          c["tv_bound_per_step_max"]) for c in curve)
    rep["findings"]["2_granularity"] = (
        "**证书粒度是成败线**(本文件第一版教训):按步取条目最大值判定 → 覆盖 0%%;"
        "改条目级局部回退后,一个大见证条目只损失它自己。n=%d 条目,层中位 %d"
        % (n_total, statistics.median(list(wn.values()))))
    rep["findings"]["3_semantics"] = (
        "第一档保证:请求内所有走压缩路径的读取都有确定性证书,不过的**逐条目**精确读取。"
        "不保证最终输出(层间传播/重置点是下一阶段);empirical 对象不入账,只回退")
    dst = os.path.join(OUT, "p80_request_ledger.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst, f"(层 {len(layers)},条目 {n_total})")
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
