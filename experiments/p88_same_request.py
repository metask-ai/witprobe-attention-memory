# -*- coding: utf-8 -*-
"""A9-B(p88):同请求闭环判读。

五审两 P0 的验收:
  P0-1 概率语义:dither 臂为**抽签前授权**(u_e 先验半径,授权不看实现值),
       事件 E_i = "授权条目实现 W_e > u_e",δ_i 望远镜入账;**写后审计**给出
       实测违约数 —— 条件失败率的运行时验证(期望 ≈ 0)。
  P0-2 同请求闭环:同一次运行里 WITCERT_LEDGER=1(读侧逐请求)与 CWRITE(写侧
       真实策略)同开;逐请求快照差分给出写侧逐请求账目 → 每请求一行:
       读侧检查数 / 写侧真实压缩与回退 / δ 消费 / 违约 / verdict。

统计口径修正(五审):
  · off vs off2:贪心解码跨服务重启的位级稳定性对照 —— 分歧度量的地板;
  · 首异位置**右删失**:完全一致的输出不计入首异中位;分开报
    "完全一致数" 与 "仅分歧样本的首异中位";
  · 8 条 prompt 为八种不同任务,但 n=8 仍是小样本 —— 方向性结论。

python3 experiments/p88_same_request.py
"""
import glob
import json
import os
import statistics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ARMS = ("off", "off2", "policy", "dither")


def outputs(arm):
    return json.load(open(os.path.join(OUT, f"p87_out_{arm}.json")))["outputs"]


def divergence(base, outs):
    """右删失处理:一致的不计首异;分开报。"""
    exact = [a == b for a, b in zip(base, outs)]
    firsts = []
    for a, b in zip(base, outs):
        if a == b:
            continue
        n = min(len(a), len(b))
        firsts.append(next((i for i in range(n) if a[i] != b[i]), n))
    return {"exact_match": f"{sum(exact)}/{len(base)}",
            "first_div_median_divergent_only": (statistics.median(firsts) if firsts else None),
            "n_divergent": len(firsts)}


def cwrite_of(arm, r):
    fs = sorted(glob.glob(os.path.join(OUT, f"p87_{arm}_r{r}.rank0")))
    if not fs:
        return None
    return (json.load(open(fs[0])) or {}).get("certified_write")


def ledger_read_total(arm, r):
    """读侧事件总数(rank0 口径;TP 复制下各 rank 检查同一逻辑条目,合并即重复计)。

    六审 P0-2:快照是**累计**状态 —— p87 旧数据的 rid 是 pool index(串行恒 0),
    没有逐请求原生键,该请求的读侧量只能由相邻快照**差分**得出(调用方负责)。"""
    fs = sorted(glob.glob(os.path.join(OUT, f"p87_{arm}_r{r}.rank0")))
    if not fs:
        return 0
    d = json.load(open(fs[0]))
    return sum(v["n_events"] for v in (d.get("request_ledgers") or {}).values())


def main():
    base = outputs("off")
    rep = {"what": "A9-B:同请求闭环(读侧逐请求账本 + 写侧真实策略同运行)+ 概率语义修正",
           "machine": "hgx",
           "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8",
           "caliber": [
               "dither = 抽签前授权:u_e 为先验半径(均值项+尾项),授权不看实现值;"
               "事件 E_i='授权条目 W_e > u_e',δ_i 望远镜入账;写后审计违约数",
               "掩码是**精度粗化非物理压缩**(字节数不变);接受率≠压缩收益保留率",
               "8 条不同任务 prompt,n=8 小样本,方向性结论;首异位置右删失已分开报",
               "逐请求账目 = 相邻请求快照差分(读写两侧;六审 P0-2:旧数据 rid=pool index 串行恒 0,无原生逐请求键);rank0 口径,TP 复制下各 rank 为同一逻辑条目",
           ],
           "arms": {}}
    for arm in ("off2", "policy", "dither"):
        rep["arms"][arm] = divergence(base, outputs(arm))
    # 同请求闭环表:每请求一行(policy 与 dither 臂)
    closure = {}
    for arm in ("policy", "dither"):
        rows, prev = [], None
        for r in range(8):
            cw = cwrite_of(arm, r)
            if cw is None:
                continue
            rd_tot = ledger_read_total(arm, r)
            delta = {k: cw.get(k, 0) - (prev or {}).get(k, 0)
                     for k in ("n_compressed", "n_fallback_exact", "delta_spent",
                               "n_prob_events", "n_auth_violations", "n_authorized")}
            rows.append({"req": r,
                         "write_side": delta,
                         # 六审 P0-2 修复:读侧同样差分 —— 此前直接求和 = 累计值冒充逐请求
                         "read_side_events": rd_tot - (rows[-1]["_read_cum"] if rows else 0),
                         "_read_cum": rd_tot})
            prev = cw
        for row in rows:                     # 差分中间量不落盘
            row.pop("_read_cum", None)
        closure[arm] = rows
    rep["same_request_closure"] = closure

    cw_pol = cwrite_of("policy", 7) or {}
    cw_dit = cwrite_of("dither", 7) or {}
    n_pol = cw_pol.get("n_compressed", 0) + cw_pol.get("n_fallback_exact", 0)
    n_dit = cw_dit.get("n_compressed", 0) + cw_dit.get("n_fallback_exact", 0)
    acc_pol = cw_pol.get("n_compressed", 0) / max(1, n_pol)
    acc_dit = cw_dit.get("n_compressed", 0) / max(1, n_dit)
    viol = cw_dit.get("n_auth_violations", 0)
    nauth = cw_dit.get("n_authorized", 0)

    d_off2 = rep["arms"]["off2"]; d_pol = rep["arms"]["policy"]; d_dit = rep["arms"]["dither"]
    rep["summary"] = {
        "stability_off_vs_off2": d_off2["exact_match"],
        "policy_acceptance_rate": acc_pol,
        "dither_preauth_rate": acc_dit,
        "preauth_conservatism": (acc_pol / max(1e-9, acc_dit)) if acc_dit else None,
        "delta_spent_dither": cw_dit.get("delta_spent", 0.0),
        "auth_violations_observed": f"{viol}/{nauth}",
    }
    s = rep["summary"]
    rep["findings"] = {
        "0_headline": (
            "**同请求闭环成立**:同一次运行里,每请求一行 —— 读侧检查数 + 写侧真实"
            "压缩/回退增量 + δ 消费增量 + 违约审计。稳定性地板 off vs off2 一致 %s。"
            "policy 接受率 %.3f(**精度粗化接受率,非物理压缩收益**);"
            "dither **预授权率** %.3f(保守代价 = policy/dither ≈ %.1f×,先验半径的"
            "尾项所致,如实报);预授条目实测违约 %s(事件语义的运行时验证);"
            "Σδ = %.6f 且每一份 δ_i 都绑定事件 E_i='授权条目实现 W 超预授半径'"
            % (s["stability_off_vs_off2"], s["policy_acceptance_rate"],
               s["dither_preauth_rate"],
               s["preauth_conservatism"] or float("nan"),
               s["auth_violations_observed"], s["delta_spent_dither"])),
        "1_quality": (
            "输出保护(右删失已剔):off2 一致 %s;policy 一致 %s(分歧样本 %d 个,"
            "其首异中位 %s);dither 一致 %s(分歧 %d,%s)—— n=8 方向性"
            % (d_off2["exact_match"], d_pol["exact_match"], d_pol["n_divergent"],
               d_pol["first_div_median_divergent_only"], d_dit["exact_match"],
               d_dit["n_divergent"], d_dit["first_div_median_divergent_only"])),
    }
    dst = os.path.join(OUT, "p88_same_request.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
