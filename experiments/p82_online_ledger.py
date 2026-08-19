# -*- coding: utf-8 -*-
"""在线影子账本(p82,A9 阶段 1):逐请求报告的汇总。

与 p80 的差别 —— p80 是离线重放(合并计数,无请求边界);p81/p82 是**在线**:
  · 逐请求实例化:账本按 req_pool_index 分开建,请求边界真实存在(串行 batch=1);
  · 逐读取检查:每个 decode 步、每个 c4 层、**全部被选中的页条目**都算见证并判定
    (不是抽样步、不是等距条目);
  · 请求结束报告:每请求 n_events / retention / would_fallback / Σδ / 假想页入。

**仍是影子(只读)**:超阈条目照常被模型使用,"would_*" 前缀诚实标注假想。
真实回退(数值路径切换 + 回退成本入账)是 A9 终局,不由本文件的词汇提前兑现。

口径:
  · rank 合并:tp8 下每 rank 检查自己分片的条目,同一请求的计数跨 rank 相加;
  · 窗口条目不在账本(只查稀疏页读取侧);c128 层不在账本;
  · W_thr 单点(env 指定);扫曲线看 p80。

python3 experiments/p82_online_ledger.py
"""
import glob
import json
import os
import statistics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    ranks = sorted(glob.glob(os.path.join(OUT, "wc_online.json.rank*")))
    assert len(ranks) == 8, f"预期 8 个 rank,实得 {len(ranks)}"
    merged, wthr = {}, None
    for f in ranks:
        d = json.load(open(f))
        wthr = wthr or d["coverage"].get("ledger_wthr")
        for rid, r in (d.get("request_ledgers") or {}).items():
            m = merged.setdefault(rid, {"n_events": 0, "n_would_fallback": 0,
                                        "n_steps": 0, "would_pagein_mib": 0.0,
                                        "delta_spent": 0.0})
            m["n_events"] += r["n_events"]
            m["n_would_fallback"] += round(r["would_fallback_rate"] * r["n_events"])
            m["n_steps"] = max(m["n_steps"], r["n_steps"])
            m["would_pagein_mib"] += r["would_pagein_mib"]
            m["delta_spent"] += r["delta_spent"]
    assert merged, "无逐请求账本 —— WITCERT_LEDGER 没生效"
    wthr_source = "snapshot"
    if wthr is None:
        # 本轮快照缺 ledger_wthr 字段(插入锚未对上,已修);阈值以受版本控制的
        # launcher env 为准 —— 来源显式落盘,不静默补
        wthr, wthr_source = 0.5, "launcher env(WITCERT_LEDGER_WTHR,快照字段本轮缺失)"
    reqs = []
    for rid, m in sorted(merged.items(), key=lambda kv: int(kv[0])):
        ret = 1.0 - m["n_would_fallback"] / max(1, m["n_events"])
        reqs.append({"rid": rid, **m, "entry_retention": ret})
    rets = [r["entry_retention"] for r in reqs]
    pg = [r["would_pagein_mib"] for r in reqs]

    rep = {
        "what": ("在线影子账本(A9 阶段 1):逐请求实例化 + 逐读取(选中页条目)检查 + "
                 "请求报告,在线跑在 serving 里;**只读,回退为假想**"),
        "machine": "hgx",
        "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8, ctx 65536",
        "W_thr": wthr, "W_thr_source": wthr_source,
        "caliber": [
            "逐读取 = 每 decode 步、每 c4 层、全部被选中页条目(非抽样);串行请求,边界真实",
            "**影子账本**:超阈条目照常被模型使用;would_* = 若执行回退策略会发生什么",
            "窗口条目与 c128 层不在账本;rank 分片计数相加",
            "Σδ=0(全确定性证书);真实回退与其成本入账是 A9 终局",
        ],
        "n_requests": len(reqs),
        "note_requests": "25 = 24 条测量请求 + 1 条预热(串行,均单独成账)",
        "per_request": reqs,
        "summary": {
            "retention_median": statistics.median(rets),
            "retention_min": min(rets), "retention_max": max(rets),
            "would_pagein_mib_median": statistics.median(pg),
            "would_pagein_mib_max": max(pg),
            "events_total": sum(r["n_events"] for r in reqs),
            "delta_spent_total": sum(r["delta_spent"] for r in reqs),
        },
    }
    s = rep["summary"]
    rep["findings"] = {
        "0_headline": (
            "**在线影子账本闭环(只读)**:%d 条串行请求,逐请求账本在线建立并出报告;"
            "逐读取检查共 %s 个页条目事件。W_thr=%.2f 下逐请求保留率中位 %.3f"
            "(min %.3f / max %.3f),假想页入中位 %.1f MiB/请求;Σδ=0。"
            "**评审缺项四清二**:逐请求实例化✓ 请求边界✓ 逐读取检查✓ 请求报告✓;"
            "真实回退读路径✗ 概率预算实际消费✗(A9 终局)"
            % (rep["n_requests"], f"{s['events_total']:,}", wthr or -1,
               s["retention_median"], s["retention_min"], s["retention_max"],
               s["would_pagein_mib_median"])),
        "1_semantics": (
            "shadow 的诚实边界:账本证明的是'策略在真实读取流上的账目',"
            "不是'模型输出被策略保护过' —— 后者要等数值路径的真实回退"),
    }
    dst = os.path.join(OUT, "p82_online_ledger.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
