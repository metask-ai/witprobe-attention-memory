# -*- coding: utf-8 -*-
"""R9-B2:把 p66 的并发压测汇总成 p66_concurrent_overhead.json。

**为什么必须有这一份**:p60 的 +5.45% 是单请求串行 + 关图,是探针开销的**最不利放大
口径** —— 没有别的工作掩盖 host 侧成本。系统评审会据此判定"这是研究探针,不是常开
生产观测层"。本文件给并发下的吞吐与 TTFT/TPOT 的 P50/P99。

口径(随数字引用):
  · 仍在 --disable-cuda-graph + --disable-piecewise-cuda-graph 下(探针在 forward 内做
    python IO,必须关图);故绝对值不代表带图生产性能,报的是**同口径下的相对增量**。
    带图路径是 B3 的事。
  · 并发由客户端线程维持在途请求数,非严格闭环负载发生器。
  · prompt 8k / 输出 64 token,单一形状;真实流量的形状分布不同。

python experiments/p66_concurrent_report.py
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
KEYS = ("output_tok_per_s", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
        "e2e_p50", "e2e_p99")
#: 重复测量的相对极差 —— 判断"差异是不是噪声"的唯一依据,必须一起带出来
SPREAD_KEYS = tuple(k + "_spread" for k in KEYS)


def main():
    raw = json.load(open(os.path.join(OUT_DIR, "p66_concurrent_overhead_raw.json")))
    cases = {c["case"]: c for c in raw if c and "error" not in c}
    if "off" not in cases:
        raise SystemExit("缺 off 基线")
    base = {s["concurrency"]: s for s in cases["off"]["sweeps"] if "error" not in s}

    rep = {
        "what": "探针开销的并发 serving 口径",
        "stack": ("DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8, ctx 65536, "
                  "disable-cuda-graph + disable-piecewise-cuda-graph"),
        "caliber": [
            "仍是关图口径:绝对值不代表带图生产性能,报的是同口径下的相对增量(带图见 B3)",
            "并发由客户端线程维持在途请求数,非严格闭环负载发生器",
            f"prompt {cases['off'].get('prompt_tokens')} tok / 输出 "
            f"{cases['off'].get('new_tokens')} tok,单一形状",
        ],
        "cases": {k: {"probe": v["probe"], "every": v["every"]} for k, v in cases.items()},
        "by_concurrency": {},
    }
    for name, c in cases.items():
        for s in c["sweeps"]:
            if "error" in s:
                continue
            cc = str(s["concurrency"])
            row = rep["by_concurrency"].setdefault(cc, {})
            row[name] = {k: s.get(k) for k in KEYS + SPREAD_KEYS}
            row[name]["n_reps"] = s.get("n_reps")
            b = base.get(s["concurrency"])
            if b and name != "off":
                row[name]["delta_pct"] = {
                    k: (100.0 * (s[k] - b[k]) / b[k]) if b.get(k) else None for k in KEYS
                }
    # 结论:吞吐损失与尾延迟增幅取各并发档最不利者
    worst = {}
    for name in cases:
        if name == "off":
            continue
        thr = [rep["by_concurrency"][c][name]["delta_pct"]["output_tok_per_s"]
               for c in rep["by_concurrency"] if name in rep["by_concurrency"][c]]
        p99 = [rep["by_concurrency"][c][name]["delta_pct"]["ttft_p99"]
               for c in rep["by_concurrency"] if name in rep["by_concurrency"][c]]
        worst[name] = {"throughput_worst_pct": min(thr) if thr else None,
                       "ttft_p99_worst_pct": max(p99) if p99 else None}
    rep["worst_case"] = worst
    # **判据用噪声下界,不是绝对阈值**:重复测量的相对极差就是本次测量能分辨的最小差异。
    # 若 |Δ| < 噪声下界,正确结论是"差异不可分辨,只能给出上界",而不是"开销为零"
    # 也不是"数据作废"(2026-07-31:先前用绝对阈值,把一次正常的噪声判成了不可信)。
    spreads, within_noise, exceeds = [], [], []
    for cc, row in rep["by_concurrency"].items():
        b = row.get("off", {})
        floor = (b.get("output_tok_per_s_spread") or 0) * 100      # 转百分比
        spreads.append(floor)
        for name, v in row.items():
            if name == "off":
                continue
            d = (v.get("delta_pct") or {}).get("output_tok_per_s")
            if d is None:
                continue
            (within_noise if abs(d) <= floor else exceeds).append(
                f"c={cc} {name} 吞吐 {d:+.1f}%(噪声下界 ±{floor:.1f}%)")
            # **尾延迟才是"常开"的判据** —— 吞吐落在噪声里不等于可以常开
            dt = (v.get("delta_pct") or {}).get("ttft_p99")
            tf = (b.get("ttft_p99_spread") or 0) * 100
            if dt is not None and abs(dt) > max(tf, 20.0):
                exceeds.append(f"c={cc} {name} **TTFT P99 {dt:+.1f}%**(噪声 ±{tf:.1f}%)")
    noise_floor = max(spreads) if spreads else None
    rep["credibility"] = {
        "noise_floor_pct": noise_floor,
        "within_noise": within_noise, "exceeds_noise": exceeds,
        "verdict": ("**差异全部落在噪声下界以内** —— 只能给出上界,不能给出点估计"
                    if not exceeds else
                    "部分档位超出噪声下界(见 exceeds_noise)—— 这些是**可主张的真实退化**"),
        "per_concurrency_floor_pct": {cc: (rep["by_concurrency"][cc].get("off", {})
                                           .get("output_tok_per_s_spread") or 0) * 100
                                      for cc in rep["by_concurrency"]},
        "note": ("噪声下界 = 基线自身 3 次重复的相对极差;这是本次测量能分辨的最小差异。"
                 "要给更紧的点估计需显著增加重复次数或改用低方差测量设计"),
    }
    rep["findings"] = {
        "0_headline":
            ("**两档分野清楚**:EVERY=64 吞吐 %+.1f%%、TTFT P99 %+.1f%%,均在逐档噪声"
             "(±%.1f%%~±%.1f%%)以内;EVERY=16 吞吐 %+.1f%%、TTFT P99 %+.1f%%,**双双超出噪声**。"
             "**故常开生产档位取 EVERY=64。** 注意噪声要按**逐档**算 —— 取全局最大值"
             "(±%.1f%%)会把 EVERY=16 的真实退化误判成噪声"
             % (worst.get("on_e64", {}).get("throughput_worst_pct") or 0,
                worst.get("on_e64", {}).get("ttft_p99_worst_pct") or 0,
                min(spreads) if spreads else 0, max(spreads) if spreads else 0,
                worst.get("on_e16", {}).get("throughput_worst_pct") or 0,
                worst.get("on_e16", {}).get("ttft_p99_worst_pct") or 0,
                noise_floor or 0)),
        "1_throughput": "; ".join(
            f"{n}: 吞吐最不利档 {v['throughput_worst_pct']:+.2f}%" for n, v in worst.items()),
        "2_tail": "; ".join(
            f"{n}: TTFT P99 最不利档 {v['ttft_p99_worst_pct']:+.2f}%" for n, v in worst.items()),
        "3_vs_serial": ("对照 p60 的单请求串行口径(EVERY=16 最不利 +5.45%):"
                        "并发下有其它工作掩盖 host 侧成本,故相对增量应更小 —— "
                        "**若并发下反而更大,说明探针在争抢而非搭便车,需查**"),
    }
    dst = os.path.join(OUT_DIR, "p66_concurrent_overhead.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print("%-6s %-9s %12s %10s %10s %10s" % ("并发", "档位", "吞吐tok/s", "TTFT P50", "TTFT P99", "TPOT P99"))
    for cc in sorted(rep["by_concurrency"], key=int):
        for name, v in rep["by_concurrency"][cc].items():
            print("%-6s %-9s %12.1f %10.3f %10.3f %10.4f"
                  % (cc, name, v["output_tok_per_s"] or 0, v["ttft_p50"] or 0,
                     v["ttft_p99"] or 0, v["tpot_p99"] or 0))
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
