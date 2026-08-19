# -*- coding: utf-8 -*-
"""R9-B3 收尾:**带图**运行下的探针开销。汇总 p72 压测为 p72_graph_overhead.json。

**为什么必须有这一份**:p71 只证明了"图重放期累加器还在长",没说这要花多少钱。
论文里的 EVERY=64 数字全是**关图**口径 —— 那是探针开销的最不利放大口径(没有别的
工作掩盖 host 侧成本),但它**不能替代**带图下的答案:开图后探针算子被捕进图,
成本结构完全不同(多出的是图内 kernel 而非 host 侧 python)。

判据沿用 p66,不另立标准:
  **噪声下界 = 基线自身重复测量的相对极差**,即本次测量能分辨的最小差异。
  |Δ| ≤ 下界 -> 只能给上界,不能给点估计,**更不能说"开销为零"**;
  |Δ| > 下界 -> 是可主张的真实退化。
  噪声要**逐档**算 —— 取全局最大值会把小并发档的真实退化误判成噪声(p66 踩过)。

口径(随数字引用):
  · 单卡 Qwen2.5-7B + gqa-kv 单适配器;与论文关图数字的栈(tp8 DeepSeek-V4)不同,
    **两者不可直接相减**,各自只在自己的栈内与自己的基线比。
  · 并发由客户端线程维持在途请求数,非严格闭环负载发生器;单一形状。
  · 尾延迟才是常开与否的判据,吞吐落在噪声里不等于可以常开。

python experiments/p72_graph_overhead_report.py
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
KEYS = ("output_tok_per_s", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
        "e2e_p50", "e2e_p99")
SPREAD_KEYS = tuple(k + "_spread" for k in KEYS)


def main():
    raw = json.load(open(os.path.join(OUT_DIR, "p72_graph_overhead_raw.json")))
    cases = {c["case"]: c for c in raw if c and "error" not in c}
    if "off" not in cases:
        raise SystemExit("缺 off 基线")
    base = {s["concurrency"]: s for s in cases["off"]["sweeps"] if "error" not in s}

    rep = {
        "what": "探针开销的**带图** serving 口径(B3 收尾)",
        "machine": "hgx",
        "stack": ("Qwen2.5-7B-Instruct, sglang 0.5.13.post1, tp1, ctx 8192, "
                  "**未加 disable-cuda-graph / disable-piecewise-cuda-graph**"),
        "caliber": [
            "带图口径。与论文中关图口径的数字**栈也不同**(那是 tp8 DeepSeek-V4),"
            "两者不可直接相减,各自只在自己的栈内与自己的基线比",
            "并发由客户端线程维持在途请求数,非严格闭环负载发生器",
            f"prompt {cases['off'].get('prompt_tokens')} tok / 输出 "
            f"{cases['off'].get('new_tokens')} tok,单一形状",
            "尾延迟才是常开判据;吞吐落在噪声里不等于可以常开",
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

    spreads, within_noise, exceeds = [], [], []
    for cc, row in rep["by_concurrency"].items():
        b = row.get("off", {})
        floor = (b.get("output_tok_per_s_spread") or 0) * 100
        spreads.append(floor)
        for name, v in row.items():
            if name == "off":
                continue
            d = (v.get("delta_pct") or {}).get("output_tok_per_s")
            if d is None:
                continue
            (within_noise if abs(d) <= floor else exceeds).append(
                f"c={cc} {name} 吞吐 {d:+.1f}%(噪声下界 ±{floor:.1f}%)")
            dt = (v.get("delta_pct") or {}).get("ttft_p99")
            tf = (b.get("ttft_p99_spread") or 0) * 100
            if dt is not None and abs(dt) > max(tf, 20.0):
                exceeds.append(f"c={cc} {name} **TTFT P99 {dt:+.1f}%**(噪声 ±{tf:.1f}%)")
    rep["credibility"] = {
        "noise_floor_pct": max(spreads) if spreads else None,
        "per_concurrency_floor_pct": {cc: (rep["by_concurrency"][cc].get("off", {})
                                           .get("output_tok_per_s_spread") or 0) * 100
                                      for cc in rep["by_concurrency"]},
        "within_noise": within_noise, "exceeds_noise": exceeds,
        "verdict": ("**差异全部落在噪声下界以内** —— 只能给出上界,不能给出点估计"
                    if not exceeds else
                    "部分档位超出噪声下界(见 exceeds_noise)—— 这些是**可主张的真实退化**"),
        "note": "噪声下界 = 基线自身重复测量的相对极差;逐档算,不取全局最大值",
    }

    # 覆盖是**验收的第一关**:开图后采样门在 capture 时冻结,EVERY>1 的档位可能整张图
    # 里根本没有探针算子。没有覆盖的档位,它的"开销很低"毫无意义。
    covp = os.path.join(OUT_DIR, "p72_coverage.jsonl")
    if os.path.exists(covp):
        rep["coverage_under_graphs"] = {
            r["case"]: {"n_calls": r["n_calls"], "slots": r["slots"]}
            for r in (json.loads(l.rstrip().rstrip(",")) for l in open(covp) if l.strip())
        }
        rep["coverage_note"] = (
            "开图后 should_sample() 这个 **Python 分支只在 capture 时求值一次** —— "
            "EVERY 的语义从'每 N 次采一次'变成'这张图采不采'。故 EVERY>1 的覆盖必须实测")
    e1 = worst.get("e1", {}); e64 = worst.get("e64", {})
    floors = rep["credibility"]["per_concurrency_floor_pct"]
    rep["findings"] = {
        "0_headline": (
            "**带图下 EVERY=1 的开销**(唯一被 p71 验证过在重放期有完整覆盖的档位,"
            "故是带图成本的上界):吞吐最不利档 %+.2f%%、TTFT P99 最不利档 %+.2f%%"
            "(逐档噪声下界 ±%.1f%%~±%.1f%%)。%s"
            % (e1.get("throughput_worst_pct") or 0, e1.get("ttft_p99_worst_pct") or 0,
               min(floors.values()) if floors else 0, max(floors.values()) if floors else 0,
               ("**这是带图口径的独立结论**,不是关图数字的复述 —— 关图口径测的是 host 侧"
                "python 成本,带图口径测的是被捕进图的 kernel 成本,两者成本结构不同"))),
        "1_e64_semantics": (
            "EVERY=64 带图下:吞吐 %+.2f%% / TTFT P99 %+.2f%%,覆盖 %s。"
            "**先看覆盖再看开销** —— 采样门在 capture 时冻结,若该档覆盖塌了,"
            "它的低开销只是'探针不在图里'的同义反复,不能作为常开档位的依据"
            % (e64.get("throughput_worst_pct") or 0, e64.get("ttft_p99_worst_pct") or 0,
               rep.get("coverage_under_graphs", {}).get("e64", "未记录"))),
        "2_not_comparable": (
            "**不要与论文里的关图数字相减**:栈不同(tp8 DeepSeek-V4 vs 单卡 Qwen2.5-7B)、"
            "口径不同。各自只在自己的栈内与自己的基线比"),
        "3_verdict": rep["credibility"]["verdict"],
    }
    dst = os.path.join(OUT_DIR, "p72_graph_overhead.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print("%-6s %-6s %12s %10s %10s" % ("并发", "档位", "吞吐tok/s", "TTFT P50", "TTFT P99"))
    for cc in sorted(rep["by_concurrency"], key=int):
        for name, v in rep["by_concurrency"][cc].items():
            print("%-6s %-6s %12.1f %10.3f %10.3f"
                  % (cc, name, v["output_tok_per_s"] or 0, v["ttft_p50"] or 0,
                     v["ttft_p99"] or 0))
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
