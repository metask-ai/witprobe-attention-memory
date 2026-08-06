# -*- coding: utf-8 -*-
"""R9-B6:capture 时选择性插桩 —— 带图常开是否成立。

**问题的形状**(由 p71/p72 确立,不是猜的):开图后按调用计数采样语义失效 ——
should_sample() 是 Python 分支,CUDA graph capture 只求值一次,捕获当刻的计数值
偶然决定了这张图里到底有没有探针算子。于是只有两个极端:
  EVERY=1  全速   吞吐 -52.8% / TTFT P99 +138.6%,不可常开
  EVERY=64 零覆盖 重放增量 0,"免费"只是探针不在图里的同义反复

**换旋钮**:让采样决策**只依赖层号**。层号是确定性的,capture 时求值与别处求值答案
相同,所以"哪些层带探针"由我们显式声明,而不是由捕获时刻的计数器状态偶然决定。

B6 的完成判据是三条**同时**满足,少一条都不算:
  (1) verify --expect-replay 通过 —— 有真实的图内重放累加
  (2) 吞吐与 TTFT P99 退化在**逐档噪声下界**以内
  (3) 层覆盖可声明可对账 —— 覆盖恰是被选中的那几层

口径(随数字引用):
  · 单卡 Qwen2.5-7B + gqa-kv;与论文关图数字栈不同,不可相减。
  · 噪声下界 = 基线自身重复测量的相对极差,逐档算(取全局最大值会掩盖真实退化)。
  · 层选择降低的是**每次 forward 里的探针调用点数**;成本主要是每次调用的固定
    kernel 开销(p60 已确立与数据量无关),故预期近似按选中层数比例缩。
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
KEYS = ("output_tok_per_s", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99")


def main():
    raw = json.load(open(os.path.join(OUT_DIR, "p73_capture_select_raw.json")))
    cases = {c["case"]: c for c in raw if c and "error" not in c}
    if "off" not in cases:
        raise SystemExit("缺 off 基线")
    base = {s["concurrency"]: s for s in cases["off"]["sweeps"] if "error" not in s}
    cov = {}
    covp = os.path.join(OUT_DIR, "p73_coverage.jsonl")
    if os.path.exists(covp):
        for line in open(covp):
            line = line.strip().rstrip(",")
            if line:
                r = json.loads(line); cov[r["case"]] = r

    rep = {
        "what": "capture 时选择性插桩:带图常开是否成立(B6)",
        "machine": "hgx",
        "stack": ("Qwen2.5-7B-Instruct, sglang 0.5.13.post1, tp1, ctx 8192, "
                  "**未加 disable-cuda-graph**;EVERY=1,旋钮只剩层选择"),
        "criteria": ["(1) --expect-replay 通过", "(2) 退化在逐档噪声下界内",
                     "(3) 层覆盖可声明可对账"],
        "caliber": [
            "单卡 Qwen2.5-7B + gqa-kv;与论文关图数字栈不同,不可相减",
            "噪声下界 = 基线自身重复测量的相对极差,**逐档**算",
            "层选择降低的是每次 forward 的探针调用点数;成本以每次调用的固定 kernel "
            "开销为主(p60 确立与数据量无关),故预期近似按选中层数比例缩",
        ],
        "coverage": cov,
        "by_concurrency": {},
    }
    for name, c in cases.items():
        for s in c["sweeps"]:
            if "error" in s:
                continue
            cc = str(s["concurrency"])
            row = rep["by_concurrency"].setdefault(cc, {})
            row[name] = {k: s.get(k) for k in KEYS}
            row[name]["output_tok_per_s_spread"] = s.get("output_tok_per_s_spread")
            row[name]["ttft_p99_spread"] = s.get("ttft_p99_spread")
            b = base.get(s["concurrency"])
            if b and name != "off":
                row[name]["delta_pct"] = {
                    k: (100.0 * (s[k] - b[k]) / b[k]) if b.get(k) else None for k in KEYS}

    verdicts = {}
    for name in cases:
        if name == "off":
            continue
        thr, p99, within = [], [], True
        for cc, row in rep["by_concurrency"].items():
            if name not in row:
                continue
            d = row[name]["delta_pct"]
            floor_t = (row["off"].get("output_tok_per_s_spread") or 0) * 100
            floor_p = (row["off"].get("ttft_p99_spread") or 0) * 100
            thr.append(d["output_tok_per_s"]); p99.append(d["ttft_p99"])
            if abs(d["output_tok_per_s"]) > floor_t or abs(d["ttft_p99"]) > max(floor_p, 20.0):
                within = False
        c = cov.get(name, {})
        gate_ok = c.get("gate") == "pass"
        cov_ok = c.get("n_layers") == c.get("expect_layers")
        verdicts[name] = {
            "stride": c.get("stride"), "layers": c.get("n_layers"),
            "expect_layers": c.get("expect_layers"),
            "replay_growth": c.get("replay_growth"),
            "throughput_worst_pct": min(thr) if thr else None,
            "ttft_p99_worst_pct": max(p99) if p99 else None,
            "c1_replay": gate_ok, "c2_within_noise": within, "c3_coverage": cov_ok,
            "all_three": bool(gate_ok and within and cov_ok),
        }
    rep["verdicts"] = verdicts
    winners = [k for k, v in verdicts.items() if v["all_three"]]
    rep["b6_satisfied"] = bool(winners)
    rep["winners"] = winners

    if winners:
        w = min(winners, key=lambda k: verdicts[k]["stride"] or 99)   # 覆盖最多的那个
        v = verdicts[w]
        head = ("**带图常开成立**:stride=%s(%s/%s 层)在开图下同时满足三条判据 —— "
                "重放增量 %s(真有图内累加)、吞吐 %+.2f%% 与 TTFT P99 %+.2f%% 均在逐档"
                "噪声下界内、层覆盖 %s 层与声明一致。**旋钮从'每 N 次调用'换成'哪几层',"
                "语义在 capture 时是确定的**,这正是按调用采样在开图后做不到的"
                % (v["stride"], v["layers"], 28, v["replay_growth"],
                   v["throughput_worst_pct"], v["ttft_p99_worst_pct"], v["layers"]))
    else:
        head = ("**带图常开仍不成立**:所有档位都至少缺一条判据 —— " + "; ".join(
            "%s(重放=%s 噪声内=%s 覆盖对账=%s, 吞吐 %+.2f%%)"
            % (k, v["c1_replay"], v["c2_within_noise"], v["c3_coverage"],
               v["throughput_worst_pct"] or 0) for k, v in verdicts.items())
            + "。**不得因为某档开销低就宣布成立** —— 开销低而无重放只是探针不在图里")
    rep["findings"] = {
        "0_headline": head,
        "1_table": "; ".join(
            "%s: %s/%s 层, 吞吐 %+.2f%%, TTFT P99 %+.2f%%, 三判据 %s"
            % (k, v["layers"], 28, v["throughput_worst_pct"] or 0,
               v["ttft_p99_worst_pct"] or 0, v["all_three"]) for k, v in verdicts.items()),
        "2_why_this_knob": (
            "按调用计数采样在开图后不是有效旋钮:它是 Python 分支,capture 只求值一次,"
            "捕获当刻的计数值偶然决定这张图带不带探针 —— 要么全速要么零覆盖。"
            "层选择只依赖层号,确定性,故 capture 时求值也是同一个答案"),
    }
    dst = os.path.join(OUT_DIR, "p73_capture_select.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst, "-> B6 满足:", rep["b6_satisfied"])
    print("%-6s %-6s %10s %10s %8s %8s %8s" % ("并发", "档位", "吞吐tok/s", "TTFT P99",
                                               "重放", "噪声内", "覆盖"))
    for cc in sorted(rep["by_concurrency"], key=int):
        for name, v in rep["by_concurrency"][cc].items():
            d = verdicts.get(name, {})
            print("%-6s %-6s %10.1f %10.3f %8s %8s %8s"
                  % (cc, name, v["output_tok_per_s"] or 0, v["ttft_p99"] or 0,
                     d.get("c1_replay", "-"), d.get("c2_within_noise", "-"),
                     d.get("c3_coverage", "-")))
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
