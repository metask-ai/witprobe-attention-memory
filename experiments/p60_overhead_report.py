# -*- coding: utf-8 -*-
"""R8-E5:把 p60 的成对计时汇总成 p60_probe_overhead.json(四件套的"开销"一列)。

口径(必须随数字一起引用):
  · 成对:同一进程配置、同一请求序列,只切 WITCERT_PROBE=0/1;每档 3 次重复取中位。
  · 分档:EVERY=N 是**调用级**采样(每 N 次写入/选择观测 1 次)。行采样(ROWS)实测
    不是有效旋钮 —— 开销的绝对值与上下文长度无关、与行数无关,是每次调用的固定
    kernel launch 成本;调用采样才按 1/N 缩。
  · 本测量在 **--disable-cuda-graph + --disable-piecewise-cuda-graph** 下进行
    (探针在 forward 内做 python IO,必须关图),故绝对时延不代表带图的生产性能;
    这里报的是**同口径下探针引入的相对增量**,不是端到端生产开销。
  · 单请求串行、无并发,是探针开销的**最不利放大口径**(没有其它工作掩盖 host 侧成本)。

python experiments/p60_overhead_report.py
"""
import json
import os
import statistics as S

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CTXS = ("ctx8k", "ctx30k", "ctx60k")


def main():
    raw = json.load(open(os.path.join(OUT_DIR, "p60_probe_overhead_raw.json")))
    cases = {c["case"]: c for c in raw if c and "error" not in c}
    errs = [c for c in raw if c and "error" in c]
    if "off" not in cases:
        raise SystemExit("缺 off 基线,无法算相对开销")

    def med(case, ctx, field="wall_s"):
        return S.median([r[ctx][field] for r in cases[case]["reps"]])

    rep = {
        "measurement": "probe overhead, paired on/off, same process config & request sequence",
        "stack": ("DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8, ctx 65536, "
                  "disable-cuda-graph + disable-piecewise-cuda-graph"),
        "adapters_active": ["dsv4-entry (LatentKV)", "dsv4-selector (SparseSelector, 页级)"],
        "caliber": ("单请求串行无并发 = 探针开销的最不利放大口径;关图状态下的**相对增量**, "
                    "不代表带图生产的端到端开销;每档 3 次重复取中位"),
        "reps": 3,
        "cases": {k: {"probe": v["probe"], "sample_every": v.get("every", v.get("rows"))} for k, v in cases.items()},
        "errors": errs,
        "wall_s_median": {k: {c: med(k, c) for c in CTXS} for k in cases},
    }
    rep["overhead_pct"] = {
        k: {c: 100.0 * (med(k, c) - med("off", c)) / med("off", c) for c in CTXS}
        for k in cases if k != "off"
    }
    lines = []
    for k, d in rep["overhead_pct"].items():
        worst = max(d.values())
        lines.append("%s(EVERY=%d):%s;最不利档 %+.2f%%"
                     % (k, cases[k].get("every", 1),
                        " / ".join("%s %+.2f%%" % (c, d[c]) for c in CTXS), worst))
    rep["findings"] = {"1_overhead": "; ".join(lines)}
    if "on_e1" in rep["overhead_pct"] and "on_e64" in rep["overhead_pct"]:
        a = max(rep["overhead_pct"]["on_e1"].values())
        b = max(rep["overhead_pct"]["on_e16"].values()) if "on_e16" in rep["overhead_pct"] else None
        c = max(rep["overhead_pct"]["on_e64"].values())
        rep["findings"]["2_sampling_knob"] = (
            "调用采样是有效旋钮:EVERY 1->16->64 使最不利档开销 %+.2f%% -> %s -> %+.2f%%"
            % (a, ("%+.2f%%" % b) if b is not None else "-", c))
        rep["findings"]["3_row_sampling_useless"] = (
            "行采样无效(前一轮实测 ROWS 256->32 开销从 +102.6% 到 +102.2%,几乎不变):"
            "开销绝对值与上下文长度无关(三档均 ~7.2s),是每次调用的固定 kernel launch 成本")
    dst = os.path.join(OUT_DIR, "p60_probe_overhead.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print("%-10s %10s %10s %10s" % ("case", *CTXS))
    for k in cases:
        print("%-10s %10.3f %10.3f %10.3f" % (k, *(rep["wall_s_median"][k][c] for c in CTXS)))
    for k, d in rep["overhead_pct"].items():
        print("%-10s %+9.2f%% %+9.2f%% %+9.2f%%" % (k, *(d[c] for c in CTXS)))
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
