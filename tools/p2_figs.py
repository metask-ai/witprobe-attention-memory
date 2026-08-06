# -*- coding: utf-8 -*-
"""论文2 图表生成器 —— **图中每个数据点直读 experiments/out/*.json**,
与 canon 同纪律:不许手写数字。产出:

  papers/p2-attention-memory/figs/localization.pdf   九轮消融定位图
  papers/p2-attention-memory/figs/sixpath.tex        六路径税表(机器生成)

python3 tools/p2_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "out")
FIGS = os.path.join(ROOT, "papers", "p2-attention-memory", "figs")
os.makedirs(FIGS, exist_ok=True)


def J(name):
    return json.load(open(os.path.join(OUT, name), encoding="utf-8"))


def _ev(d, key):
    return ((d.get("snapshot") or {}).get(key) or {}).get(
        "ring_stats", {}).get("n_evict", 0) or 0


# ---- 统一视觉系统(2026-08-06 视觉升级):四图一种语言 ----
TEAL, ORANGE, RED, GRAY, INK = "#0b7a6e", "#e8590c", "#c92a2a", "#8b939b", "#1f2529"
PALE_TEAL, PALE_RED = "#0b7a6e22", "#c92a2a14"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5,
    "axes.edgecolor": "#c6ccd2", "axes.linewidth": 0.8,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#5c646b", "ytick.color": "#5c646b",
    "grid.color": "#e9edf0", "grid.linewidth": 0.7,
})


def _style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, zorder=0)
        ax.set_axisbelow(True)


FIGDATA = json.load(open(os.path.join(
    ROOT, "papers", "p2-attention-memory", "figdata.json"), encoding="utf-8"))


def fig_math():
    """数学主图 v2:左 = 覆盖率跃升(棒棒糖+注释);右 = Pareto(空洞域
    着色 + 前沿虚线 + 预算注释框)。数据源 figdata.json(figdata⊆prose)。"""
    cv, pa = FIGDATA["coverage"], FIGDATA["pareto"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.35),
                                   gridspec_kw={"wspace": 0.32})
    # 左:棒棒糖跃升
    _style(ax1)
    y0, y1 = cv["old_radius_pct"], cv["new_radius_pct"]
    ax1.vlines([0, 1], 0, [y0, y1], color=[GRAY, TEAL], lw=7, zorder=3,
               capstyle="round")
    ax1.scatter([0, 1], [y0, y1], s=[160, 220], color=[GRAY, TEAL], zorder=4)
    ax1.annotate("", xy=(0.93, y1 - 3), xytext=(0.07, y0 + 3),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color=ORANGE,
                                 connectionstyle="arc3,rad=-0.25"))
    ax1.text(0.42, 44, "$%.1f\\times$" % cv["ratio"], fontsize=15,
             fontweight="bold", color=ORANGE, ha="center")
    ax1.text(0, y0 + 4.5, "%.1f%%" % y0, ha="center", fontsize=11,
             fontweight="bold", color=GRAY)
    ax1.text(1, y1 + 4.5, "%.1f%%" % y1, ha="center", fontsize=12,
             fontweight="bold", color=TEAL)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["prior\nradius", "finite-product\nradius"])
    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(0, 84)
    ax1.set_ylabel("pre-authorization rate (%)")
    ax1.set_title("authorization coverage")
    ax1.text(0.5, -0.30,
             "$%d\\;/\\;%s$ authorized entries violate their radius"
             % (cv["violations"], format(cv["authorized_entries"], ",")),
             transform=ax1.transAxes, ha="center", fontsize=8.5, color=INK)
    # 右:Pareto
    _style(ax2, grid="both")
    ax2.axhspan(1.0, 1.85, color=PALE_RED, zorder=0)
    ax2.text(3, 1.68, "vacuous region (bound $>$ 1)", fontsize=8,
             color=RED, va="top")
    ps = sorted(pa["points"], key=lambda p: p["retention_pct"])
    ax2.plot([p["retention_pct"] for p in ps], [p["tv_bound"] for p in ps],
             ls="--", lw=1.1, color=GRAY, zorder=2)
    for p in ps:
        good = not p["vacuous"]
        c = TEAL if good else RED
        ax2.scatter(p["retention_pct"], p["tv_bound"], s=170, color=c,
                    marker="o" if good else "X", zorder=4,
                    edgecolor="white", linewidth=1.2)
        ax2.annotate(
            "%s\nTV $\\leq$ %.2f\nkeep %.1f%%" % (
                p["label"].replace(" (bound vacuous)", ""),
                p["tv_bound"], p["retention_pct"]),
            xy=(p["retention_pct"], p["tv_bound"]),
            xytext=(10, -34 if good else 12), textcoords="offset points",
            fontsize=8, color=c, fontweight="bold")
    ax2.set_xlabel("compression retention (%)")
    ax2.set_ylabel("per-step attention-TV bound")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1.85)
    ax2.set_title("ledger working points")
    ax2.text(0.985, 0.06,
             "$\\delta_{req}=%.2f$,  max request spend $%.6f$"
             % (pa["delta_req"], pa["max_request_spend"]),
             transform=ax2.transAxes, ha="right", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c6ccd2"))
    fig.tight_layout()
    out = os.path.join(FIGS, "math.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("→", out)


def fig_overhead():
    """开销 v3(评审 P1):双面板 —— 吞吐(symlog 条形)+ TTFT P99。
    数字全部正文冻结;28/28 融合档无 TTFT 读数即不画(不编数)。"""
    rows = [   # (short, throughput_pct, ttft_pct_or_None, mode)
        ("fused 1/28\ngraphs on",   -0.94,  -0.14,  "on"),
        ("fused 28/28\ngraphs on", -10.16,  None,   "on"),
        ("sampled 1/64\ngraphs off", -3.9,  10.5,   "off"),
        ("sampled 1/16\ngraphs off", -19.6, 140.8,  "off"),
        ("serial, full\nsampling", -113.24, None,   "serial"),
    ]
    color = {"on": TEAL, "off": GRAY, "serial": "#4a5157"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 2.9),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    _style(ax, grid="x")
    ys = list(range(len(rows)))[::-1]
    for y, (lab, v, tt, mode) in zip(ys, rows):
        ax.barh(y, v, height=0.6, color=color[mode], zorder=3,
                alpha=0.92 if mode == "on" else 0.78)
        ax.text(v * 1.06 if v < -2 else v - 0.5, y, "$%+.2f\\%%$" % v,
                va="center", ha="right", fontsize=8.6, fontweight="bold",
                color=color[mode])
        ax.text(1.2, y, lab.replace("\n", ", "), va="center", ha="left",
                fontsize=8.2, color=INK)
    ax.axvspan(-1.5, 0, color=PALE_TEAL, zorder=1)
    ax.text(-1.4, -0.42, "noise floor", fontsize=7.2, color=TEAL, ha="left")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlim(-150, 90)
    ax.set_xticks([-100, -30, -10, -3, -1, 0])
    ax.set_xticklabels(["$-100$", "$-30$", "$-10$", "$-3$", "$-1$", "$0$"])
    ax.set_yticks([])
    ax.set_xlabel("throughput change (%, symlog)")
    ax.set_title("throughput cost")
    # TTFT 面板(有读数的三档)
    _style(ax2, grid="y")
    tt_rows = [(lab, tt, mode) for lab, _v, tt, mode in rows if tt is not None]
    xs = range(len(tt_rows))
    for x, (lab, tt, mode) in zip(xs, tt_rows):
        ax2.bar(x, tt, width=0.55, color=color[mode], zorder=3,
                alpha=0.92 if mode == "on" else 0.78)
        ax2.text(x, tt + (4 if tt >= 0 else -1), "$%+.2f\\%%$" % tt,
                 ha="center", fontsize=8.6, fontweight="bold",
                 color=color[mode],
                 va="bottom" if tt >= 0 else "top")
    ax2.axhline(0, color="#c6ccd2", lw=0.8)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([lab for lab, _t, _m in tt_rows], fontsize=7.6)
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_ylim(-2, 260)
    ax2.set_yticks([0, 10, 100])
    ax2.set_yticklabels(["$0$", "$+10$", "$+100$"])
    ax2.set_ylabel("TTFT P99 change (%, symlog)")
    ax2.set_title("tail latency cost")
    fig.tight_layout()
    out = os.path.join(FIGS, "overhead.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("→", out)


def fig_budget():
    """伸缩预算曲线:delta_i = delta_req/((i+1)(i+2)),部分和恒 < delta_req
    —— 账本'未知长度仍 sound'的核心公式可视化(纯解析,与正文公式同源)。"""
    import numpy as np
    dreq = FIGDATA["pareto"]["delta_req"]
    i = np.arange(0, 40)
    di = dreq / ((i + 1) * (i + 2))
    csum = np.cumsum(di)
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    _style(ax, grid="y")
    ax.axhline(dreq, color=RED, lw=1.2, ls="--", zorder=2)
    ax.text(38.5, dreq * 1.015, "$\\delta_{req}$", color=RED, fontsize=10,
            ha="right", va="bottom", fontweight="bold")
    ax.fill_between(i, 0, csum, color=PALE_TEAL, zorder=1)
    ax.plot(i, csum, color=TEAL, lw=2, zorder=3)
    ax.bar(i[:12], di[:12], width=0.55, color=ORANGE, alpha=0.85, zorder=3)
    ax.annotate("per-event spend\n$\\delta_i = \\delta_{req}/((i{+}1)(i{+}2))$",
                xy=(2, di[2]), xytext=(9, 0.0062), fontsize=8.5, color=ORANGE,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("cumulative spend:\nsound at any length,\nnever exceeds "
                "$\\delta_{req}$",
                xy=(30, csum[30]), xytext=(16, 0.0035), fontsize=8.5,
                color=TEAL,
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("certified probabilistic event $i$ within a request")
    ax.set_ylabel("failure budget")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, dreq * 1.12)
    ax.set_title("the telescoping request budget")
    fig.tight_layout()
    p = os.path.join(FIGS, "budget.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("→", p)


def fig_localization():
    """九轮消融:acc vs 配置,按'哪个池在驱逐'着色 —— 双解离一图读出。"""
    rows = []  # (label, acc, evict_kind, n_evict_c128, n_evict_c4)
    for rid, tag, label in (
            ("q6o", "graphb2", "serial (W1), ring 513"),
            ("q6n", "graphb2", "W4, both rings 513"),
            ("q6p3", "ring1025", "W4, both rings 1025"),
            ("q6p3", "ring513", "W4, both rings 513 (control)"),
            ("q6p5", "c128small", "W4, evict 128x only"),
            ("q6p5", "c4small", "W4, evict 4x only"),
            ("q6p6", "c128evict", "W4, evict 128x only (rep 2)"),
            ("q6p6", "c4evict", "W4, evict 4x only (rep 2)"),
            ("q6p7", "c128evict", "W4, evict 128x only (rep 3)"),
            ("q6p7", "c4evict", "W4, evict 4x only (rep 3)")):
        d = J("%s_%s.json" % (rid, tag))
        e128, e4 = _ev(d, "pool_swap"), _ev(d, "pool_swap_c4")
        kind = ("none" if e128 == 0 and e4 == 0 else
                "c128" if e4 == 0 else "c4" if e128 == 0 else "both")
        rows.append((label, d["acc"], kind, e128, e4))
    colors = {"none": "#2b8a3e", "c128": "#1971c2", "c4": "#e03131",
              "both": "#e8590c"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ys = range(len(rows))[::-1]
    for y, (label, acc, kind, e128, e4) in zip(ys, rows):
        ax.plot([0.45, acc], [y, y], color="#dee2e6", lw=1, zorder=1)
        ax.scatter([acc], [y], s=46, color=colors[kind], zorder=3)
        ev = ("no eviction" if kind == "none" else
              "%s evictions" % (e128 if kind == "c128" else
                                e4 if kind == "c4" else e128 + e4))
        ax.annotate("%.3f  (%s)" % (acc, ev), (acc, y),
                    textcoords="offset points",
                    xytext=(-6, 0) if acc > 0.9 else (7, 0),
                    ha="right" if acc > 0.9 else "left",
                    va="center", fontsize=7.4, color="#495057")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("needle retrieval accuracy (12 documents)", fontsize=9)
    ax.set_xlim(0.45, 1.13)
    ax.axvline(1.0, color="#adb5bd", lw=0.8, ls=":")
    for k, lab in (("none", "no pool evicts"), ("c128", "only 128x evicts"),
                   ("c4", "only 4x evicts"), ("both", "both evict")):
        ax.scatter([], [], s=46, color=colors[k], label=lab)
    ax.legend(fontsize=7.4, loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGS, "localization.pdf")
    fig.savefig(p)
    print("→", p, "(%d 轮)" % len(rows))


def fig_dose():
    """剂量-响应全景 v2:历史可归因轮(灰)+ W4 对照(青)+ W4 诚实截断
    臂(橙三角)—— 三系共一条曲线 = '同剂量下截断与腐坏等价伤害' 的
    视觉定谳;零驱逐簇 = 认证工作点。每点直读产物 JSON。"""
    hist, hist_conf = [], []
    for rid_tag in (("q6o", "graphb2"), ("q6n", "graphb2"),
                    ("q6p2", "graphb2"), ("q6p3", "ring1025"),
                    ("q6p3", "ring513"), ("q6p5", "c128small"),
                    ("q6p5", "c4small"), ("q6p6", "c4evict"),
                    ("q6p6", "c128evict"), ("q6p7", "c4evict"),
                    ("q6p7", "c128evict"), ("q6p8", "c4evict"),
                    ("q6p8", "full513"), ("q6p9", "c4evict"),
                    ("q6p9", "full513"), ("q6p10", "c4evict"),
                    ("q6p10", "full513"), ("q6p12", "control"),
                    ("q6p13", "control"), ("q6p14", "control")):
        try:
            d = J("%s_%s.json" % rid_tag)
        except FileNotFoundError:
            continue
        e128, e4 = _ev(d, "pool_swap"), _ev(d, "pool_swap_c4")
        (hist if e128 == 0 or e4 == 0 else hist_conf).append((e4, d["acc"]))
    ctl, trunc = [], []
    for rid, tag, bucket in (
            ("q11u", "c4513ctl", "ctl"), ("q11u", "c4513fix", "fix"),
            ("q11v", "ctlrep1", "ctl"), ("q11v", "ctlrep2", "ctl"),
            ("q11w1", "ctl", "ctl"), ("q11w1", "fix", "fix"),
            ("q11w2", "ctl", "ctl"), ("q11w2", "fix", "fix"),
            ("q11w3", "ctl", "ctl"), ("q11w3", "fix", "fix"),
            ("q11w4", "ctl", "ctl"), ("q11w4", "fix", "fix")):
        try:
            d = J("%s_%s.json" % (rid, tag))
        except FileNotFoundError:
            continue
        (ctl if bucket == "ctl" else trunc).append(
            (_ev(d, "pool_swap_c4"), d["acc"]))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    _style(ax, grid="both")
    # 曲线暗示:历史可归因点按剂量排序连浅线
    hs = sorted(hist)
    ax.plot([p[0] for p in hs], [p[1] for p in hs], color="#d4d9dd",
            lw=1.4, zorder=1)
    if hist_conf:
        ax.scatter(*zip(*hist_conf), s=30, facecolors="none",
                   edgecolors="#b9c0c6", zorder=2,
                   label="confounded (both pools evict)")
    ax.scatter(*zip(*hist), s=44, color=GRAY, zorder=3,
               label="attributable rounds (untreated)")
    if ctl:
        ax.scatter(*zip(*ctl), s=68, color=TEAL, zorder=4,
                   edgecolor="white", linewidth=0.9,
                   label="dose-stratified W4 controls")
    if trunc:
        ax.scatter(*zip(*trunc), s=84, marker="^", color=ORANGE, zorder=5,
                   edgecolor="white", linewidth=0.9,
                   label="honest-truncation arms")
    # 注释:认证簇 + 等价伤害
    ax.annotate("certified regime\n(zero eviction,\nidentity-isolated)",
                xy=(8, 1.0), xytext=(70, 0.62), fontsize=8, color=TEAL,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("consistent with one common\ndose--response curve; no treatment\nshift detectable at this sample size",
                xy=(311, 0.667), xytext=(480, 0.82), fontsize=8,
                color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("$4\\times$-ring evictions during run")
    ax.set_ylabel("retrieval accuracy")
    ax.set_ylim(0.33, 1.06)
    ax.axhline(1.0, color="#b9c0c6", lw=0.8, ls=":", zorder=1)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.set_title("degradation tracks eviction dose; no detectable treatment shift")
    fig.tight_layout()
    p = os.path.join(FIGS, "dose.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("→", p, "(hist %d + conf %d + ctl %d + trunc %d)"
          % (len(hist), len(hist_conf), len(ctl), len(trunc)))


def tex_sixpath():
    """六路径**维度热力矩阵**:从 p110_sixpath.json 机器生成。

    升级(2026-08-06 评审):扁平 verdict 单列 → 逐维裁决(capture /
    identity-generation / packed read / evict-reuse / transfer / quality),
    显示每条路径**坏在哪一层**。逐维符号由本函数按判别战役结论编码
    (注释即出处),并断言与产物 verdict 一致 —— 编码漂移即构建失败。"""
    d = J("p110_sixpath.json")
    label = {"cudagraph": "CUDA graphs",
             "radix": "Radix cache",
             "mtp": "MTP (spec.\\ decode)",
             "hisparse": "HiSparse",
             "tpcouple": "TP repeat",
             "pd": "PD disagg."}
    tax = {"PASS": r"\textsc{pass}", "FAIL_METHOD": r"\textsc{fail-m}",
           "BLOCKED_UPSTREAM": r"\textsc{blocked}",
           "PARTIAL": r"\textsc{partial}",
           "NOT_MEASURED": r"\textsc{n/m}"}
    OK, NO, PA, NA = r"\ok", r"\bad", r"\pt", "---"
    dims = {
        "cudagraph": (OK, OK, OK, PA, NA, PA),
        "tpcouple": (OK, OK, OK, NA, NA, OK),
        "radix": (OK, NO, NA, NO, NA, NO),
        "mtp": (OK, OK, OK, OK + r"$^*$", NA, OK),
        "pd": (OK, OK, NO, NA, NO, NO),
        "hisparse": (NA, NA, NA, NA, NA, NA),
    }
    for k, row in dims.items():
        v = d["summary"][k]
        if any(c == NO for c in row):
            assert v in ("FAIL_METHOD", "PARTIAL", "BLOCKED_UPSTREAM"), (k, v)
        if all(c in (OK, OK + r"$^*$", NA) for c in row):
            assert v in ("PASS", "PARTIAL", "BLOCKED_UPSTREAM"), (k, v)
    L = [r"% Generated by tools/p2_figs.py from experiments/out/p110_sixpath.json. Do not edit by hand.",
         r"\newcommand{\ok}{\textcolor{teal!60!black}{$\checkmark$}}",
         r"\newcommand{\bad}{\textcolor{red!60!black}{$\times$}}",
         r"\newcommand{\pt}{\textcolor{orange!80!black}{$\triangle$}}",
         r"\begin{table}[t]\centering\small",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}lcccccc l@{}}", r"\toprule",
         r"Path & Capt. & Id/gen & Packed & Evict & Xfer & Qual. & Verdict\\",
         r"\midrule"]
    for k in ("cudagraph", "tpcouple", "radix", "mtp", "pd", "hisparse"):
        L.append("%s & %s & %s\\\\" % (
            label[k], " & ".join(dims[k]), tax[d["summary"][k]]))
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Path-contract matrix}, generated from run",
          r"artifacts. Cells mark which contract layer each path violates",
          r"($\triangle$ = holds at a stated caliber; --- = n/a or",
          r"unmeasurable; $^*$ = zero-eviction caliber, machine-annotated);",
          r"per-verdict evidence in \S\ref{sec:pathcontract}.}",
          r"\label{tab:sixpath}", r"\end{table}"]
    p = os.path.join(FIGS, "sixpath.tex")
    open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("→", p)


if __name__ == "__main__":
    fig_math()
    fig_overhead()
    fig_budget()
    fig_localization()
    fig_dose()
    tex_sixpath()
    import figprint  # tools/ 同目录;指纹纪律见该模块头注
    figprint.stamp(FIGS, ["math.pdf", "overhead.pdf", "budget.pdf",
                          "localization.pdf", "dose.pdf", "sixpath.tex"])
