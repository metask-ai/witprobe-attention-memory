# -*- coding: utf-8 -*-
"""R8-E5:平台覆盖矩阵(p61)——"一把尺子插进多少种模型"的单一事实来源。

汇总各架构探针产物,生成 experiments/out/p61_platform_matrix.json:每行 = (模型, 架构族,
记忆对象, 探针类, 层覆盖, 验收结论, 关键读数)。论文2 的覆盖表直接由它生成。

**收录门槛**:必须通过 witcert.probe 的验收门(覆盖->量纲->soundness)。
没过门的行不许进矩阵 —— 未验收的数字等于没有。

python experiments/p61_platform_matrix.py
"""
import glob
import json
import os
import statistics as S
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "experiments", "out")
sys.path.insert(0, os.path.join(ROOT, "src"))
from witcert.probe import verify  # noqa: E402

# (标签, 快照通配, 模型, 架构族, 记忆对象, 探针类, 路径, 期望该路径层数, 备注)
ROWS = [
    ("qwen7b", "wc_gqa_qwen7b.json.rank*", "Qwen2.5-7B-Instruct", "GQA",
     "K/V cache(GQA)", "KV", "mha", 28, "通用 GQA 适配器,零模型专用代码,首次接入即通过"),
    ("llama8b", "wc_gqa_llama8b.json.rank*", "Meta-Llama-3.1-8B-Instruct", "GQA",
     "K/V cache(GQA)", "KV", "mha", 32, "同上"),
    ("v4_entry", "wc_fw.json.rank0", "DeepSeek-V4-Flash-FP8", "MLA+压缩条目+稀疏索引",
     "压缩条目 448 nope FP8 + 64 rope BF16", "LatentKV", "swa_norm_rope", 43,
     "与手写探针等价性 <=0.04%"),
    ("v4_sel", "wc_fw.json.rank0", "DeepSeek-V4-Flash-FP8", "MLA+压缩条目+稀疏索引",
     "C4 索引器(**页级** top-512)", "SparseSelector", "c4", 21,
     "与 entry 探针同一次运行采到"),
    ("glm_mla", "wc_glm.json.rank1", "GLM-5.2-W4AFP8", "MLA+DSA",
     "latent 512 + k_pe 64", "LatentKV", "mla", 78, "生产量化栈 tp8+ep8+dp8,全 78 层"),
    ("glm_dsa", "wc_glm.json.rank1", "GLM-5.2-W4AFP8", "MLA+DSA",
     "DSA 索引器(**token 级** top-2048)", "SparseSelector", "ragged", 21,
     "与 latent 探针同一次运行采到;paged 路径同样 21 层"),
    ("v2lite", "wc_v2lite.json.rank0", "DeepSeek-V2-Lite", "MLA(无稀疏索引)",
     "latent 512 + k_pe 64", "LatentKV", "mla", 27,
     "**第五个架构:零新代码接入** —— 三步的第一步就发现写入口已被 mla-latent 覆盖"),
    ("kda", "wc_kda.json.rank0", "Kimi-Linear-48B-A3B", "KDA+MLA 混合",
     "递归状态遗忘门 a_t", "RecurrentState", "qkvbfg", 20,
     "同一次运行里 mla-latent 另覆盖 7 个 MLA 层"),
]


def summarize_selector(snap, suffix):
    """选择类探针没有带范数残差,报边界余量与 top-1 认证率。"""
    m1, c1, rows = [], [], 0
    for k, s in snap["slots"].items():
        if suffix and not k.endswith(suffix):
            continue
        mr = s.get("margin_r_mean") or {}
        if "1" in mr:
            m1.append(mr["1"])
        n = s.get("rows", 0); rows += n
        cr = s.get("cert_r") or {}
        if n and "0.001/r1" in cr:
            c1.append(cr["0.001/r1"] / n)
    if not m1:
        return None
    return {"margin_r1_median": S.median(m1), "cert_rate_r1_median": S.median(c1) if c1 else None,
            "rows": rows}


def summarize_state(snap, suffix):
    """递归状态类:报收缩因子中位、近 1 尾部与半衰期(均值口径)。"""
    import math
    am, pg, mx = [], [], 0.0
    for k, s in snap["slots"].items():
        if suffix and not k.endswith(suffix):
            continue
        if "a_t" in s:
            am.append(s["a_t"]["mean"]); mx = max(mx, s["a_t"]["max"])
            if s.get("a_hist"):
                pg.append(s["a_hist"][-1])
    if not am:
        return None
    hl = [math.log(2) / (-math.log(v)) for v in am if 0 < v < 1]
    return {"a_mean_median": S.median(am), "a_mean_range": [min(am), max(am)],
            "a_max": mx, "p_ge_0999_median": S.median(pg) if pg else None,
            "half_life_median": S.median(hl) if hl else None}


def summarize(snap, suffix=None):
    rel, tig, xn = [], [], []
    for k, s in snap["slots"].items():
        if suffix and not k.endswith(suffix):
            continue
        if "rel_int8" in s:
            rel.append(s["rel_int8"]["mean"]); tig.append(s["tight_int8"]["mean"])
            xn.append(s["x_norm"]["mean"])
    if not rel:
        return None
    return {"rel_int8_median": S.median(rel), "rel_int8_range": [min(rel), max(rel)],
            "tight_int8_median": S.median(tig), "x_norm_median": S.median(xn)}


def main():
    matrix, skipped = [], []
    for label, pat, model, family, mem, cls, path, layers, note in ROWS:
        fs = sorted(glob.glob(os.path.join(OUT_DIR, pat)))
        if not fs:
            skipped.append(f"{label}: 无快照({pat})"); continue
        snap = json.load(open(fs[0]))
        ok, msgs = verify.check(snap, expect_per_path={path: layers})
        if not ok:
            skipped.append(f"{label}: **未过验收门**,不予收录 —— " + "; ".join(
                m for m in msgs if "**" in m)); continue
        suffix = "|" + path
        row = {"model": model, "family": family, "memory_object": mem, "probe_class": cls,
               "path": path,
               "layers_covered": len(verify._layers_by_path(snap).get(path, ())),
               "layers_expected": layers,
               "n_calls": snap["coverage"]["n_calls"],
               "sample_every": snap["coverage"].get("sample_every", 1),
               "verify": "通过", "note": note}
        st = (summarize_selector(snap, suffix) if cls == "SparseSelector"
              else summarize_state(snap, suffix) if cls == "RecurrentState"
              else summarize(snap, suffix))
        if st:
            row.update(st)
        matrix.append(row)

    rep = {
        "what": "witcert.probe 平台覆盖矩阵:同一套尺子数学插进多少种模型",
        "caliber": ("逐路径对账(多探针同跑时全路径并集的层数没有意义);采样档位见各行 "
                    "sample_every;跨架构的 margin 量纲不可直接比(页级 vs token 级)"),
        "gate": "只收录通过验收门(覆盖->量纲->soundness)的行;未验收的数字等于没有",
        "adapters": 6, "injection_points": 11,
        "probe_classes": ["KV", "LatentKV", "SparseSelector", "RecurrentState"],
        "families_covered": sorted({r["family"] for r in matrix}),
        # 字段名要说清楚数的是什么:7 行 != 7 个模型(同一模型可有多条探针行)
        "n_probe_rows_verified": len(matrix),
        "n_unique_models": len({r["model"] for r in matrix}),
        "matrix": matrix,
        "skipped": skipped,
        "notes": [
            "GQA 两个模型是**首次接入**,只走 apply->起服->verify 三步,未写任何模型专用代码",
            "DeepSeek-V2-Lite 是**第五个架构族的零新代码接入**:按'接新架构三步'走,"
            "第一步(实测写入口)就发现它走 MLA 池的 set_mla_kv_buffer,已被 mla-latent 覆盖 —— "
            "这比'三步能接入'更强,说明记忆对象的分类学本身泛化",
            "V4 / GLM / Kimi-Linear 三者都在**同一次运行**里同时采到两类探针"
            "(以前每类要单独一个脚本跑一遍)",
            "legacy 的 apply_*_patch.py 已全部退役,能力由本框架承担",
        ],
    }
    dst = os.path.join(OUT_DIR, "p61_platform_matrix.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print("过验收门 %d 行 / %d 个唯一模型 / %d 个架构族"
          % (rep["n_probe_rows_verified"], rep["n_unique_models"],
             len(rep["families_covered"])))
    print("%-30s %-8s %-15s %9s  %s" %
          ("模型", "架构族", "探针类", "层覆盖", "关键读数"))
    for r in matrix:
        if r["probe_class"] == "SparseSelector":
            key = "margin(r=1) %.3f,top-1 认证率 %.4f" % (
                r.get("margin_r1_median", 0), r.get("cert_rate_r1_median") or 0)
        elif r["probe_class"] == "RecurrentState":
            key = "ā 中位 %.4f,半衰期 %.1f 步,P(a>=.999) %.4f,a_max %.4f" % (
                r.get("a_mean_median", 0), r.get("half_life_median") or 0,
                r.get("p_ge_0999_median") or 0, r.get("a_max", 0))
        else:
            key = "int8 残差 %.4f%%,保守度 %.2f×" % (
                r.get("rel_int8_median", 0) * 100, r.get("tight_int8_median", 0))
        print("%-30s %-8s %-15s %5d/%-3d  %s" %
              (r["model"][:30], r["family"][:8], r["probe_class"],
               r["layers_covered"], r["layers_expected"], key))
    for s in skipped:
        print(" 跳过:", s)


if __name__ == "__main__":
    main()
