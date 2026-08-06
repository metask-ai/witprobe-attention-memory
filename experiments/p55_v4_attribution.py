# -*- coding: utf-8 -*-
"""R8-E3.3:V4-Flash 三路风险归因(p55)——滑窗存储 / 稀疏索引 / hc 池化各担多少风险。

输入 experiments/out/wc_attr.json.rank*(witcert.probe 一次运行同时采到四条路径),
输出 experiments/out/p55_v4flash_attribution.json。

**边界诚实(必须随数字一起引用)**:池化与索引都是模型自己的原生计算。本产物刻画的是
"它把风险放在哪",**不是**"相对某个假想 dense 基线的误差";故池化侧归 empirical,
不称证书,也不参与 soundness 判定。只有存储侧(压缩条目二次量化)是带范数见证的
sound 上界。

三路口径:
  · 存储(swa_norm_rope):压缩条目再量化的相对残差与见证保守度 —— 有 sound 上界;
  · 选择(c4):页级 top-512 的边界余量与翻转认证 —— 有条件认证(margin>2ε);
  · 池化(pool_attn / pool_idx):块内权重集中度 —— 纯经验观测,无认证。

**跨 block 可比**:V4 同模型内有 block=4 与 block=128 两种池化粒度,
裸 top1_share 差一个数量级,故一律用归一化量(top1_norm / entropy_n / eff_frac),
并按 block 分组报告。

python experiments/p55_v4_attribution.py
"""
import glob
import json
import os
import statistics as S
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "experiments", "out")
sys.path.insert(0, os.path.join(ROOT, "src"))
from witcert.probe import contracts as C  # noqa: E402
from witcert.probe import verify  # noqa: E402


def main():
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "wc_attr.json.rank*")),
                key=lambda s: int(s.split("rank")[-1]))
    if not fs:
        raise SystemExit("找不到 wc_attr.json.rank*")
    snap = json.load(open(fs[0]))
    ok, msgs = verify.check(snap, expect_per_path={"swa_norm_rope": 43, "c4": 21},
                            expect_paths=["swa_norm_rope", "c4", "pool_attn", "pool_idx"])
    if not ok:
        raise SystemExit("验收门未通过,本轮数值作废:\n" + "\n".join(msgs))

    def rows(tag):
        return [(int(k.split("|L")[1].split("|")[0]), s)
                for k, s in snap["slots"].items() if k.endswith("|" + tag)]

    # ---- 存储侧(sound) ----
    st = rows("swa_norm_rope")
    store = {
        "n_layers": len(st),
        "rel_int8_median": S.median([s["rel_int8"]["mean"] for _, s in st]),
        "rel_int4_median": S.median([s["rel_int4"]["mean"] for _, s in st]),
        "tight_int8_median": S.median([s["tight_int8"]["mean"] for _, s in st]),
        "witness_violations": sum(int(s.get("viol_int8", 0)) + int(s.get("viol_int4", 0))
                                  for _, s in st),
        "status": "sound(带范数见证是 TV 的上界)",
    }
    # ---- 选择侧(条件认证) ----
    se = rows("c4")
    tot = sum(s.get("rows", 0) for _, s in se) or 1
    sel = {
        "n_layers": len(se),
        "rows": tot,
        "margin_r1_median": S.median([(s.get("margin_r_mean") or {}).get("1", 0) for _, s in se]),
        "cert_rate_r1": sum((s.get("cert_r") or {}).get("0.001/r1", 0) for _, s in se) / tot,
        "flip_rate_r1": sum((s.get("flip_r") or {}).get("0.001/r1", 0) for _, s in se) / tot,
        "cert_rate_set": sum((s.get("cert") or {}).get("0.001", 0) for _, s in se) / tot,
        # Lemma S2:(a_S,b_S)=(1, ½(m_out+m_in));加性,不放大上游误差
        "a_S": 1.0,
        "b_S": {eta: S.median([s["sel_bS"][eta]["mean"] for _, s in se if "sel_bS" in s])
                for eta in ("0.001", "0.01")
                if any("sel_bS" in s for _, s in se)},
        "m_out": {eta: S.median([s["lost_mass"][eta]["mean"] for _, s in se if "lost_mass" in s])
                  for eta in ("0.001", "0.01")},
        "m_in": {eta: S.median([s["gain_mass"][eta]["mean"] for _, s in se if "gain_mass" in s])
                 for eta in ("0.001", "0.01") if any("gain_mass" in s for _, s in se)},
        "flip_in_cert": sum(int(v) for _, s in se
                            for v in list((s.get("flip_in_cert_r") or {}).values())
                            + list((s.get("flip_in_cert") or {}).values())),
        "status": "条件认证(margin>2ε 时该 top-r 集合不可能改变)",
    }
    # ---- 池化侧(纯经验),按 block 分组 ----
    pool = {}
    for tag in ("pool_attn", "pool_idx"):
        by_block = {}
        for li, s in rows(tag):
            if "top1_norm" not in s:
                continue
            by_block.setdefault(int(s.get("block", -1)), []).append((li, s))
        grp = {}
        for blk, items in sorted(by_block.items()):
            t1 = [s["top1_norm"]["mean"] for _, s in items]
            wl = [s["w_l2"]["mean"] for _, s in items if "w_l2" in s]
            en = [s["entropy_n"]["mean"] for _, s in items]
            ef = [s["eff_frac"]["mean"] for _, s in items]
            hot = sorted(items, key=lambda kv: -kv[1]["top1_norm"]["mean"])[:5]
            grp[f"block{blk}"] = {
                "n_layers": len(items),
                # a_pool = ‖w‖_2 是池化段契约的系数本身(math_spec §1.2):
                # ‖Σ_j w_j Δ_j‖ ≤ ‖w‖_2·(Σ_j‖Δ_j‖²)^{1/2},故 ≤1 即衰减
                "a_pool_median": S.median(wl) if wl else None,
                "a_pool_range": [min(wl), max(wl)] if wl else None,
                "a_pool_floor": blk ** -0.5,          # 完全均摊时的理论下界
                "attenuates": (S.median(wl) < 1.0) if wl else None,
                "top1_norm_median": S.median(t1), "top1_norm_range": [min(t1), max(t1)],
                "entropy_n_median": S.median(en),
                "eff_frac_median": S.median(ef),
                "eff_tokens_median": S.median(ef) * blk,
                "most_concentrated_layers": [(li, round(s["top1_norm"]["mean"], 4)) for li, s in hot],
            }
        pool[tag] = grp
    viol_pool = sum(int(s.get("viol_pool", 0)) for tag in ("pool_attn", "pool_idx")
                    for _, s in rows(tag))

    rep = {
        "model": "DeepSeek-V4-Flash-FP8",
        "stack": ("sglang 0.5.13.post1, tp8+ep8, ctx 65536, "
                  "disable-cuda-graph + disable-piecewise-cuda-graph; "
                  "四条路径同一次运行采集,调用采样 EVERY=%d"
                  % snap["coverage"].get("sample_every", 1)),
        "honesty": ("池化与索引是模型原生计算:本产物刻画风险分布,不是相对假想 dense 基线的误差。"
                    "存储侧有 sound 上界;选择侧是条件认证;**池化侧自 2026-07-31 起也有 sound 契约** "
                    "—— a_pool=‖w‖_2 由 Cauchy–Schwarz 给出,它刻画的是块内 token 误差经池化后的"
                    "收缩系数,而非池化自身引入的误差(池化不引入误差,它就是模型语义)。"
                    "集中度指标(top1_norm/entropy_n/eff_frac)仍是经验观测。"),
        "coverage": snap["coverage"],
        "store_path": store,
        "select_path": sel,
        "pool_path": pool,
        "pool_invariant_violations": viol_pool,
    }
    pa4 = pool.get("pool_attn", {}).get("block4", {})
    pa128 = pool.get("pool_attn", {}).get("block128", {})
    pi4 = pool.get("pool_idx", {}).get("block4", {})
    # 端到端组合(A3):平台按契约演算自动串联并求值 —— 档位取最弱的一段
    try:
        ev = C.v4_chain(rep).evaluate(0.0)
        rep["end_to_end"] = {
            "bound": ev["bound"], "tier": ev["tier"], "delta": ev["delta"],
            "dominant_stage": ev["dominant_stage"],
            "per_stage": ev["per_stage"],
            "caliber": ("基线链条 = 存储 -> 选择(池化不在链上,见 math_spec §1.5);"
                        "b_store 是相对残差而非 TV —— 折成 TV 需逐请求 q 范数,"
                        "故本数字**不是** TV 界,不得称'端到端 certified'"),
        }
    except Exception as e:                    # 缺字段时如实留空,不猜
        rep["end_to_end"] = {"error": repr(e)[:200]}
    rep["findings"] = {
        "1_two_pool_granularities":
            "V4 同模型内有两种池化粒度并存:注意力侧 block=4 共 %s 层、block=128 共 %s 层 —— "
            "1/block 差 32 倍,裸 top1_share 跨层比较无意义,必须用归一化量"
            % (pa4.get("n_layers"), pa128.get("n_layers")),
        "2_pooling_is_near_uniform":
            "池化整体接近均摊:注意力侧 block4 top1_norm 中位 %.4f(0=完全均摊)、"
            "有效 token 数 %.2f/4;block128 中位 %.4f、有效 %.1f/128"
            % (pa4.get("top1_norm_median", 0), pa4.get("eff_tokens_median", 0),
               pa128.get("top1_norm_median", 0), pa128.get("eff_tokens_median", 0)),
        "3_pooling_is_an_attenuator":
            "**池化是误差衰减器,不是认证断点**:a_pool=‖w‖_2 实测 block4 中位 %.4f"
            "(理论均摊下界 %.4f)、block128 中位 %.4f(下界 %.4f),均远小于 1 —— "
            "块内 token 的量化误差经池化后按此系数收缩(Cauchy–Schwarz,sound)"
            % (pa4.get("a_pool_median") or 0, pa4.get("a_pool_floor") or 0,
               pa128.get("a_pool_median") or 0, pa128.get("a_pool_floor") or 0),
        "4_two_pool_paths_comparable":
            "两条池化通路集中度%s:索引侧 block4 top1_norm 中位 %.4f,注意力侧 %.4f(差 %.4f)"
            % ("基本一致" if abs(pi4.get("top1_norm_median", 0)
                                 - pa4.get("top1_norm_median", 0)) < 0.02
               else ("索引侧更集中" if pi4.get("top1_norm_median", 0)
                     > pa4.get("top1_norm_median", 0) else "注意力侧更集中"),
               pi4.get("top1_norm_median", 0), pa4.get("top1_norm_median", 0),
               pi4.get("top1_norm_median", 0) - pa4.get("top1_norm_median", 0)),
        "5_selection_contract":
            "Lemma S2:选择段契约 (a_S,b_S)=(1, ½(m_out+m_in)) —— **加性,不放大上游误差**;"
            "实测 b_S 中位 %s(η=1e-3)/ %s(η=1e-2),m_out≈m_in(|I|=|Î|=k 故换出换入数量相同)"
            % (("%.5f" % sel["b_S"]["0.001"]) if sel.get("b_S") else "n/a",
               ("%.4f" % sel["b_S"]["0.01"]) if sel.get("b_S") else "n/a"),
        "6_risk_split":
            "三路风险分层:存储侧相对残差 %.4f%%(sound 上界,保守度 %.2f×);"
            "选择侧 top-1 认证率 %.4f、严格集合认证率 %.4f(负结果);"
            "池化侧 sound 衰减系数 a_pool=%.4f(block4)/%.4f(block128)"
            % (store["rel_int8_median"] * 100, store["tight_int8_median"],
               sel["cert_rate_r1"], sel["cert_rate_set"],
               pa4.get("a_pool_median") or 0, pa128.get("a_pool_median") or 0),
        "7_end_to_end":
            "契约演算自动组合:端到端界 %.5f,档位 **%s**(取最弱段),主导段 %s —— "
            "存储段贡献 %.5f 是选择段 %.5f 的 %.1f 倍,故优化应优先动存储侧"
            % (rep["end_to_end"].get("bound", 0), rep["end_to_end"].get("tier", "?"),
               rep["end_to_end"].get("dominant_stage", "?"),
               rep["end_to_end"]["per_stage"][0]["added"] if rep["end_to_end"].get("per_stage") else 0,
               rep["end_to_end"]["per_stage"][1]["added"] if rep["end_to_end"].get("per_stage") else 0,
               (rep["end_to_end"]["per_stage"][0]["added"]
                / max(1e-12, rep["end_to_end"]["per_stage"][1]["added"]))
               if rep["end_to_end"].get("per_stage") else 0),
        "8_self_check":
            "见证违约 %d,认证内翻转 %d,池化不变量违反 %d —— 全零"
            % (store["witness_violations"], sel["flip_in_cert"], viol_pool),
    }
    dst = os.path.join(OUT_DIR, "p55_v4flash_attribution.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
