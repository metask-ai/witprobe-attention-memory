# -*- coding: utf-8 -*-
"""R8-E3:把 p58 的逐 rank 原始探针文件汇总成 p58_v4_topk.json(论文可引口径)。

输入 experiments/out/wc_v4topk_probe.json.rank*(由 integration/apply_v4_topk_probe_patch.py
在 sglang 服务内落盘),输出 experiments/out/p58_v4_topk.json。

口径提醒(写进产物,防止跨模型误比):
  · V4 的 top-k 选**压缩页**(topk_transform_512),GLM/DSA 选 **token**(index_topk=2048),
    两者 margin 量纲不可直接比较,只可比"随排名衰减"的定性形状。
  · C4 索引 logits 在 TP 间**复制**(实测 8 rank 逐位相同)⇒ 有效样本 = 单 rank 行数,
    **不得**按 rank 数相乘。
  · 只统计"选择非平凡"的行(有效页数 > k);有效页 <= k 时 top-k 即全选,margin 无定义。

python experiments/p58_v4_topk_report.py
"""
import glob
import json
import os
import statistics as S

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "wc_v4topk_probe.json.rank*")),
                key=lambda s: int(s.split("rank")[-1]))
    if not fs:
        raise SystemExit("找不到 wc_v4topk_probe.json.rank*")
    ds = [json.load(open(f)) for f in fs]

    # 跨 rank 一致性:复制则取单 rank,否则求和(并在产物里如实标注)
    sig = {(round(d["all_layers"]["margin_mean"], 12), d["all_layers"]["rows"]) for d in ds}
    replicated = len(sig) == 1
    d0 = ds[0]
    a = d0["all_layers"]
    rows = max(1, a["rows"])
    trivial = sum(v.get("rows_trivial", 0) for v in d0["layers"].values())

    per_layer = []
    for li in sorted(d0["layers"], key=int):
        L = d0["layers"][li]
        n = max(1, L["rows"])
        per_layer.append({
            "layer": int(li), "rows": L["rows"],
            "margin_r1": L["margin_r_mean"]["1"],
            "margin_r8": L["margin_r_mean"]["8"],
            "margin_r64": L["margin_r_mean"]["64"],
            "margin_k": L["margin_mean"],
            "cert_rate_r1_eta1e-3": L["cert_r"]["0.001/r1"] / n,
        })

    def med(key):
        v = [p[key] for p in per_layer]
        return {"median": S.median(v), "min": min(v), "max": max(v)}

    rep = {
        "model": "DeepSeek-V4-Flash-FP8 (sgl-project 官方 fp8 重打包)",
        "stack": "sglang 0.5.13.post1, tp8+ep8, ctx 65536, disable-cuda-graph + disable-piecewise-cuda-graph",
        "granularity": d0["granularity"],
        "topk": d0["topk"],
        "protocol": ("在线探针注入 C4IndexerBackendMixin.forward_c4_indexer 的 topk_transform_512 "
                     "单点(覆盖 torch/v2/默认三分支);每次抽 <=64 行;按 c4_seq_lens 屏蔽 padding "
                     "(deep_gemm 以 clean_logits=False 调用,有效长度外是未初始化显存);"
                     "扰动 eta in {1e-3,1e-2} 相对分数尺度;探针流量 8k/30k/60k"),
        "tp_ranks_seen": len(ds),
        "logits_replicated_across_tp": replicated,
        "effective_rows": rows if replicated else sum(d["all_layers"]["rows"] for d in ds),
        "rows_trivial_all_selected": trivial,
        "n_layers_with_c4_index": d0["n_layers"],
        "layers": sorted(map(int, d0["layers"])),
        "margin_r_mean": a["margin_r_mean"],
        "margin_k_mean": a["margin_mean"],
        "margin_min": a["margin_min"],
        "score_absmax": a["score_absmax"],
        "cert_rate_r": {k: v / rows for k, v in a["cert_r"].items()},
        "flip_rate_r": {k: v / rows for k, v in a["flip_r"].items()},
        "flip_in_cert_r": a["flip_in_cert_r"],
        "cert_rate_k": {k: v / rows for k, v in a["cert"].items()},
        "flip_rate_k": {k: v / rows for k, v in a["flip"].items()},
        "flip_in_cert_k": a["flip_in_cert"],
        "lost_mass_mean": a["lost_mass_mean"],
        "per_layer": per_layer,
        "per_layer_summary": {k: med(k) for k in
                              ("margin_r1", "margin_r8", "margin_r64", "margin_k",
                               "cert_rate_r1_eta1e-3")},
    }
    viol = sum(a["flip_in_cert_r"].values()) + sum(a["flip_in_cert"].values())
    rep["soundness_violations"] = viol
    rep["findings"] = {
        "1_margin_decays_with_rank":
            "边界余量随排名衰减:r=1 %.4g / r=8 %.4g / r=64 %.4g / r=k=%d %.4g"
            % (a["margin_r_mean"]["1"], a["margin_r_mean"]["8"], a["margin_r_mean"]["64"],
               d0["topk"], a["margin_mean"]),
        "2_top1_certifiable":
            "top-1 可认证:eta=1e-3 认证率 %.4f 实测翻转 %.4f;逐层区间 [%.4f, %.4f]"
            % (rep["cert_rate_r"]["0.001/r1"], rep["flip_rate_r"]["0.001/r1"],
               rep["per_layer_summary"]["cert_rate_r1_eta1e-3"]["min"],
               rep["per_layer_summary"]["cert_rate_r1_eta1e-3"]["max"]),
        "3_strict_set_uncertifiable":
            "严格 top-%d 页集合认证率 %.4f / 翻转率 %.4f —— 集合口径过强(负结果,与 GLM 同向)"
            % (d0["topk"], rep["cert_rate_k"]["0.001"], rep["flip_rate_k"]["0.001"]),
        "4_soundness":
            "认证内翻转合计 %d(判据 sound;非平凡性由上面非零翻转率佐证)" % viol,
        "5_selection_binds_only_long":
            "平凡行(有效页<=k,top-k 即全选)%d vs 非平凡 %d —— 8k/30k/60k 下稀疏选择只在长上下文真正生效"
            % (trivial, rows),
    }
    dst = os.path.join(OUT_DIR, "p58_v4_topk.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
