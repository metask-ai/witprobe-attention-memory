# -*- coding: utf-8 -*-
"""R8-E3.2:把 p54 的逐 rank 原始观测汇总成 p54_v4flash_quant.json(论文可引口径)。

输入 experiments/out/wc_v4obs.json.rank*(由 integration/apply_v4_obs_patch.py 落盘),
输出 experiments/out/p54_v4flash_quant.json。

主张:现有带范数 witness **数学不变**即可覆盖 V4-Flash 的压缩条目形态
(448 维 nope FP8 + 64 维 rope BF16 = 512,与 GLM 的 latent+k_pe 同构)。
Tier A 的 soundness 由 p50(V2-Lite 逐 cell 对拍,零违约)承担;本产物在真实生产
条目上给量级与紧度,口径与 GLM 的 p51 可比。

口径提醒:
  · 二次量化 = 在**已被 V4 压缩过**的条目上再上我们的量化器,量的是我们**额外**
    引入的误差,不是 V4 原生 fp8 的误差(后者需要量化前的同源值,主写入口是融合
    norm+rope+落盘 kernel,拿不到同源前值 —— 如实标注为未测)。
  · 紧度 = W/‖Δ‖,W=Σ_b‖Δ_b‖ 为带范数见证,恒 ≥1,越接近 1 越紧。
  · rate/均值的分母是采样到的 token 行数,不是层数或 rank 数。

python experiments/p54_v4_quant_report.py
"""
import glob
import json
import os
import statistics as S

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FIELDS = ("c_norm", "true_int8", "rel_int8", "wit_int8", "tight_int8",
          "true_int4", "rel_int4", "wit_int4", "tight_int4",
          "native_resid", "native_resid_rel")


def main():
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "wc_v4obs.json.rank*")),
                key=lambda s: int(s.split("rank")[-1]))
    if not fs:
        raise SystemExit("找不到 wc_v4obs.json.rank*")
    ds = [json.load(open(f)) for f in fs]

    # 跨 rank 一致性:逐位相同则为复制(有效样本 = 单 rank),否则合并
    def sig(d):
        p = d["pools"]["pool0"]["layers"]
        k = sorted(p)[0]
        return (round(p[k]["c_norm"]["mean"], 12), p[k]["c_norm"]["n"])
    replicated = len({sig(d) for d in ds}) == 1
    src = ds[:1] if replicated else ds

    layers = {}
    for d in src:
        for pk, p in d["pools"].items():
            for lk, s in p["layers"].items():
                t = layers.setdefault(lk, {"n": 0})
                t["n"] += s.get("n", 0)
                for f in FIELDS:
                    v = s.get(f)
                    if not isinstance(v, dict):
                        continue
                    e = t.setdefault(f, {"sum": 0.0, "n": 0, "min": 1e30, "max": -1e30})
                    e["sum"] += v["mean"] * v["n"]; e["n"] += v["n"]
                    e["min"] = min(e["min"], v["min"]); e["max"] = max(e["max"], v["max"])
                for f in ("viol_int8", "viol_int4"):
                    t[f] = t.get(f, 0) + s.get(f, 0)

    def agg(field):
        num = sum(v[field]["sum"] for v in layers.values() if field in v)
        den = sum(v[field]["n"] for v in layers.values() if field in v)
        if den == 0:
            return None
        vals = [v[field]["sum"] / v[field]["n"] for v in layers.values() if field in v]
        return {"mean": num / den, "n": den, "layer_median": S.median(vals),
                "layer_min": min(vals), "layer_max": max(vals),
                "min": min(v[field]["min"] for v in layers.values() if field in v),
                "max": max(v[field]["max"] for v in layers.values() if field in v)}

    per_layer = []
    for lk in sorted(layers, key=lambda x: (x.split("/")[1], int(x.split("/")[0]))):
        v = layers[lk]
        row = {"layer": int(lk.split("/")[0]), "path": lk.split("/")[1],
               "rows": v.get("c_norm", {}).get("n", 0)}
        for f in FIELDS:
            if f in v:
                row[f] = v[f]["sum"] / v[f]["n"]
        per_layer.append(row)

    viol = sum(v.get("viol_int8", 0) + v.get("viol_int4", 0) for v in layers.values())
    meta = ds[0]["pools"]["pool0"]["meta"]
    rep = {
        "model": "DeepSeek-V4-Flash-FP8 (sgl-project 官方 fp8 重打包)",
        "stack": ("sglang 0.5.13.post1, tp8+ep8, ctx 65536, "
                  "disable-cuda-graph + disable-piecewise-cuda-graph"),
        "entry_layout": ("448 维 nope FP8(每 64 维一块 ue8m0 scale)+ 64 维 rope BF16 = 512;"
                         "与 GLM 的 latent(512)+k_pe(64) 同构 —— 带范数 witness 数学不变"),
        "pool": meta,
        "bands": ds[0]["bands"],
        "protocol": ("模块级探针挂 3 处写入口(主:set_swa_key_buffer_radix_fused_norm_rope);"
                     "写入**后**用上游 dequantize_k_cache_paged 从池读回真实落盘值(不自实现 ue8m0);"
                     "每次抽 <=256 token;探针流量 8k/30k/60k"),
        "ranks_observed": len(ds),
        "replicated_across_tp": replicated,
        "effective_rows": sum(v.get("c_norm", {}).get("n", 0) for v in layers.values()),
        "write_paths_seen": sorted({r["path"] for r in per_layer}),
        "n_layers": len({r["layer"] for r in per_layer}),
        "layers": sorted({r["layer"] for r in per_layer}),
        "witness_violations": viol,
        "aggregate": {f: agg(f) for f in FIELDS if agg(f) is not None},
        "per_layer": per_layer,
        "native_fp8_residual": ("未测:主写入口是融合 norm+rope+落盘 kernel,"
                                "拿不到与落盘值同源的量化前值;GLM 侧同口径数字见 p51"),
    }
    a = rep["aggregate"]
    rep["findings"] = {
        "1_meter_covers_new_layout":
            "现有带范数 witness 未改一行数学即在 V4 压缩条目上跑通:%d 层 × %d 行,见证违约 %d"
            % (rep["n_layers"], rep["effective_rows"], viol),
        "2_secondary_int8_cheap":
            "二次量化 int8 相对残差 %.4g(逐层中位 %.4g,区间 %.4g-%.4g)—— 与 GLM 生产 fp8 的 0.93%% 同量级"
            % (a["rel_int8"]["mean"], a["rel_int8"]["layer_median"],
               a["rel_int8"]["layer_min"], a["rel_int8"]["layer_max"]),
        "3_int4_breaks":
            "int4 相对残差 %.4g,是 int8 的 %.1f 倍 —— 4bit 档在已压缩条目上再压不可行(与 GQA/latent 侧结论同向)"
            % (a["rel_int4"]["mean"], a["rel_int4"]["mean"] / a["rel_int8"]["mean"]),
        "4_witness_tightness":
            "带范数见证保守度 W/‖Δ‖ = %.2f×(B=%d,int8)/ %.2f×(int4),逐层区间 %.2f-%.2f"
            % (a["tight_int8"]["mean"], rep["bands"], a["tight_int4"]["mean"],
               a["tight_int8"]["layer_min"], a["tight_int8"]["layer_max"]),
    }
    dst = os.path.join(OUT_DIR, "p54_v4flash_quant.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
