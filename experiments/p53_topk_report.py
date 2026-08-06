# -*- coding: utf-8 -*-
"""R8-E2:把 p53 的逐 rank 原始探针文件汇总成 p53_topk.json(论文可引口径)。

输入 experiments/out/wc_topk_probe.json.rank*(由 integration/apply_topk_probe_patch.py
在 sglang 服务内落盘),输出 experiments/out/p53_topk.json。

**本文件存在的原因**:2026-07-30 首版 p53_topk.json 是**单层口径**冒充模型级 ——
探针状态挂实例(GLM 每层一个 Indexer)且扁平无层键,各层 flush 同一文件互相覆盖,
落盘只剩最后一层,而无层键使塌陷在产物里不可见。修复后重跑,并把"逐层 + 合并"
两级口径一起写进产物,使同类塌陷今后一眼可见(n_layers / 层清单)。

口径提醒:
  · GLM/DSA 的 top-k 选 **token**(index_topk=2048);V4/C4 选**压缩页**(见 p58)。
    两者 margin 量纲不可直接比较,只可比"随排名衰减"的定性形状。
  · rate 的分母一律是合并后的 rows_total,不是层数或 rank 数。

python experiments/p53_topk_report.py
"""
import glob
import json
import os
import statistics as S

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# 首版(单层口径,已作废)——留档以便审计能看到更正幅度
SUPERSEDED = {
    "note": "2026-07-30 首版:探针状态挂实例导致落盘只剩最后一层,以下数字为单层(层号未知)口径,已作废",
    "rows_total": 4601,
    "margin_r_mean": {"1": 6.482286860336042, "8": 0.27441841404066475,
                      "64": 0.03487038179678234},
    "margin_k_mean": 0.004166017535250489,
    "cert_rate_r": {"0.001/r1": 0.9936970223864378, "0.001/r8": 0.2677678765485764,
                    "0.001/r64": 0.07215822647250597, "0.01/r1": 0.9771788741577918,
                    "0.01/r8": 0.0730276026950663, "0.01/r64": 0.000869376222560313},
    "flip_rate_r": {"0.001/r1": 0.002825472723321017, "0.001/r8": 0.41643121060638993,
                    "0.001/r64": 0.7615735709628342, "0.01/r1": 0.008476418169963052,
                    "0.01/r8": 0.7928711149750054, "0.01/r64": 0.9763094979352315},
    "cert_rate_k": {"0.001": 0.0, "0.01": 0.0},
    "flip_rate_k": {"0.001": 0.9939143664420778, "0.01": 1.0},
}


def merge(ds):
    """跨 rank 合并:dp-attention 下各 rank 处理不同请求分片,层集合相同 -> 逐层求和。"""
    layers = {}
    n_calls = 0
    for d in ds:
        n_calls += d.get("n_calls", 0)
        for li, s in d["layers"].items():
            t = layers.setdefault(li, {"rows": 0, "margin_sum": 0.0, "margin_min": 1e30,
                                       "score_absmax": 0.0})
            t["rows"] += s["rows"]
            t["margin_sum"] += s["margin_sum"]
            t["margin_min"] = min(t["margin_min"], s["margin_min"])
            t["score_absmax"] = max(t["score_absmax"], s["score_absmax"])
            for fld in ("cert", "flip", "flip_in_cert", "cert_r", "flip_r", "flip_in_cert_r"):
                dd = t.setdefault(fld, {})
                for k, v in s.get(fld, {}).items():
                    dd[k] = dd.get(k, 0) + v
            for fld in ("margin_r", "lost_mass"):
                dd = t.setdefault(fld, {})
                for k, v in s.get(fld, {}).items():
                    e = dd.setdefault(k, {"sum": 0.0, "n": 0, "max": 0.0})
                    e["sum"] += v["sum"]; e["n"] += v["n"]
                    e["max"] = max(e["max"], v.get("max", 0.0))
    return layers, n_calls


def main():
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "wc_topk_probe.json.rank*")),
                key=lambda s: int(s.split("rank")[-1]))
    if not fs:
        raise SystemExit("找不到 wc_topk_probe.json.rank*")
    ds = [json.load(open(f)) for f in fs]
    if any("layers" not in d for d in ds):
        raise SystemExit("检测到旧格式(无 layers 键)——请用修复后的探针重跑,勿用旧文件生成产物")

    layers, n_calls = merge(ds)
    topk = ds[0]["topk"]

    agg = {"rows": 0, "margin_sum": 0.0, "margin_min": 1e30, "score_absmax": 0.0}
    for s in layers.values():
        agg["rows"] += s["rows"]; agg["margin_sum"] += s["margin_sum"]
        agg["margin_min"] = min(agg["margin_min"], s["margin_min"])
        agg["score_absmax"] = max(agg["score_absmax"], s["score_absmax"])
        for fld in ("cert", "flip", "flip_in_cert", "cert_r", "flip_r", "flip_in_cert_r"):
            d = agg.setdefault(fld, {})
            for k, v in s.get(fld, {}).items():
                d[k] = d.get(k, 0) + v
        for fld in ("margin_r", "lost_mass"):
            d = agg.setdefault(fld, {})
            for k, v in s.get(fld, {}).items():
                e = d.setdefault(k, {"sum": 0.0, "n": 0, "max": 0.0})
                e["sum"] += v["sum"]; e["n"] += v["n"]
                e["max"] = max(e["max"], v.get("max", 0.0))
    rows = max(1, agg["rows"])

    per_layer = []
    for li in sorted(layers, key=int):
        s = layers[li]
        n = max(1, s["rows"])
        per_layer.append({
            "layer": int(li), "rows": s["rows"],
            "margin_r1": s["margin_r"]["1"]["sum"] / max(1, s["margin_r"]["1"]["n"]),
            "margin_r8": s["margin_r"]["8"]["sum"] / max(1, s["margin_r"]["8"]["n"]),
            "margin_r64": s["margin_r"]["64"]["sum"] / max(1, s["margin_r"]["64"]["n"]),
            "margin_k": s["margin_sum"] / n,
            "cert_rate_r1_eta1e-3": s["cert_r"]["0.001/r1"] / n,
        })

    def spread(key):
        v = [p[key] for p in per_layer]
        return {"median": S.median(v), "min": min(v), "max": max(v)}

    viol = sum(agg["flip_in_cert_r"].values()) + sum(agg["flip_in_cert"].values())
    rep = {
        "model": "GLM-5.2-W4AFP8 (DSA indexer, index_topk=%d)" % topk,
        "stack": ("sglang 0.5.13.post1, tp8+ep8+dp8(dp-attention), kv fp8_e4m3, ctx 131072, "
                  "disable-cuda-graph + disable-piecewise-cuda-graph"),
        "granularity": "token(dsa)",
        "topk": topk,
        "protocol": ("在线探针注入 Indexer 的 5 处 metadata.topk_transform 调用点;每次抽 <=64 行"
                     "索引分数;扰动 eta in {1e-3,1e-2} 相对分数尺度(模拟索引 K 低精度存储);"
                     "探针流量 8k/30k/60k。本次未并开 WITCERT_OBS,以隔离 topk 探针开销"),
        # dp_size=8 但只有收到探针请求的 dp rank 会产出文件;如实记观测到的数量
        "ranks_observed": len(ds),
        "dp_size": 8,
        "ranks_note": "dp-attention 下请求被路由到部分 dp rank,只有收到请求的 rank 产出探针文件",
        "n_calls": n_calls,
        "n_layers": len(layers),
        "layers": sorted(map(int, layers)),
        "rows_total": agg["rows"],
        "margin_r_mean": {k: v["sum"] / max(1, v["n"]) for k, v in agg["margin_r"].items()},
        "margin_k_mean": agg["margin_sum"] / rows,
        "margin_min": agg["margin_min"],
        "score_absmax": agg["score_absmax"],
        "cert_rate_r": {k: v / rows for k, v in agg["cert_r"].items()},
        "flip_rate_r": {k: v / rows for k, v in agg["flip_r"].items()},
        "flip_in_cert_r": agg["flip_in_cert_r"],
        "cert_rate_k": {k: v / rows for k, v in agg["cert"].items()},
        "flip_rate_k": {k: v / rows for k, v in agg["flip"].items()},
        "flip_in_cert_k": agg["flip_in_cert"],
        "lost_mass_mean": {k: v["sum"] / max(1, v["n"]) for k, v in agg["lost_mass"].items()},
        "lost_mass_max": max((v.get("max", 0.0) for v in agg["lost_mass"].values()), default=0.0),
        "soundness_violations": viol,
        "per_layer": per_layer,
        "per_layer_summary": {k: spread(k) for k in
                              ("margin_r1", "margin_r8", "margin_r64", "margin_k",
                               "cert_rate_r1_eta1e-3")},
        "superseded_single_layer_v1": SUPERSEDED,
    }
    m = rep["margin_r_mean"]
    rep["findings"] = {
        "1_margin_decays_with_rank":
            "边界余量随排名衰减:r=1 %.4g / r=8 %.4g / r=64 %.4g / r=k=%d %.4g"
            % (m["1"], m["8"], m["64"], topk, rep["margin_k_mean"]),
        "2_top1_certifiable":
            "top-1 可认证:eta=1e-3 认证率 %.4f 实测翻转 %.4f;逐层区间 [%.4f, %.4f]"
            % (rep["cert_rate_r"]["0.001/r1"], rep["flip_rate_r"]["0.001/r1"],
               rep["per_layer_summary"]["cert_rate_r1_eta1e-3"]["min"],
               rep["per_layer_summary"]["cert_rate_r1_eta1e-3"]["max"]),
        "3_strict_set_uncertifiable":
            "严格 top-%d 集合认证率 %.4f / 翻转率 %.4f —— 集合口径过强(负结果)"
            % (topk, rep["cert_rate_k"]["0.001"], rep["flip_rate_k"]["0.001"]),
        "4_soundness":
            "认证内翻转合计 %d(判据 sound;非平凡性由非零翻转率佐证)" % viol,
        "5_coverage":
            "覆盖 %d 层 × %d 个产出探针的 dp rank(dp_size=8,只有收到请求的 rank 有数据),"
            "合并 %d 行;首版为单层 4601 行" % (len(layers), len(ds), agg["rows"]),
    }
    dst = os.path.join(OUT_DIR, "p53_topk.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
