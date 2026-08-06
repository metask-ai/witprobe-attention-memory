# -*- coding: utf-8 -*-
"""R9-B6:融合算子的数值等价性与单次调用成本。

**顺序是硬的:先证数值一致,再谈快不快。** 一个算错的快算子毫无价值,而"融合后结果变了"
恰恰最容易被性能数字盖过去 —— 所以本文件先跑等价性,不过就直接退出,不报速度。

等价判据:融合版与 torch 版在**同一批输入**上,finalize 后每个指标的 mean/min/max
相对误差 ≤ 1e-4(fp32 归约顺序不同,不要求 bit 级一致),且 viol 计数**必须完全相等**
(那是 soundness 自检,不容许"差不多")。

成本判据:单次调用的 kernel 数从几十降到 1,墙钟应显著下降。这里量的是**探针自身的
单调用耗时**,不是 serving 端到端 —— 端到端由 p75 在开图下测。
"""
import json
import os
import time

import torch

from witcert import kernels
from witcert.kernels import band_witness_fused as fused
from witcert.probe import meters

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def _flat(d):
    """把 finalize 后的嵌套结构摊平成 {路径: 值},便于逐项比对。"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}/{kk}"] = vv
        else:
            out[k] = v
    return out


def main():
    if not fused.available():
        raise SystemExit("无 triton/CUDA,跳过")
    torch.manual_seed(0)
    dev = "cuda"
    rep = {"what": "融合 band_witness 的数值等价性与单调用成本",
           "machine": "hgx",
           "caliber": ["先证数值一致再谈速度;等价不过则不报速度",
                       "相对误差阈 1e-4(fp32 归约顺序不同,不要求 bit 级一致)",
                       "viol 计数必须完全相等 —— 那是 soundness 自检,不容许近似",
                       "量的是探针单次调用耗时,不是 serving 端到端(端到端见 p75)"],
           "cases": []}

    ok_all = True
    for n, D, bands in ((128, 128, 8), (128, 512, 8), (64, 1024, 16), (7, 128, 8)):
        x = torch.randn(n, D, device=dev, dtype=torch.float32)
        ref = meters.finalize(meters.band_witness(x.clone(), bands=bands,
                                                  quantizers=(("int8", 8),)))
        got = meters.finalize(fused.fused_band_witness(x.clone(), bands=bands, bits=8))
        rf, gf = _flat(ref), _flat(got)
        worst, worst_k = 0.0, None
        for k, v in rf.items():
            if k.startswith("_") or k not in gf:
                continue
            a, b = float(v), float(gf[k])
            if k.endswith("/min") or k.endswith("/max") or k.endswith("/mean"):
                e = abs(a - b) / max(1e-12, abs(a))
                if e > worst:
                    worst, worst_k = e, k
        viol_ref = int(ref.get("viol_int8", 0)); viol_got = int(got.get("viol_int8", 0))
        n_ref = int(rf.get("x_norm/n", 0)); n_got = int(gf.get("x_norm/n", 0))
        case_ok = worst <= 1e-4 and viol_ref == viol_got and n_ref == n_got
        ok_all &= case_ok
        rep["cases"].append({"n": n, "D": D, "bands": bands, "worst_rel_err": worst,
                             "worst_key": worst_k, "viol_ref": viol_ref,
                             "viol_fused": viol_got, "n_ref": n_ref, "n_fused": n_got,
                             "ok": case_ok})
        print(f"  n={n} D={D} bands={bands}: 最大相对误差 {worst:.2e}({worst_k}), "
              f"viol {viol_ref}/{viol_got}, n {n_ref}/{n_got} -> {'一致' if case_ok else '**不一致**'}")

    # **多次累加也必须一致**:计数与 min/max 的累加语义在融合版里是读改写,容易写错
    x = torch.randn(128, 128, device=dev)
    r = {}; g = {}
    for _ in range(5):
        meters.band_witness(x, bands=8, quantizers=(("int8", 8),), out=r)
        fused.fused_band_witness(x, bands=8, bits=8, out=g)
    rf, gf = _flat(meters.finalize(r)), _flat(meters.finalize(g))
    acc_ok = (int(rf["x_norm/n"]) == int(gf["x_norm/n"]) == 5 * 128
              and abs(rf["x_norm/mean"] - gf["x_norm/mean"]) / abs(rf["x_norm/mean"]) < 1e-4)
    rep["repeat_accumulate"] = {"n_ref": int(rf["x_norm/n"]), "n_fused": int(gf["x_norm/n"]),
                                "ok": acc_ok}
    print(f"  重复累加 5 次:n {int(rf['x_norm/n'])}/{int(gf['x_norm/n'])} -> "
          f"{'一致' if acc_ok else '**不一致**'}")
    ok_all &= acc_ok
    rep["equivalent"] = bool(ok_all)
    if not ok_all:
        rep["findings"] = {"0_headline": "**数值不等价,不报速度** —— 算错的快算子没有价值"}
        json.dump(rep, open(os.path.join(OUT_DIR, "p74_fused_equiv.json"), "w"),
                  ensure_ascii=False, indent=1)
        raise SystemExit("等价性未通过")

    # 速度:同一输入,各跑 200 次取墙钟
    def bench(fn, reps=200):
        o = {}
        fn(o)                                   # 预热 + 分配
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            fn(o)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e6      # 微秒

    x = torch.randn(128, 128, device=dev)
    t_ref1 = bench(lambda o: meters.band_witness(x, bands=8, quantizers=(("int8", 8),), out=o))
    t_ref2 = bench(lambda o: meters.band_witness(x, bands=8, out=o))
    t_fus = bench(lambda o: fused.fused_band_witness(x, bands=8, bits=8, out=o))
    rep["latency_us"] = {"torch_1quant": t_ref1, "torch_2quant": t_ref2, "fused_1quant": t_fus}
    rep["speedup"] = {"vs_torch_1quant": t_ref1 / t_fus, "vs_torch_2quant": t_ref2 / t_fus}
    rep["findings"] = {
        "0_headline": (
            "**数值等价成立**(最大相对误差 %.1e,viol 与 n 完全相等),"
            "融合后单次调用 %.1f us,对 torch 单量化档 %.2f×、双量化档 %.2f× —— "
            "这正是 B6 要压的常数:开图后成本的驱动量是每次执行的 kernel 数"
            % (max(c["worst_rel_err"] for c in rep["cases"]), t_fus,
               rep["speedup"]["vs_torch_1quant"], rep["speedup"]["vs_torch_2quant"])),
        "1_caveat": ("这是**探针单调用**耗时,不是 serving 端到端。端到端要在开图下测,"
                     "且必须同时过重放累加与层覆盖对账(见 p75)"),
    }
    dst = os.path.join(OUT_DIR, "p74_fused_equiv.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
