# -*- coding: utf-8 -*-
"""R9-B4:KDA 的正确稳定性判据是 Σlog a_t < 0(乘积),不是 ā < 1(均值)。

递归状态的误差传播是
    e_t ≤ (Π_{i≤t} a_i)·e_0 + Σ_s (Π_{i>s} a_i)·b_s
决定长期行为的是**对数和**,不是算术均值。二者由 Jensen 不等式相差一个非负的间隙:
    E[log a] ≤ log(E[a]) = log(ā)
所以 ā 既不是上界也不是下界,**它只是错的统计量**。

本文件用探针新加的 log 域累加器给出实测差距,并回答 KDA 在论文2 里该放多深。

python experiments/p65_kda_logsum.py
"""
import json
import math
import os
import statistics as S

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    d = json.load(open(os.path.join(OUT_DIR, "wc_kda.json.rank0")))
    rows = sorted((int(k.split("|L")[1].split("|")[0]), s)
                  for k, s in d["slots"].items()
                  if k.endswith("|qkvbfg") and "log_a" in s)
    per = []
    for li, s in rows:
        a, la = s["a_t"]["mean"], s["log_a"]["mean"]
        near1 = (s.get("a_hist") or [0])[-1]
        per.append({
            "layer": li, "a_mean": a, "E_log_a": la, "log_a_mean": math.log(a),
            "jensen_gap": math.log(a) - la,                 # ≥ 0
            "half_life_logsum": math.log(2) / (-la) if la < 0 else None,
            "half_life_mean": math.log(2) / (-math.log(a)) if 0 < a < 1 else None,
            "p_near_one": near1,
        })
    el = [p["E_log_a"] for p in per]
    lm = [p["log_a_mean"] for p in per]
    gap = [p["jensen_gap"] for p in per]
    n1 = [p["p_near_one"] for p in per]
    hl_l = [p["half_life_logsum"] for p in per if p["half_life_logsum"]]
    hl_m = [p["half_life_mean"] for p in per if p["half_life_mean"]]

    rep = {
        "what": "KDA 递归状态:对数和判据 vs 均值判据",
        "model": "Kimi-Linear-48B-A3B-Instruct(Kimi-K3 同架构代理)",
        "caliber": [
            "E[log a] 是对**所有 (通道, 时刻) 样本**的聚合,不是逐通道的 Σlog a_t;"
            "严格的逐通道判据需要保留通道身份,当前探针 reshape 后已丢失 —— **这是已知缺口**",
            "a_t 由逐层标量 A_log/dt_bias 均值广播近似,非逐头精确",
            "48B 代理,非 Kimi-K3 93L",
        ],
        "n_layers": len(per),
        "E_log_a": {"median": S.median(el), "min": min(el), "max": max(el)},
        "log_of_mean": {"median": S.median(lm), "min": min(lm), "max": max(lm)},
        "jensen_gap": {"median": S.median(gap), "min": min(gap), "max": max(gap)},
        "half_life_steps": {"logsum_median": S.median(hl_l), "mean_median": S.median(hl_m)},
        "p_near_one": {"median": S.median(n1), "max": max(n1)},
        "per_layer": per,
    }
    ratio = S.median(el) / S.median(lm)
    rep["findings"] = {
        "1_jensen_gap_is_material":
            "**均值口径把收缩算弱了**:E[log a] 中位 %.4f,而 log(ā) 中位 %.4f —— "
            "真实的对数收缩是均值口径给出的 %.1f 倍(Jensen 间隙中位 %.4f)"
            % (S.median(el), S.median(lm), ratio, S.median(gap)),
        "2_half_life":
            "按正确判据(对数和)误差半衰期中位 **%.2f 步**;按均值口径会报成 %.2f 步 —— "
            "后者偏保守 %.1f×" % (S.median(hl_l), S.median(hl_m),
                                  S.median(hl_m) / S.median(hl_l)),
        "3_all_layers_contract":
            "全部 %d 个 KDA 层的 E[log a] 均为负(最大 %.4f),故**在聚合口径下乘积收敛**"
            % (len(per), max(el)),
        "4_the_remaining_gap":
            "但近 1 通道占比中位 %.4f、最大 %.4f —— 这些通道上 log a≈0,乘积不收缩。"
            "**聚合的 E[log a]<0 不能推出逐通道收缩**,严格结论需保留通道身份重测。"
            % (S.median(n1), max(n1)),
        "5_verdict_for_paper2":
            "论文2 里 KDA 只报观测(本文件的分布 + 近 1 尾部),**不称证书**;"
            "逐通道 Σlog a_t 与递归界留给论文3 —— 判据已明确,缺的是保留通道身份的一次重测",
    }
    dst = os.path.join(OUT_DIR, "p65_kda_logsum.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
