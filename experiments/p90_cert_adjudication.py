# -*- coding: utf-8 -*-
"""E1(p90):四证书离线裁决 —— 更换认证对象前的定量审判。

对象:掩 2 位尾数(e4m3→粗网格)的**随机舍入**(SR:以 θ 概率向上取整,条件均值恰零)。
θ/Δ 全部由捕获字节离线重建(x=LUT[code]×2^scale;lo/hi 为目标网格邻点)。

四证书(全部**抽签前**可算,单位:logit 误差半径 → TV 界):
  C1 现行形:u = scale·Σ_b ‖q_b‖·w_b,w_b = 带 ℓ2 of 最坏残差 max(θ,1−θ)Δ → e-form
  C2 中心化:osc ≤ max(D+u)−min(D−u) → TV ≤ tanh(osc/4)(两点极值 sharp);
     tightness 对同一对象:osc 上界 / 实现 osc(六审修正,不再除以未中心化 |Δs|)
  C3 带方差 sub-G(Hoeffding):σ² = Σ_j (scale·q_j·Δ_j)²/4,u = √(2σ²ln(2/δ)) → e-form
  C4 精确两点 MGF:ψ_j(λ)=log[(1−θ)e^{−λθa_j}+θ e^{λ(1−θ)a_j}],a_j=scale·q_j·Δ_j,
     Chernoff 对 λ 网格取 min → e-form
地面真值:按 θ 逐坐标抽 64 次 SR 实现,给 |Δs| 实测分位 → 各证书 tightness。

决策门(先立后看):最优家族非空洞 ≥30/43;τ=0.2 retention ≥1.5× C1;
在线元数据分路裁决 —— C1/C2/C3 半径只需存储字节+当次 q(**元数据 0%**),
C4 需逐坐标 θ(约 +39% 存储,离线精确≠在线可部署,评估时已指出)。

口径:q 与条目跨 forward 配对(与现行 qn×W 同口径);地面真值 TV 用 nope 段
分数分布(rope 字节未捕获,基线分布近似,tightness 主指标不受影响);δ=1e-6 统一。

python3 experiments/p90_cert_adjudication.py
"""
import glob
import json
import math
import os
import statistics

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LUT = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
SCALE = 512 ** -0.5           # head_dim^{-1/2}(q 流里也带,以捕获值为准)
DELTA = 1e-6                  # 统一对比 δ(**联合预算**:逐条目取 δ/E,再对条目取 max)
TAU = 0.2                     # retention 阈(TV)


def load_raw():
    raw = {}
    for f in sorted(glob.glob(os.path.join(OUT, "wc_e1raw.rank*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for k, lst in d.items():
            raw.setdefault(k, []).extend(lst)
    return raw


def decode_entries(rec):
    """字节 → (x[|E|,448], lo, hi, Δ, θ)。lo/hi 为目标网格(掩 2 位)邻点值。"""
    nope, scales = rec["nope"], rec["scales"]
    x8 = LUT[nope.long()]
    sc = torch.exp2(scales.float() - 127.0).repeat_interleave(64, dim=1)
    x = x8 * sc
    mag = nope & 0x7F
    gf = mag & 0xFC
    gc = torch.minimum(gf + 4, torch.full_like(gf, 0x7C))
    vf = LUT[((nope & 0x80) | gf).long()] * sc
    vc = LUT[((nope & 0x80) | gc).long()] * sc
    lo, hi = torch.minimum(vf, vc), torch.maximum(vf, vc)
    delta = (hi - lo).clamp_min(0)
    theta = torch.where(delta > 0, (x - lo) / delta.clamp_min(1e-30), torch.zeros_like(x))
    theta = theta.clamp(0, 1)
    # **顶部夹取区**(mag∈[0x7C,0x7F)):目标网格 lo=hi≠x,误差为确定性平移 lo−x
    # —— 不是随机量,漏掉它曾让 C4 半径被实测击穿(soundness 自检抓出)
    det_e = torch.where(delta > 0, torch.zeros_like(x), lo - x)
    return x, lo, hi, delta, theta, det_e


def certs_for(q, delta_j, theta, det_e, scale):
    """四证书:随机部分半径 u + 逐条目确定性平移 D(顶部夹取,精确已知)。"""
    a = scale * q[None, :] * delta_j                       # [E,448] 有符号
    D = (scale * q[None, :] * det_e).sum(-1)               # [E] 确定性平移,精确
    # C1 现行形:带结构最坏残差
    worst = torch.maximum(theta, 1 - theta) * delta_j
    qb = q.view(16, 28).norm(dim=-1)                       # [16]
    wb = worst.view(-1, 16, 28).norm(dim=-1)               # [E,16]
    u1 = scale * (qb[None, :] * wb).sum(-1)
    # C3 Hoeffding sub-G
    sig2 = (a.pow(2) / 4).sum(-1)
    u3 = (2 * sig2 * math.log(2 / DELTA)).sqrt()
    # C4 精确 MGF(λ 网格 Chernoff)。**双侧 = 两个单侧半径取 max,各付 δ/2** ——
    # 初版取了 min:半径低于实测,五分钟的"完美"是 unsound(裁决自检当场抓出)
    lam = [2.0 ** k for k in range(-6, 14)]
    sides = []
    for sgn in (1.0, -1.0):
        aj = sgn * a                                       # [E,448]
        best = torch.full_like(u1, float("inf"))
        interior = (theta > 0) & (theta < 1)
        for l in lam:
            t1 = -l * theta * aj
            t2 = l * (1 - theta) * aj
            m = torch.maximum(t1, t2)
            lse = m + ((1 - theta) * (t1 - m).exp()
                       + theta * (t2 - m).exp()).log()
            # θ∈{0,1} 的坐标误差恒 0,ψ=0;零权重分支若被 max 选中会产生 log(0)=−inf,
            # 再被 clamp 掩成 0 —— u4_rand=0 的假完美即由此而来(数值探查定位)
            psi = torch.where(interior, lse, torch.zeros_like(lse)).sum(-1)
            best = torch.minimum(best, (math.log(2 / DELTA) + psi) / l)
        assert torch.isfinite(best).all(), "ψ 出现非有限值 —— 修复失效"
        sides.append(best)
    u4 = torch.maximum(sides[0], sides[1])
    # e-form 族的总半径 = |D| + 随机半径;D 单独返回给中心化界用(D_t 在 osc 中抵消)
    return {"C1": u1 + D.abs(), "C3": u3 + D.abs(), "C4": u4 + D.abs(),
            "u4_rand": u4, "D": D}


def tv_eform(u):  # noqa: E306
    return min(1.0, 0.5 * (math.exp(2 * u) - 1))


def tv_tanh(osc):
    return math.tanh(osc / 4)


def main():
    raw = load_raw()
    layers = sorted({int(k.split("|L")[1]) for k in raw if k.startswith("q|")})
    assert len(layers) == 43, f"q 流应覆盖 43 层,实得 {len(layers)}"
    per_layer, mc_ratio = {}, {"C1": [], "C2": [], "C3": [], "C4": []}
    g = torch.Generator().manual_seed(7)
    for L in layers:
        qrecs = raw.get(f"q|L{L}") or []
        erecs = (raw.get(f"swa_bytes|L{L}") or []) + (raw.get(f"page_bytes|L{L}") or [])
        if not qrecs or not erecs:
            continue
        scale = float(qrecs[0]["scale"])
        q = qrecs[0]["q"][0].float()[:448]
        x, lo, hi, dj, th, det_e = decode_entries(erecs[0])
        # 六审 P1-①:逐条目同 δ 后取 max 而不再联合,是漏账 —— 半径按 δ/E 联合
        global DELTA
        _saved_delta = DELTA
        DELTA = _saved_delta / max(1, erecs[0]["nope"].shape[0])
        u = certs_for(q, dj, th, det_e, scale)
        DELTA = _saved_delta
        u_max = {k: float(u[k].max()) for k in ("C1", "C3", "C4")}
        # **中心化的真正优势**:确定性平移 D_t 进 osc 而非绝对值 ——
        # osc ≤ max(D+u_rand) − min(D−u_rand),公共分量抵消
        osc = float((u["D"] + u["u4_rand"]).max() - (u["D"] - u["u4_rand"]).min())
        tv = {"C1": tv_eform(u_max["C1"]),
              "C2": tv_tanh(osc),
              "C3": tv_eform(u_max["C3"]),
              "C4": tv_eform(u_max["C4"])}
        # retention @ TV=0.2:e-form u* 与 tanh 的 osc* 各家换算
        u_star_e = 0.5 * math.log(1 + 2 * TAU)
        u_star_t = 2 * math.atanh(TAU)                 # osc*=4atanh(τ) → 每条目 u ≤ osc*/2
        Dc = (u["D"] - u["D"].median()).abs()          # 中心化后逐条目的偏移贡献
        ret = {"C1": float((u["C1"] <= u_star_e).float().mean()),
               "C2": float(((u["u4_rand"] + Dc) <= u_star_t / 2).float().mean()),
               "C3": float((u["C3"] <= u_star_e).float().mean()),
               "C4": float((u["C4"] <= u_star_e).float().mean())}
        # 地面真值:64 次 SR 实现的 |Δs| 最大值(逐条目),tightness = 半径/实测
        draws = 64
        E = x.shape[0]
        up = (torch.rand(draws, E, 448, generator=g) < th[None]).float()
        e_real = torch.where(up.bool(), (hi - x)[None], (lo - x)[None])
        ds = (scale * q[None, None, :] * e_real).sum(-1).abs()   # [draws,E]
        ds_max = ds.max(0).values.clamp_min(1e-12)
        viol_counts = {}
        for k in ("C1", "C3", "C4"):
            viol = int((ds_max > u[k]).sum())
            assert viol == 0, \
                f"L{L} {k} 半径被实测超越 {viol} 次 —— unsound,裁决作废"
            viol_counts[k] = viol
            mc_ratio[k].append(float((u[k] / ds_max).median()))
        # 六审 P1-②:C2 的 tightness 必须对**同一对象** —— osc 上界 vs 实现 osc,
        # 逐 realization(此前拿中心化界除以未中心化 |Δs|,不是同一个数学量)
        ds_signed = (scale * q[None, None, :] * e_real).sum(-1)      # [draws,E]
        osc_real = (ds_signed.max(-1).values - ds_signed.min(-1).values)  # [draws]
        osc_bound = float((u["D"] + u["u4_rand"]).max() - (u["D"] - u["u4_rand"]).min())
        assert bool((osc_real <= osc_bound + 1e-9).all()), \
            f"L{L} osc 上界被实现超越 —— unsound"
        mc_ratio["C2"].append(float(osc_bound / osc_real.median().clamp_min(1e-12)))
        per_layer[L] = {"tv": tv, "retention": ret, "n_entries": E,
                        "soundness_violations": viol_counts,
                        "u_median": {k: float(u[k].median())
                                     for k in ("C1", "C3", "C4")}}

    fams = ("C1", "C2", "C3", "C4")
    nonvac = {f: sum(1 for v in per_layer.values() if v["tv"][f] < 1.0) for f in fams}
    ret_med = {f: statistics.median(v["retention"][f] for v in per_layer.values())
               for f in fams}
    tight = {f: statistics.median(mc_ratio[f]) for f in fams}
    best = max(fams, key=lambda f: (nonvac[f], ret_med[f]))
    gates = {
        "gate1_nonvacuous_ge_30": nonvac[best] >= 30,
        "gate2_retention_x1.5": ret_med[best] >= 1.5 * max(ret_med["C1"], 1e-9),
        "gate3_metadata": {"C1/C2/C3": "0%(半径只需存储字节+当次 q)",
                          "C4_exact": "需逐坐标 θ ≈ +39% 存储 —— 在线不可部署,"
                                      "只能写时用或降级"},
    }
    rep = {
        "what": "E1:四证书离线裁决(对象=掩 2 位 SR;θ/Δ 由捕获字节重建)",
        "caliber": [
            "q 与条目跨 forward 配对(与现行 qn×W 同口径)",
            "δ=1e-6 统一;TV 界:C2 用 tanh(osc/4)(两点极值重推,常数 sharp),"
            "余用 e-form ½(e^{2u}−1)",
            "地面真值 = 64 次 SR 实现的逐条目 |Δs| 最大;tightness = 半径/实测中位",
            "C2 的 osc 界用 C4 每条目半径(最紧组合);C4 在线部署受 θ 元数据约束",
        ],
        "n_layers_covered": len(per_layer),
        "nonvacuous_43": nonvac, "retention_at_tv0.2_median": ret_med,
        "tightness_median_ratio": tight, "best_family": best,
        "gates": gates, "per_layer": {str(k): v for k, v in per_layer.items()},
    }
    rep["findings"] = {
        "0_headline": (
            "裁决(先立门后看数):非空洞 %s;retention@0.2 中位 %s;"
            "tightness(半径/实测,越小越紧)%s;最优家族 %s。"
            "门1(≥30/43)=%s 门2(≥1.5×)=%s;C4 精确形在线需 θ(+39%% 存储),"
            "C2/C3 半径元数据为零 —— 部署路径按此分野"
            % ({k: f"{v}/43" for k, v in nonvac.items()},
               {k: round(v, 3) for k, v in ret_med.items()},
               {k: round(v, 1) for k, v in tight.items()}, best,
               gates["gate1_nonvacuous_ge_30"], gates["gate2_retention_x1.5"])),
    }
    dst = os.path.join(OUT, "p90_cert_adjudication.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print(" ", rep["findings"]["0_headline"])


if __name__ == "__main__":
    main()
