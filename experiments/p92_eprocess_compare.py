# -*- coding: utf-8 -*-
"""E3(p92):望远镜逐事件分账 vs e-process 的数值对比。

六审修正(两处):
  1. 初版数字由内联脚本产出、未落仓 —— 违反"数字必须可回溯"纪律,本文件补正;
  2. 公平口径应比**双侧**累计对象:e-process 双侧 = 两条单侧过程各付 δ/2,
     半径 σ√(2T·ln(2/δ))(初版单侧 555σ 偏乐观,双侧 595σ,倍率 390×)。

**对象差异必须直说**(六审 P0-3):望远镜给的是"任一逐事件半径被突破"的界
(条目级授权用);e-process 给的是**有符号累计和**的任意时刻界。二者不是同一
失败事件 —— e-process 是累计误差对象的可形式化路线(Lean: Ville.eprocess_ville),
**尚未接入当前逐条目授权账本**;接入本身是 F4 的工作,不由本对比宣告完成。

python3 experiments/p92_eprocess_compare.py
"""
import json
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
T = 33407                # p91 实测概率事件数
DELTA = 0.01
SIGMA = 1.0              # 归一,比较只看倍率


def main():
    tele = sum(SIGMA * math.sqrt(2 * math.log(2 * i * (i + 1) / DELTA))
               for i in range(1, T + 1))
    epro_two_sided = SIGMA * math.sqrt(2 * T * math.log(2 / DELTA))
    rep = {
        "what": "E3:望远镜逐事件分账 vs e-process(双侧累计对象,公平口径)",
        "caliber": [
            "**对象不同,必须分开说**:望远镜界'任一逐事件半径被突破'(条目授权用);"
            "e-process 界'有符号累计和的任意时刻偏离'。后者不覆盖前者的失败事件 ——"
            "e-process 是累计误差对象的可形式化路线;同对象接入见 p95(模型校验语义,互补不替代)",
            "双侧 e-process = 两条单侧过程各付 δ/2:σ√(2T·ln(2/δ))",
            f"T={T:,} 取自 p91 实测;σ 归一",
        ],
        "T": T, "delta_req": DELTA,
        "telescoping_cumulative_radius": tele,
        "eprocess_anytime_radius_two_sided": epro_two_sided,
        "ratio": tele / epro_two_sided,
        "lean": ["Ville.ville(有限 Ω,零测度论)", "Ville.eprocess_ville"],
        "findings": {"0_headline":
            f"双侧累计对象上:e-process {epro_two_sided:,.0f}σ vs 望远镜求和 "
            f"{tele:,.0f}σ = {tele/epro_two_sided:.0f}× —— 量级差真实(√T vs T√ln),"
            "但**对象不同**:这是累计误差的路线图数字,不是对现有逐条目账本的替代"},
    }
    json.dump(rep, open(os.path.join(OUT, "p92_eprocess_compare.json"), "w"),
              ensure_ascii=False, indent=1)
    print(rep["findings"]["0_headline"])


if __name__ == "__main__":
    main()
