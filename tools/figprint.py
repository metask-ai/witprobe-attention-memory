# -*- coding: utf-8 -*-
"""图内容指纹:剔除 PDF 时间戳后哈希,写 figs/FIGS.sha256 并与上版比对。

2026-08-06 评审:两个图生成器只覆盖、从不比对 —— 读者看到 REPRODUCTION OK,
并不包含"重生成的图与论文里的一致"这个命题(每次重生成只差时间戳,
永远无法字节校验)。此后:
  · 生成器出图后调 stamp(),剔 /CreationDate、/ModDate 后 sha256,
    落 figs/FIGS.sha256(随包,给读者校验的凭据);
  · 与上版哈希不一致 = 内容漂移:monorepo 打显式警告并更新指纹;
    WITCERT_FIG_STRICT=1(make_release dist 终验注入)下漂移即 exit 1 ——
    出包时刻图与指纹必须一致,读者侧(matplotlib 版本可致字节差)只警告。
"""
import hashlib
import os
import re
import sys

_TS = re.compile(rb"/(?:Creation|Mod)Date \(D:[^)]*\)")


def digest(path):
    b = open(path, "rb").read()
    if path.endswith(".pdf"):
        b = _TS.sub(b"", b)
    return hashlib.sha256(b).hexdigest()


def stamp(fig_dir, names):
    """names:本次生成器产出的文件名列表(显式列出,不扫目录 —— 手绘
    tex 图不归生成器管)。返回漂移文件名列表。"""
    man = os.path.join(fig_dir, "FIGS.sha256")
    old = {}
    if os.path.exists(man):
        for ln in open(man, encoding="utf-8"):
            if "  " in ln:
                h, n = ln.rstrip("\n").split("  ", 1)
                old[n] = h
    cur, drift = {}, []
    for n in sorted(set(names)):
        p = os.path.join(fig_dir, n)
        if not os.path.exists(p):
            continue
        cur[n] = digest(p)
        if n in old and old[n] != cur[n]:
            drift.append(n)
    keep = dict(old)
    keep.update(cur)
    with open(man, "w", encoding="utf-8") as f:
        for n in sorted(keep):
            f.write(f"{keep[n]}  {n}\n")
    if drift:
        msg = "图指纹漂移(内容变了):" + ", ".join(drift)
        if os.environ.get("WITCERT_FIG_STRICT") == "1":
            print("FAIL: " + msg + " —— 出包时刻图与随包指纹必须一致")
            sys.exit(1)
        print("⚠ " + msg + "(已更新 FIGS.sha256;matplotlib 版本差异也会致 PDF 字节漂移)")
    else:
        print(f"图指纹:{len(cur)} 个文件与 FIGS.sha256 一致/已登记")
    return drift
