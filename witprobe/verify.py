# -*- coding: utf-8 -*-
"""验收门:把"探针类实验怎么算跑对了"从操作者的纪律变成工具的硬检查。

顺序是固定的(TinyKG 10481),前一步不过后面的数字全部作废:

  1. **覆盖** —— 触发了吗?层齐吗?声明的写入路径都走到了吗?
     零触发和只测到一层都不会自己报错,必须主动对账。
  2. **量纲** —— 数值落在物理合理区间吗?出现 inf/nan/1e30 量级说明读到了
     未初始化显存或分带算错(当日两次)。
  3. **soundness** —— 见证违约 / 认证内翻转必须为 0。这不是科学结论,是实现自检。

    python -m witcert.probe.verify out.json.rank0 --expect-layers 43
    python -m witcert.probe.verify 'out.json.rank*' --expect-layers-from config.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys

MAG_LIMIT = 1e6          # 索引分数/残差范数的物理上限,超过即疑似未初始化显存
SANE_KEYS = ("x_norm", "score_abs", "margin", "true_int8", "wit_int8", "a_t")
RANGE_KEYS = {"top1_norm": (0.0, 1.0), "entropy_n": (0.0, 1.0),
              "eff_frac": (0.0, 1.0)}   # 定义域有界的指标,越界即算错


# 计数类字段不参与量纲检查:n / 各种计数器本来就可以很大,
# 把它们当"数值离谱"是误报(2026-07-30 实测踩到)。
COUNT_SUFFIXES = ("/n", "_calls", "rows", "rows_trivial", "rowlen_mismatch",
                  "n_seen", "n_skipped_by_sampling", "sample_every")


def _is_count(path: str) -> bool:
    return any(path.endswith(s) or s in path.rsplit("/", 1)[-1] for s in COUNT_SUFFIXES)


def _walk_numbers(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk_numbers(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk_numbers(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


def _layers_by_path(snap: dict) -> dict:
    """逐路径的层集合。多探针同跑时(如 Kimi-Linear 同时被 kda-state 与 mla-latent 采),
    全路径并集的层数没有意义,必须按路径分开对账。"""
    out = {}
    for k in snap.get("slots", {}):
        parts = k.split("|")
        if len(parts) == 3:
            out.setdefault(parts[2], set()).add(parts[1][1:])
    return out


def check(snap: dict, expect_layers=None, expect_paths=None,
          expect_per_path=None, expect_replay=False) -> tuple[bool, list[str]]:
    cov = snap.get("coverage", {})
    msgs, ok = [], True

    # 1. 覆盖
    if snap.get("err"):
        ok = False; msgs.append(f"[覆盖] 探针内部异常: {snap['err']}")
    n = cov.get("n_calls", 0)
    if n == 0:
        return False, ["[覆盖] **零触发** —— 探针一次都没跑到。写入口八成不对:"
                       "grep 模型文件里真正调用的方法名,别按类结构推断。"]
    msgs.append(f"[覆盖] n_calls={n} slots={cov.get('n_slots')} "
                f"owners={cov.get('owners')} paths={cov.get('paths')}")
    got = cov.get("layers", [])
    msgs.append(f"[覆盖] 层数 {cov.get('n_layers')}:{got if len(got) <= 32 else str(got[:32]) + '…'}")
    if expect_layers is not None:
        if isinstance(expect_layers, int):
            if cov.get("n_layers", 0) != expect_layers:
                ok = False
                msgs.append(f"[覆盖] **层数不符**:得到 {cov.get('n_layers')},期望 {expect_layers}"
                            " —— 典型成因是状态挂了实例导致各层互相覆盖")
        else:
            miss = sorted(set(expect_layers) - set(got))
            if miss:
                ok = False; msgs.append(f"[覆盖] **缺层** {miss[:20]}")
    if expect_paths:
        miss = sorted(set(expect_paths) - set(cov.get("paths", [])))
        if miss:
            ok = False; msgs.append(f"[覆盖] **未走到的写入路径** {miss}")
    lbp = _layers_by_path(snap)
    if lbp:
        msgs.append("[覆盖] 逐路径层数:" + ", ".join(
            f"{p}={len(v)}" for p, v in sorted(lbp.items())))
    if expect_per_path:
        for p, want in expect_per_path.items():
            got = len(lbp.get(p, ()))
            if got != want:
                ok = False
                msgs.append(f"[覆盖] **路径 {p} 层数不符**:得到 {got},期望 {want}")

    # 1b. **图内活性** —— 层覆盖齐全不代表探针还活着。
    #
    # 开图后 should_sample() 这个 python 分支**只在 capture 时求值一次**,决定的是
    # "这张图里到底有没有探针算子"。若捕获当刻恰好没采,该图此后每一次重放都不带探针,
    # 而起服期留下的陈旧样本**照样让层覆盖显示 28/28** —— 门就这么放行了
    # (2026-07-31 实测:EVERY=64 带图跑完整轮负载,Δ累加 = 0,验收却全过)。
    #
    # 判据:设备侧累加计数 vs python 侧调用计数。CUDA graph 重放不执行 Python,
    # 故 **device_n > python_n 才说明有重放贡献**;两者相等 = 每次累加都来自 python 执行
    # = 图里没有探针。这是"覆盖塌陷没有免费警报"的又一个实例,必须由门来喊。
    if cov.get("graph_safe"):
        dev = pyc = 0
        have_dev = False
        for s in snap.get("slots", {}).values():
            if not isinstance(s, dict):
                continue
            pyc += int(s.get("_n_calls") or 0)
            for v in s.values():
                # **必须用 calls(每次执行 +1)而不是 n(元素数)**:n/_n_calls 只是
                # 每次调用的行数,和有没有重放毫无关系 —— 拿它当判据会被骗(踩过)
                if isinstance(v, dict) and "calls" in v:
                    dev += int(v["calls"] or 0); have_dev = True; break
        if not have_dev:
            msgs.append("[覆盖] 图内活性:产物无设备侧调用计数(旧版探针),无法判定")
        else:
            msgs.append(f"[覆盖] 图内活性:设备执行 {dev} 次 / python 执行 {pyc} 次")
        if expect_replay:
            if not have_dev:
                ok = False
                msgs.append("[覆盖] **无法判定图内活性** —— 产物缺设备侧调用计数,"
                            "带图运行必须能判定,故不放行")
            elif dev <= pyc:
                ok = False
                msgs.append(
                    "[覆盖] **图重放期无累加** —— 设备计数未超过 python 计数,说明探针算子"
                    "没进任何一张图。此时的低开销只是'探针不在图里'的同义反复,"
                    "层覆盖数字来自起服期的陈旧样本,不可用")
            else:
                msgs.append(f"[覆盖] 重放贡献 {dev - pyc} 次(只能来自图内 in-place 累加)")

    # 2. 量纲
    bad = []
    for p, v in _walk_numbers(snap.get("slots", {})):
        if _is_count(p):
            continue
        if math.isnan(v) or math.isinf(v):
            bad.append((p, v))
        elif any(k in p for k in SANE_KEYS) and abs(v) > MAG_LIMIT:
            bad.append((p, v))
        else:
            for rk, (lo, hi) in RANGE_KEYS.items():
                if f"/{rk}/" in p and not (lo - 1e-6 <= v <= hi + 1e-6):
                    bad.append((p, v))
    if bad:
        ok = False
        msgs.append(f"[量纲] **{len(bad)} 处数值离谱**(inf/nan 或 >{MAG_LIMIT:g}),"
                    f"疑似读到未初始化显存或分带错误;样例 {bad[:4]}")
    else:
        msgs.append("[量纲] 未见 inf/nan 或超限值")

    # 3. soundness
    viol = 0
    for p, v in _walk_numbers(snap.get("slots", {})):
        if ("viol_" in p or "flip_in_cert" in p or "block_mixed" in p) and v:
            viol += int(v)
    if viol:
        ok = False; msgs.append(f"[soundness] **违约 {viol} 例** —— 见证/判据实现有错")
    else:
        msgs.append("[soundness] 见证违约与认证内翻转均为 0")

    # 提示项(不判失败)
    triv = sum(v for p, v in _walk_numbers(snap.get("slots", {})) if p.endswith("rows_trivial"))
    if triv:
        msgs.append(f"[提示] 平凡行(选择即全选)共 {int(triv)},引用覆盖率时要连这个一起说")
    mism = sum(v for p, v in _walk_numbers(snap.get("slots", {})) if p.endswith("rowlen_mismatch"))
    if mism:
        msgs.append(f"[提示] 长度向量与 logits 行数不整除 {int(mism)} 次,已按补齐处理")
    return ok, msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="witcert.probe.verify")
    ap.add_argument("files", nargs="+", help="快照 JSON(可用通配)")
    ap.add_argument("--expect-layers", default=None,
                    help="期望层数(整数)或逗号分隔的层号清单")
    ap.add_argument("--expect-paths", default=None, help="逗号分隔的写入路径标签(必须走到)")
    ap.add_argument("--expect", default=None,
                    help="逐路径期望层数,如 qkvbfg=20,mla=7 —— 多探针同跑时用这个而非 --expect-layers")
    ap.add_argument("--expect-replay", action="store_true",
                    help="**带图运行必加**:要求设备侧累加计数超过 python 调用计数,"
                         "即探针算子确实进了图并在重放期累加。层覆盖齐全不代表探针还活着 —— "
                         "起服期的陈旧样本足以让覆盖显示满层(2026-07-31 实测放行过一次)")
    a = ap.parse_args(argv)
    fs = sorted({f for pat in a.files for f in glob.glob(pat)})
    if not fs:
        print("找不到快照文件", file=sys.stderr); return 2
    el = a.expect_layers
    if el is not None:
        el = int(el) if el.isdigit() else [int(x) for x in el.split(",")]
    ep = a.expect_paths.split(",") if a.expect_paths else None
    epp = None
    if a.expect:
        epp = {}
        for kv in a.expect.split(","):
            k2, v2 = kv.split("=")
            epp[k2.strip()] = int(v2)
    allok = True
    for f in fs:
        ok, msgs = check(json.load(open(f)), el, ep, epp, a.expect_replay)
        allok &= ok
        print(f"=== {f}  {'通过' if ok else '**不通过**'}")
        for m in msgs:
            print("   ", m)
    print("\n验收:", "全部通过" if allok else "**存在不通过项 —— 本轮数值全部作废**")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
