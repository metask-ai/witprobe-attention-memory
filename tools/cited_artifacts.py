# -*- coding: utf-8 -*-
"""引用产物清单生成器:从四个生成器/门禁**机械推导**论文引用的产物文件集。

发布纪律的关键一环:公开仓只带被引用的 run(全量 q*/p* 通配会把未发表的
探索轨迹一并公开 —— 那是研发过程,不随论文开源)。清单必须机械推导,
手维护必漏。

    python3 tools/cited_artifacts.py p2-attention-memory
写出 papers/<slug>/artifacts.list;make_release 的 include 用 "@<file>" 引用。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "out")

#: 各论文的产物消费者(生成器与门禁 —— 它们读什么,论文就引用什么)
CONSUMERS = {
    "p2-attention-memory": [
        "tools/make_canon.py", "tools/p2_figs.py",
        "experiments/q6_sixpath_build.py", "experiments/flagship_gate.py",
        "tools/adjudication_export.py",
    ],
    "p1-kv-certificates": [
        "tools/make_canon.py", "tools/lean_extract.py",
        "tools/make_figs.py",
    ],
    # 2026-08-19:p3/p4 此前只走 canon 推导 —— 那覆盖"数字溯源"所需的 run,
    # 但**出图器读得更多**。以外人身份从干净 clone 跑 tools/p3_figs.py 当场
    # FileNotFoundError(w3cf_s0_exact_client.json)。图能从原始产物重生成,
    # 是比"数字对得上账"强得多的复现主张,所以出图器必须纳入。
    # **但不能整个脚本一起追踪**:p3_figs.py 同时出 p3 与 p4 的图,整体追踪
    # 得到两篇合并的 248 条 —— 会把**尚未发表的 p4 证据塞进 p3 的公开仓**。
    # 故出图器走 collect_figs(按图的 OWNER 归属逐图追踪),不进 CONSUMERS。
    # **不含 make_canon**:它重建全项目 canon,会读遍四篇的产物 —— 追踪它
    # 等于把每一篇的证据都收进来(实测 248 条,两篇雷同)。p3/p4 的 canon
    # 覆盖由 collect_from_canon 按论文号精确给出,不需要再靠追踪。
    "p3-witcert-v": [],
    "p4-moe-protect": [],
}


def collect_traced(slug):
    """执行期追踪:挂钩 builtins.open,逐个运行消费者,记录真实读取的
    experiments/out* 文件 —— regex 推导会漏动态加载(p1 dist 终验实证),
    追踪不会。消费者的落盘副作用(canon/figs 重生成)与 monorepo 常规
    重产等价,无害。"""
    import builtins
    import runpy
    seen = set()
    real_open = builtins.open

    def spy(f, *a, **k):
        try:
            fp = os.path.abspath(os.fspath(f))
            marker = os.sep + os.path.join("experiments", "out")
            if marker in fp and ("r" in str(a[0]) if a else True):
                rel = os.path.relpath(fp, ROOT)
                seen.add(rel)
        except Exception:
            pass
        return real_open(f, *a, **k)

    builtins.open = spy
    try:
        for src in CONSUMERS.get(slug, []):
            sp = os.path.join(ROOT, src)
            argv0 = sys.argv[:]
            sys.argv = [sp] + (["--json"] if "flagship" in src else [])
            try:
                runpy.run_path(sp, run_name="__main__")
            except SystemExit:
                pass
            except Exception as ex:
                print("  追踪警告 %s: %r" % (src, ex))
            finally:
                sys.argv = argv0
    finally:
        builtins.open = real_open
    return {r for r in seen if os.path.exists(os.path.join(ROOT, r))}


def collect(slug):
    files = set()
    for src in CONSUMERS.get(slug, []):
        t = open(os.path.join(ROOT, src), encoding="utf-8").read()
        # 直接字面引用
        files |= set(re.findall(r'"([A-Za-z0-9_.]+\.json)"', t))
        files |= set(re.findall(r'"([A-Za-z0-9_.]+\.rank\*?)"', t))
        # 动态 (rid, tag) 元组:J("%s_%s.json" % rid_tag) 形态
        for rid, tag in re.findall(r'\("([a-z0-9]+)",\s*"([a-z0-9_]+)"[,)]', t):
            files.add("%s_%s.json" % (rid, tag))
        # ROUNDS 三元组 ("ident", "rid", "tag"):捕获后两元
        for rid, tag in re.findall(
                r'\("[a-zA-Z0-9_]+",\s+"([a-z0-9]+)",\s+"([a-z0-9_]+)"\)', t):
            files.add("%s_%s.json" % (rid, tag))
    exist = sorted(f for f in files
                   if os.path.exists(os.path.join(OUT, f)))
    return exist


#: canon 每条都记着它的 source 产物 —— 对 p3/p4 这是**比追踪消费者更紧**的推导:
#: 出的包里恰好是"正文数字溯源所需的那些 run",不多不少。2026-08-17 出包时,
#: p3 的清单还停在骨架期通配(rrd_*/w2c*),漏了 39 条 w3* 证据 —— 手维护必漏,
#: 这正是本文件开头那句话。
def collect_from_canon(slug):
    import json
    m = re.match(r"p(\d+)-", slug)
    if not m:
        return set()
    pn = int(m.group(1))
    canon = os.path.join(ROOT, "papers", "p1-kv-certificates", "canon.json")
    d = json.load(open(canon, encoding="utf-8"))
    ent = d["entries"] if isinstance(d, dict) else d
    out = set()
    for e in ent:
        if e.get("paper") != pn:
            continue
        s = e.get("source") or ""
        if not s or s.startswith("derived:"):
            continue
        fp = s.split(":")[0]
        for base in ("", "experiments/out"):
            p = os.path.join(ROOT, base, fp)
            if os.path.isfile(p):
                out.add(os.path.relpath(p, ROOT))
                break
    return out


def collect_figs(slug):
    """**逐图**追踪出图器读了哪些产物,按 p3_figs 的 OWNER 归属到论文。

    整脚本追踪会把两篇的产物混在一起(实测 248 条,两篇完全相同)——
    对尚未发表的那一篇,那就是提前公开它的证据。"""
    import builtins
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    figs = importlib.import_module("p3_figs")
    real_open, cur, hits = builtins.open, {"fn": None}, {}

    def spy(f, *a, **k):
        try:
            fp = os.path.abspath(os.fspath(f))
            if os.sep + os.path.join("experiments", "out") in fp:
                hits.setdefault(cur["fn"], set()).add(
                    os.path.relpath(fp, ROOT))
        except Exception:
            pass
        return real_open(f, *a, **k)

    builtins.open = spy
    try:
        for name in dir(figs):
            fn = getattr(figs, name)
            if not (name.startswith("f") and callable(fn)
                    and name[1:2].isdigit()):
                continue
            cur["fn"] = name
            try:
                fn()
            except Exception as e:
                print("  警告:%s 追踪时报错(%s),其产物可能漏收" % (name, e))
    finally:
        builtins.open = real_open
    out = set()
    for name, files in hits.items():
        owner = figs.OWNER.get(name.split("_", 1)[0] if False else
                               _fig_key(figs, name), figs.P3)
        if owner == slug:
            out |= files
    return out


def _fig_key(figs, fname):
    """函数名 -> save() 时登记的图名(OWNER 的键)。"""
    for k in figs.OWNER:
        if k.startswith(fname.split("_")[0] + "_") or k == fname:
            return k
    return fname


def main():
    slug = sys.argv[1]
    rels = {"experiments/out/" + f for f in collect(slug)}
    rels |= collect_traced(slug)
    rels |= collect_from_canon(slug)
    rels |= collect_figs(slug)
    dst = os.path.join(ROOT, "papers", slug, "artifacts.list")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("# 机械推导+执行期追踪的引用产物清单"
                "(tools/cited_artifacts.py,勿手改)\n")
        for x in sorted(rels):
            f.write("%s\n" % x)
    print("→ %s(%d 个产物)" % (dst, len(rels)))


if __name__ == "__main__":
    main()
