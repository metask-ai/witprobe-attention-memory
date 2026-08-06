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
        for src in CONSUMERS[slug]:
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
    for src in CONSUMERS[slug]:
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


def main():
    slug = sys.argv[1]
    rels = {"experiments/out/" + f for f in collect(slug)}
    rels |= collect_traced(slug)
    dst = os.path.join(ROOT, "papers", slug, "artifacts.list")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("# 机械推导+执行期追踪的引用产物清单"
                "(tools/cited_artifacts.py,勿手改)\n")
        for x in sorted(rels):
            f.write("%s\n" % x)
    print("→ %s(%d 个产物)" % (dst, len(rels)))


if __name__ == "__main__":
    main()
