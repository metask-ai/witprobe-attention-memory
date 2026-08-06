"""论文数字守卫:用冻结清单(papers/p1-kv-certificates/canon.json)核查**所有**论文文档。

## 为什么需要它

2026-07-28 的两次核查各暴露一类问题:
 1. 论文数字与实验 JSON 不符 —— 13 处硬性错误,三处在摘要里,偏差方向几乎全朝着高估自己;
 2. **中文稿被约束、英文稿没有** —— 旧版守卫只对中文稿做正向核查,对 `main.tex` 与
    `abstract_en.md` 只查禁用值(正向 0/0)。结果代码比论文成熟得快时英文侧一路漂移:
    main.tex 同时存在"Yi 因 rope_theta 更难"与后文的撤回;同时存在"SGLang 存储侧未接入"
    与后文"存储侧已接入"的整节;abstract_en 停在两模型 / LongBench 未跑 / 存储侧未接。

## 机制

- 数字唯一来源是 `experiments/out/*.json`,由 `tools/make_canon.py` 编译成 `papers/p1-kv-certificates/canon.json`;
- 本测试对 **中文稿 / main.tex / abstract_en.md** 三份都做正向核查:
  清单里每条数字的"数值 token"必须出现在文档中(LaTeX 格式已归一);
- 同时对三份都做反向核查:已证伪的旧值不得出现(带更正上下文的除外);
- 重跑实验后若数字变了,先 `python tools/make_canon.py`,再改论文,守卫会指出遗漏之处。

python tests/test_paper_claims.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "papers", "p1-kv-certificates", "canon.json")
# (标签, 路径, 是否只查 headline 子集, 归属论文)——摘要不可能承载全部数字,只受头条约束。
# 归属论文用来切分冻结清单:论文2 的数字冻结进 canon 后不应去论文1 的文档里找,
# 否则一加新数字就把论文1 的守卫打红。文档不存在时跳过并提示(论文2 尚未动笔)。
DOCS = [("中文稿", os.path.join(ROOT, "papers", "p1-kv-certificates", "draft_zh_v1.0.md"), False, 1),
        ("main.tex", os.path.join(ROOT, "papers", "p1-kv-certificates", "arxiv", "main.tex"), False, 1),
        ("abstract_en", os.path.join(ROOT, "papers", "p1-kv-certificates", "abstract_en.md"), True, 1),
        ("mlsys_en", os.path.join(ROOT, "papers", "p1-kv-certificates", "mlsys", "main.tex"), True, 1),
        ("mlsys_zh", os.path.join(ROOT, "papers", "p1-kv-certificates", "mlsys", "witcert_mlsys_zh.md"), True, 1),
        ("论文2 正文", os.path.join(ROOT, "papers", "p2-attention-memory", "main.tex"), False, 2)]

# 已被证伪 / 无来源的值,不得出现(带更正上下文的除外)
FORBIDDEN = [
    ("94.31", "RULER WitCert 旧错值,真值 94.42"),
    ("93.31", "RULER EA 旧错值,真值 93.88"),
    ("29–87", "换页率下降上限错,实测 28.7–77.0%"),
    ("29--87", "同上(LaTeX)"),
    ("87.5%", "四联报覆盖率旧错值,真值 79.2%"),
    ("87.5\\%", "同上(LaTeX)"),
    ("3.227", "四联报时延旧错值,真值 8.627 ms"),
    ("48.1%", "字节口径旧错值(仅 key 侧),真值 K+V 46.5%"),
    ("48.1\\%", "同上(LaTeX)"),
    ("+7.5~9.2%", "证书开销无来源"),
    ("7.5--9.2", "同上(LaTeX)"),
    ("0/262144", "4 输出复用后实为 1/262144 舍入平局"),
    # 2026-07-31 被度量类型检查废掉的三个组合数:相对见证与 TV 直接相加,不属任何度量
    ("0.0231", "p64 单层'端到端界':跨度量相加,连界都不是(p76 为合法重建)"),
    ("0.02406", "p64 V4 组合界:同上"),
    ("0.982", "p64 请求级'界':同上;禁与 p76 的聚合预算比较"),
    # 2026-07-31 三审:哨兵数学曾按无放回超几何建模,实现是 torch.randint 有放回
    ("0.897", "哨兵单块检出概率(无放回模型,与实现不符);有放回真值 0.881"),
    ("146/72/35/17", "哨兵检出延迟(无放回模型);有放回为 148/74/37/19"),
    ("46.67", "EA qa_2 旧错值,真值 48.89"),
    ("1.7--1.8", "P22 的 K-as-V 代理结论已撤回"),
    ("1.7–1.8", "同上(中文)"),
    ("64.9–81.5", "两模型族旧区间;单一 δ=10⁻² 口径为 54.4–81.0%"),
    ("64.9--81.5", "同上(LaTeX)"),
    ("0.866", "serving 覆盖率旧口径(逐步局部分配);请求级为 0.812"),
    ("0.987", "同上;请求级为 0.940"),
    ("86.6\\%", "同上(遥测叙述);请求级为 81.3%"),
    ("75.9\\%", "同上;请求级为 75.3%"),
    ("21,765", "clamp(40) 截断伪影;log 域真值为 26,433×"),
    ("21{,}765", "同上(LaTeX)"),
    ("53.9–81.5", "跨 δ 档拼接区间(53.9 来自 δ=10⁻⁴,81.5 来自 δ=5·10⁻²);头条口径应为 δ=10⁻² 下 54.4–81.0%"),
    ("53.9--81.5", "同上(LaTeX)"),
    ("12--52", "tanh 旧区间,应为 11.8–52.0%"),
    ("535.3 vs 536.2", "单次测量噪声;规范测量后证书代价为 6.4%/16.2%"),
    ("证书在生产中实质免费", "同上"),
    ("证书在生产里实质免费", "同上"),
    ("free in production", "同上(英文)"),
    ("推测与 rope_theta", "P33 已给出实测答案:是逐块 scale 所致"),
    ("rope\\_theta} of $5{\\times}10^{6}$ is far larger", "同上;该推测已撤回"),
    ("storage} side (dithered INT8 pool $+$ outlier bypass) is \\textbf{not} integrated",
     "存储侧已接入并端到端跑通"),
    ("storage** side (dithered-INT8 pool + outlier bypass) is not", "同上"),
]
EXEMPT = ("已更正", "旧错值", "此前记为", "已废弃", "真值", "已删除", "实为", "已撤回", "已重做",
          "不成立", "retracted", "corrected", "原先", "superseded", "no longer", "earlier",
          "not the", "instead of", "两族", "we retract", "We retract", "噪声", "noise")


def norm(t: str) -> str:
    """归一化 LaTeX 与排版差异,使数值 token 可跨文档比较。"""
    t = t.replace("{,}", ",").replace("\\%", "%").replace("\\,", "")
    t = re.sub(r"\$?\\times\$?", "×", t)
    t = t.replace("$-$", "−").replace("--", "–").replace("-", "−")
    t = re.sub(r"\$([^$]*)\$", r"\1", t)          # 去掉行内数学环境
    return t


def variants(tok: str):
    """千分位可有可无:32,256 与 32256 视为同一数。"""
    return {tok, tok.replace(",", "")}


def tokens(value: str):
    """从清单值里取出必须出现的数值 token。只保留有辨识度的(含小数点/千分位/≥3 位)。"""
    out = []
    for m in re.findall(r"\d[\d.,]*\d|\d", value):
        m = m.rstrip(".,")
        if "." in m or "," in m or len(m) >= 3:
            out.append(m)
    return out


#: 骨架期豁免:正文里带此标记的论文只报告缺口、不判失败。
#: 写作时**每写完一节就删掉对应的 TODO**,全部删完后去掉本标记,守卫即转为强制。
DRAFT_MARK = "\\emph{TODO}"

#: 零事件/短数字条目的上下文核查(2026-08-06 评审:tokens() 的辨识度过滤
#: 把 23 条静默跳过 —— 零违约类**最强主张**恰好落在守卫的结构盲区)。
#: 键 = (group, name);正则须命中该论文每个在检文档(headline 域同普通条目)。
#: 上膛纪律:先验证今天全部在检文档正向命中再入表;命不中的(如 0/28 缺席
#: abstract_en、12/12 未入 p2 正文)留在**可见**盲区清单,措辞入文后再上膛。
CONTEXT_PATTERNS = {
    ("128k覆盖率", "违约"):
        r"zero violations|no violations|0 violations|零违约|无违约|违约[^\n]{0,15}0",
    ("R8平台", "坏块哨兵假阳性率"):
        r"false-positive rate of zero|假阳性率[^\n]{0,8}0|0% false",
}


def check(label, path, entries, headline_only=False, optional=False):
    if not os.path.exists(path):
        _rel = (os.environ.get("WITCERT_RELEASE") == "1"
                or os.path.exists(os.path.join(ROOT, "RELEASE.json")))
        if optional or _rel:
            # 发布仓是单论文子集:不随发布的内部稿(中文稿/abstract)缺席
            # 属设计;monorepo 不设该 env,缺文件仍判红
            print(f"  [{label}] 尚未动笔或不随发布,跳过"
                  f"({len(entries)} 条数字已冻结)")
            return 0, 0
        print(f"  [{label}] 文件不存在"); return 1, 0
    raw = open(path, encoding="utf-8").read()
    drafting = DRAFT_MARK in raw
    text = norm(raw)
    if headline_only:
        entries = [e for e in entries if e.get("headline")]
    miss, blind = [], []
    for e in entries:
        pat = CONTEXT_PATTERNS.get((e["group"], e["name"]))
        if pat:
            if not (re.search(pat, text) or re.search(pat, raw)):
                miss.append(f"    缺 [{e['group']}/{e['name']}] = {e['value']}"
                            f"  (零事件上下文模式未命中)")
            continue
        tk = tokens(e["value"])
        if not tk:
            blind.append(f"{e['group']}/{e['name']}")
            continue
        absent = [t for t in tk if not any(v in text for v in variants(t))]
        if absent:
            miss.append(f"    缺 [{e['group']}/{e['name']}] = {e['value']}  (未找到 {absent})")
    bad = []
    for tok, why in FORBIDDEN:
        i = raw.find(tok)
        while i >= 0:
            ctx = raw[max(0, i - 150): i + 150]
            if not any(m in ctx for m in EXEMPT):
                bad.append(f"    禁用值 {tok!r}({why})")
                break
            i = raw.find(tok, i + 1)
    n = sum(1 for e in entries
            if tokens(e["value"]) or (e["group"], e["name"]) in CONTEXT_PATTERNS)
    if drafting:
        n_todo = raw.count(DRAFT_MARK)
        print(f"  [{label}] **骨架期**({n_todo} 处 TODO):正向 {n - len(miss)}/{n},"
              f"待写入 {len(miss)} 个数字;禁用值命中 {len(bad)}")
        for m in bad:
            print(m)
        return 0, len(bad)          # 骨架期不判失败,但禁用值仍然拦
    print(f"  [{label}] 正向 {n - len(miss)}/{n},禁用值命中 {len(bad)}"
          + (f",盲区 {len(blind)} 条(无辨识 token 且未配上下文模式)"
             if blind else ""))
    for m in miss + bad:
        print(m)
    return len(miss), len(bad)


def check_figdata():
    """figdata.json(图数据单一来源)的每个数字必须出现在 main.tex ——
    图与正文只能同源;figdata 改了正文没改即红(2026-08-06 四图两表)。"""
    import json as _j
    fd_p = os.path.join(ROOT, "papers", "p2-attention-memory", "figdata.json")
    tex_p = os.path.join(ROOT, "papers", "p2-attention-memory", "main.tex")
    if not (os.path.exists(fd_p) and os.path.exists(tex_p)):
        print("  figdata⊆prose: 跳过(论文2 不在本仓 —— 单论文发布仓)")
        return []
    tex = open(tex_p, encoding="utf-8").read()
    tex_flat = tex.replace("{,}", "")
    def nums(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("source", "_note", "label", "vacuous"):
                    continue
                yield from nums(v)
        elif isinstance(o, list):
            for v in o:
                yield from nums(v)
        elif isinstance(o, (int, float)) and o is not None:
            yield o
    bad = []
    for n in nums(_j.load(open(fd_p, encoding="utf-8"))):
        if n == 0 or n is None:
            continue
        af = abs(float(n))
        s = str(int(af)) if af == int(af) else ("%g" % af)
        if s not in tex_flat:
            bad.append(s)
    if bad:
        print("  figdata⊆prose: 失败", bad)
    else:
        print("  figdata⊆prose: 通过(图数据与正文同源)")
    return bad


def audit_canon_sources(entries):
    """canon 溯源棘轮(2026-08-06 评审 C1:编译器里的手打值与产物脱钩,
    正是整套体系要防的失效在自己家里复现)。新条目必须带 source
    ("产物文件:路径说明" 或 "derived:推导口径");存量未溯源按 (group,name)
    冻结于 tests/canon_unsourced_baseline.json,只许减少;指向文件的
    source,文件必须真实存在。"""
    base = {tuple(x) for x in json.load(open(
        os.path.join(ROOT, "tests", "canon_unsourced_baseline.json"),
        encoding="utf-8"))}
    fails = []
    unsourced = [(e["group"], e["name"]) for e in entries if not e.get("source")]
    for g, n in unsourced:
        if (g, n) not in base:
            fails.append(f"    canon 新条目未声明 source:[{g}/{n}]"
                         "(手打值必须给产物来路或 derived: 口径)")
    for e in entries:
        s = e.get("source") or ""
        if s and not s.startswith("derived:"):
            fp = s.split(":")[0]
            if not any(os.path.exists(os.path.join(ROOT, d, fp))
                       for d in ("", "experiments/out")):
                fails.append(f"    canon source 指向不存在的文件:"
                             f"[{e['group']}/{e['name']}] -> {fp}")
    print(f"  canon 溯源:已溯源 {len(entries) - len(unsourced)},"
          f"存量未溯源 {len(unsourced)}(基线 {len(base)},只许减少)")
    if len(unsourced) < len(base):
        print(f"    基线可收紧:{len(base)} -> {len(unsourced)}"
              "(更新 tests/canon_unsourced_baseline.json)")
    return fails


def main():
    if not os.path.exists(CANON):
        print("papers/p1-kv-certificates/canon.json 不存在,请先运行 python tools/make_canon.py")
        sys.exit(1)
    entries = json.load(open(CANON, encoding="utf-8"))["entries"]
    by_paper = {}
    for e in entries:
        by_paper.setdefault(e.get("paper", 1), []).append(e)
    print(f"冻结清单 {len(entries)} 条(来源 experiments/out/*.json,由 tools/make_canon.py 生成);"
          + "按论文切分:" + ", ".join(f"论文{p} {len(v)} 条" for p, v in sorted(by_paper.items()))
          + "\n")
    tm = tb = 0
    for label, path, ho, paper in DOCS:
        sub = by_paper.get(paper, [])
        if not sub:
            continue
        m, b = check(label, path, sub, ho, optional=(paper != 1))
        tm += m; tb += b
    # 定理名守卫:正文里 \texttt{snake\_case} 引用的定理必须真实存在于
    # Lean 导出(papers/*/theorems.json)。历史上两次证书公式被外部评审
    # 证伪,根因都是"纸上写的"与"实际成立的"分家 —— 名字层面同理:引用
    # 一个不存在(或改名后失联)的定理名,散文照样通顺,只有对账能抓。
    tn = 0
    for paper_dir, tex in (("p2-attention-memory", "main.tex"),):
        tj = os.path.join(ROOT, "papers", paper_dir, "theorems.json")
        tx = os.path.join(ROOT, "papers", paper_dir, tex)
        if not (os.path.exists(tj) and os.path.exists(tx)):
            continue
        names = json.load(open(tj, encoding="utf-8"))["names"]
        body = open(tx, encoding="utf-8").read()
        # 字符类必须含大写:首版只认 [a-z],"ledger\_soundX" 这类错名整个
        # 不被匹配 → 静默逃逸(变异检验当场揭穿,守卫险成装饰品)
        cited = {m.replace("\\_", "_")
                 for m in re.findall(r"\\texttt\{([A-Za-z][A-Za-z0-9\\_]*)\}",
                                     body)
                 if "\\_" in m}
        missing = sorted(c for c in cited
                         if not any(n.endswith("." + c) or n == c
                                    for n in names))
        for c in missing:
            print(f"    [{paper_dir}] 正文引用定理名 \\texttt{{{c}}} 不在 "
                  f"Lean 导出中(theorems.json)")
        tn += len(missing)
        if cited and not missing:
            print(f"  [{paper_dir}] 定理名对账:{len(cited)} 个引用全部命中 "
                  f"Lean 导出({len(names)} 条)")
    fig_bad = check_figdata()
    src_bad = audit_canon_sources(entries)
    for s in src_bad:
        print(s)
    print()
    if tm or tb or tn or fig_bad or src_bad:
        print(f"FAILED: {tm} 处数字缺失/不符,{tb} 处使用了已证伪的值,"
              f"{tn} 处定理名失联,{len(fig_bad)} 处图数据与正文脱钩,"
              f"{len(src_bad)} 处 canon 溯源问题")
        print("提示:数字变了先跑 tools/make_canon.py,再同步三份文档")
        sys.exit(1)
    print("ALL PAPER CLAIM CHECKS PASSED")


if __name__ == "__main__":
    main()
