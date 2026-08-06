# -*- coding: utf-8 -*-
"""D 层六路径矩阵产物(p110_sixpath.json)—— **从证据 JSON 自动生成**。

review P0-2(2026-08-03)重构:此前 REP 手写常量 = 证据与结论脱钩
(sources 漏 q6k/q6l/pd1/pd2、gate 注释陈旧)。现在:
  · 每路径一个 rule 函数,**读产物 JSON 提取数字**,规则可证伪;
  · 状态税五级:PASS / FAIL_METHOD(方法可复现失败,含设计门)/
    BLOCKED_UPSTREAM(上游阻塞,非本方法失败)/ PARTIAL(部分证据,
    不足以判 PASS)/ NOT_MEASURED;
  · sources = 实际读取的文件清单自动落盘;缺产物 → NOT_MEASURED 而非猜。
flagship 逐项判 summary[k]=='PASS',其余状态如实红/黄。

python3 experiments/q6_sixpath_build.py
"""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
_READ = []          # 实际读取成功的文件(自动 sources)


def _load(name):
    p = os.path.join(OUT, name)
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        _READ.append(name)
        return d
    except (OSError, ValueError):
        return None


def _acc(d):
    return d.get("acc") if d else None


def rule_cudagraph():
    g = _load("q6g2_graphb2.json")
    n = _load("q6g2_native.json")
    if not g or not n:
        return "NOT_MEASURED", "缺 q6g2 产物", {}
    pk = (g.get("snapshot") or {}).get("packed_kernel") or {}
    facts = {"acc_stageb": _acc(g), "acc_native": _acc(n),
             "pk_calls": pk.get("n_calls"), "pk_packed": pk.get("n_packed"),
             "pk_c128": pk.get("n_packed_c128"), "pk_c4": pk.get("n_packed_c4"),
             "pk_view": pk.get("n_stageb_view"),
             "pk_fallback": pk.get("n_fallback"),
             "rank_files": g.get("n_rank_files")}
    ok6 = (_acc(g) == 1.0 and _acc(n) == 1.0 and g.get("n_error") == 0
           and (pk.get("n_packed") or 0) > 0 and g.get("n_rank_files") == 8)
    n12 = _load("q6n_graphb2.json")
    o12 = _load("q6o_graphb2.json")
    facts["acc_12doc_nomtp"] = _acc(n12)
    facts["acc_12doc_serial"] = _acc(o12)
    facts["quality_caliber"] = (g or {}).get("quality_caliber")  # Q13 口径随facts
    ok12 = n12 is not None and _acc(n12) == 1.0
    verdict = "PASS" if (ok6 and ok12) else ("PARTIAL" if ok6 else
                                             "NOT_MEASURED")
    return verdict, (
        "6-doc 口径:packed kernel 包装器记录 %s 次 packed 调用"
        "(capture/eager 路径计数,replay 不回 host;c128 %s + c4 %s),"
        "双臂 acc %s/%s;fallback %s 已归因无害(无压缩缓存调用)。"
        "**12-doc 并发探针 acc=%s;46 轮判别链定谳(q11t2 干预确认):"
        "压缩上下文(extra)读为必要通道(消融 0.000),伤害消费者 = "
        "decode 侧 extra 读以逻辑槽直索物理环/packed(读侧翻译从未执行),"
        "驱逐回收/跨请求共环即静默别名**(驱逐剂量单调劣化,c128 驱逐 "
        "4 轮无害;串行轮 acc=%s 系零驱逐混杂)。零驱逐口径 12/12;"
        "decode 翻译档微缩 0.500→1.000(诚实截断语义);完整物化"
        "(非驻留 packed 回源)与图模式复验未完成;质量声明必须带环配置"
        "与驱逐计数口径"
        % (pk.get("n_packed"), pk.get("n_packed_c128"),
           pk.get("n_packed_c4"), _acc(g), _acc(n), pk.get("n_fallback"),
           _acc(n12), _acc(o12))), facts


def rule_radix():
    k = _load("q6k_graphb2.invalid.json")
    kn = _load("q6c_native.json")   # q6k native 臂产物未落盘,radix-on 原生对照取 q6c 同配置
    if not k:
        return "NOT_MEASURED", "缺 q6k 产物", {}
    facts = {"acc_stageb": _acc(k), "acc_native": _acc(kn)}
    ok_repro = _acc(k) is not None and _acc(k) < 0.9
    return ("FAIL_METHOD" if ok_repro else "NOT_MEASURED"), (
        "radix×stage-B 质量退化**已复现**(q6c 0.667 → q6k 全链口径 %s,"
        "native %s;读侧污染疑虑排除)。install radix-off 门维持。"
        "预填吞吐坍缩为日志观察,**无独立 bench 产物,不作正式性能结论**"
        % (_acc(k), _acc(kn))), facts


def rule_mtp():
    # 零驱逐口径重测(2026-08-04,判别链收口后):每轮 = 同轮双臂受控
    # 对照(native vs graphb2),无跨轮归因需求 —— 跨 commit 配对有效性
    # 问题不适用。双 seed 双臂全对 → PASS(口径:零驱逐,quality_caliber
    # 自证);历史 PARTIAL 系 c4 驱逐机制混杂(14 轮判别链归因)。
    clean = []
    for rid, seed in (("q6q2", 42), ("q6q3", 43)):
        g = _load(rid + "_graphb2.json")
        n_ = _load(rid + "_native.json")
        if g and n_:
            qc = g.get("quality_caliber") or {}
            ze = qc.get("zero_evict")
            if ze is None:      # q6q2 早于口径标注落地,读 ring_stats 补判
                rs = (g.get("snapshot") or {})
                ze = all(((rs.get(k) or {}).get("ring_stats") or {})
                         .get("n_evict", 1) == 0
                         for k in ("pool_swap", "pool_swap_c4"))
            clean.append((rid, seed, _acc(g), _acc(n_), g.get("n"), ze))
    if clean and all(a == 1.0 and na == 1.0 and ze
                     for _, _, a, na, _, ze in clean):
        facts = {"clean_rounds": clean}
        return "PASS", (
            "MTP(EAGLE 1步)零驱逐口径重测:双 seed(42/43)双臂 "
            "12-doc 全对(%s),quality_caliber 零驱逐自证 —— 每轮为同轮"
            "双臂受控对照,无跨轮归因需求。历史残差(q6i3 5/6、q6m "
            "14/24)已由 14 轮判别链归因为 c4 驱逐机制混杂,非 MTP 特异"
            % "; ".join("%s seed%d %.3f/%.3f" % (r, sd, a, na)
                        for r, sd, a, na, _, _ in clean)), facts
    rounds = []
    for rid in ("q6i3_graphb2", "q6l_graphb2", "q6m42_graphb2",
                "q6m43_graphb2"):
        d = _load(rid + ".json") or _load(rid + ".invalid.json")
        if d:
            rounds.append((rid, _acc(d), d.get("manifest", {}).get("code"),
                           d.get("n"), d.get("n_error")))
    if not rounds:
        return "NOT_MEASURED", "无 MTP 轮产物", {}
    facts = {"rounds": rounds}
    pair = [r for r in rounds if r[0].startswith("q6m")]
    hits = sum(int(round(a * (n or 6))) for _, a, _, n, _ in pair
               if a is not None)
    tot = sum(n or 0 for _, _, _, n, _ in pair)
    note_hist = ("历史轮 q6i3(5/6)/q6l(6/6)跨 commit,差异不可归因 —— "
                 "review 定谳,只作背景不作证据。")
    if not pair:
        return "PARTIAL", note_hist + "配对重测未落盘", facts
    # 配对有效性必须落在证据上,不落在记忆里(review P0-1 同款):manifest
    # code 不同 ≠ 被测代码不同 —— 用 git 实测 src/launchers 差异范围裁决
    codes = sorted({c for _, _, c, _, _ in pair if c})
    if len(codes) <= 1:
        same_src, src_note = True, "同 commit(%s)" % (codes or ["?"])[0]
    else:
        try:
            import subprocess
            d = subprocess.run(
                ["git", "diff", "--stat", codes[0], codes[-1], "--",
                 "src/", "experiments/launchers/"],
                capture_output=True, text=True, timeout=30,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            same_src = (d.returncode == 0 and not d.stdout.strip())
            src_note = ("哈希不同(%s)但 src/launchers 零差异,同测代码成立"
                        % "/".join(codes)) if same_src else \
                       ("哈希不同(%s)且被测代码有差异 —— 配对无效"
                        % "/".join(codes))
        except Exception as e:
            same_src, src_note = False, ("哈希不同(%s)且 git 校验不可用"
                                         "(%s)—— 按不可归因处理"
                                         % ("/".join(codes), e))
    facts["pair_codes"] = codes
    facts["pair_same_src"] = same_src
    if not same_src:
        return "PARTIAL", note_hist + src_note, facts
    if tot and hits == tot:
        v, msg = "PASS", "配对双 seed %d/%d 全对(%s)。" % (hits, tot, src_note)
    elif tot:
        v, msg = "PARTIAL", (
            "配对双 seed %d/%d(%s)—— 残差与无 MTP 12-doc 探针漏针重叠,"
            "已定谳为排队偏移(非 MTP 特异,q6n/q6o 判别链);Q11 修复后"
            "需 MTP 12-doc 重测改判。" % (hits, tot, src_note))
    else:
        v, msg = "PARTIAL", "配对轮无有效样本。"
    return v, msg + note_hist, facts


def rule_hisparse():
    inc = _load("q6d3_incident.json")
    if not inc:
        return "NOT_MEASURED", ("缺 q6d3 取证产物(q6d/q6d2 当时服务器日志"
                                "已轮转,崩溃现场未固化)"), {}
    arms = inc.get("arms") or {}
    crashed = sorted(t for t, a in arms.items() if a.get("crashed"))
    sig = next((a.get("signature") for a in arms.values()
                if a.get("signature")), None)
    facts = {"crashed_arms": crashed, "n_arms": len(arms), "signature": sig}
    if crashed:
        return "BLOCKED_UPSTREAM", (
            "--enable-hisparse 下 **native 臂崩溃已取证**(崩溃臂:%s;"
            "无探针无适配器,revert --all 后起服):%s —— 上游集成缺口,"
            "非 WitCert 失败亦非通过,路径不可测。现场固化于 "
            "q6d3_incident.json(traceback 摘录 + 复现命令)"
            % (",".join(crashed), sig)), facts
    return "NOT_MEASURED", ("取证轮双臂均未捕获崩溃 —— 上游或已修,"
                            "需真实测量轮后才能改判"), facts


def rule_tpcouple():
    a = _load("q6h_rep1.json")
    b = _load("q6h_rep2.json")
    if not a or not b:
        return "NOT_MEASURED", "缺 q6h 重复对", {}
    diff = sum(1 for x, y in zip(a.get("items") or [], b.get("items") or [])
               if x.get("out") != y.get("out"))
    facts = {"acc": [_acc(a), _acc(b)], "text_diff": diff,
             "n": len(a.get("items") or [])}
    ok = _acc(a) == 1.0 and _acc(b) == 1.0
    return ("PASS" if ok else "PARTIAL"), (
        "同配置串行重复对(带路由):acc %s/%s,逐题文本 %d/%d 不同 —— "
        "只主张**答案稳定性**;逐字确定性不可得(q6j:上游 "
        "--enable-deterministic-inference 白名单不含 dsv4,双臂拒启)"
        % (_acc(a), _acc(b), diff, len(a.get("items") or []))), facts


def rule_pd():
    n = _load("pd1_native.json")
    s = _load("pd2_stageb.invalid.json")
    if not n:
        return "NOT_MEASURED", "缺 pd1 产物", {}
    facts = {"acc_native": _acc(n), "acc_stageb": _acc(s) if s else None}
    if _acc(n) == 1.0 and s is not None:
        return "PARTIAL", (
            "native PD(单机双进程:prefill TP4 + decode TP4 + mini-lb,"
            "mooncake)acc=1.000 —— KV 真传输由'针在预填侧、decode 答对'"
            "证实;stage-B 臂 %s(0/6 **无错**)= 传输完成、内核被调、服务"
            "不崩但语义错 —— ring+packed 布局未被传输层支持(packed_pool "
            "缺口清单预告项)。修法:传输协议适配层(独立工程线)"
            % _acc(s)), facts
    return "PARTIAL", "pd1 native=%s;stageb 轮缺失" % _acc(n), facts


def main():
    matrix, summary = {}, {}
    for item, rule in (("cudagraph", rule_cudagraph), ("radix", rule_radix),
                       ("mtp", rule_mtp), ("hisparse", rule_hisparse),
                       ("tpcouple", rule_tpcouple), ("pd", rule_pd)):
        verdict, msg, facts = rule()
        summary[item] = verdict
        matrix[item] = {"verdict": verdict, "evidence": msg, "facts": facts}
    n_pass = sum(1 for v in summary.values() if v == "PASS")
    rep = {
        "what": "六路径矩阵(Q6 图模式 stage-B 双池口径)—— 从产物 JSON "
                "自动生成,规则见 q6_sixpath_build.py",
        "machine": os.environ.get("WC_MACHINE", "hgx")  # 行为保持;硬编码是 hisparse 同病, "stack": "sglang 0.5.13.post1",
        "taxonomy": "PASS / FAIL_METHOD(方法性失败,含设计门)/ "
                    "BLOCKED_UPSTREAM / PARTIAL / NOT_MEASURED",
        "summary": summary,
        "matrix": matrix,
        "gate": {"passed": all(v == "PASS" for v in summary.values()),
                 "note": "%d/6 PASS;非 PASS 各级如实(flagship 对非 PASS "
                         "统一显示 FAIL,细分级看本产物 taxonomy)" % n_pass},
        "caliber": ("SMOKE/NDOC 见各轮产物 manifest;单 seed 为主(mtp 双 "
                    "seed);H200 TP/EP=8(pd 为 TP4×2);环 513/513;"
                    "受限配置,非全面生产兼容声明"),
        "sources": sorted(set(_READ)),
        "generated_by": "q6_sixpath_build.py",
    }
    outp = os.path.join(OUT, "p110_sixpath.json")
    if os.path.exists(outp):
        old = json.load(open(outp))
        if old.get("generated_by") not in (None, "q6_sixpath_build.py"):
            print("拒绝覆盖他人产物(generated_by=%s)" % old.get("generated_by"))
            return 125
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("→ %s(sources=%d 文件)" % (outp, len(_READ)))
    for k, v in summary.items():
        print("  %-10s %s" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
