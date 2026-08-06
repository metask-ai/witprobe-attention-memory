"""从 experiments/out/*.json 生成冻结的"数字与口径清单" papers/p1-kv-certificates/CANON.md。

动机:代码比论文成熟得快,导致中文稿、英文稿、摘要、README 相互矛盾(2026-07-28 实测:
英文 main.tex 仍说 Yi 因 rope_theta 更难、仍说 SGLang 存储侧未接入,而后文自己又反过来;
abstract_en 停在两模型 / LongBench 未跑 / 存储侧未接;README 停在 v0.1)。

对策:**唯一数字来源是实验 JSON**。本脚本把它们编译成一份带口径的清单,
`tests/test_paper_claims.py` 用这份清单同时核查中文稿、main.tex、abstract_en.md —
任何一份文档漏掉或写错一个数字都会 FAIL。

    python tools/make_canon.py            # 生成 papers/p1-kv-certificates/CANON.md + papers/p1-kv-certificates/canon.json
"""
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "out")


def J(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def ruler():
    res = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "experiments", "out_siteB", "ruler_results", "*", ""))):
        f = os.path.join(d, "metrics.json")
        if not os.path.exists(f):
            continue
        m = json.load(open(f, encoding="utf-8"))
        v = [x["string_match"] for x in m.values()]
        res[os.path.basename(d.rstrip("/")).split("Instruct__")[1]] = (sum(v) / len(v), m)
    return res


def build():
    C = []           # (分组, 名称, 数值字符串, 口径)
    def add(g, n, v, scope, headline=False, paper=1, src=None):
        # headline=True 的条目摘要里也必须出现;其余只约束正文两版
        # paper 指该数字归属哪篇论文:守卫只拿本篇的条目去查本篇的文档,
        # 否则论文2 的数字一冻结就会把论文1 的守卫打红
        # src(2026-08-06 评审 C1):值的机器可溯来源 —— "文件.json:路径说明"
        # 或 "derived:推导口径"。本编译器要防的失效正是"数字与产物脱钩",
        # 手打值必须显式声明来路;存量未溯源条目由 test_paper_claims 按
        # (group,name) 冻结棘轮,新条目必须带 src。
        C.append({"group": g, "name": n, "value": v, "scope": scope,
                  "headline": headline, "paper": paper, "source": src})

    # ---- 理论 / 形式化 ----
    add("理论", "反例倍数", "7.8", "两 token 反例:真实 TV 是错误式给出的『界』的 7.8 倍", headline=True)
    add("理论", "错误式通过的检查组数", "32,256", "均值口径检查数;逐 query 检查才暴露问题")
    # 定理计数由 tools/lean_extract.py 从 formal/ 抽取,不手写
    th = json.load(open(os.path.join(ROOT, "papers", "p1-kv-certificates", "theorems.json"), encoding="utf-8"))
    add("理论", "Lean 主定理数", str(th["n_main"]),
        "Mathlib 层 L1–L4,公理仅 propext/Classical.choice/Quot.sound;由 lean_extract 抽取", headline=True,
        src="papers/p1-kv-certificates/theorems.json:n_main")
    add("理论", "机器检查定理总数", str(th["n_total"]),
        f"Mathlib 层 {th['n_mathlib']} + 零依赖层 {th['n_standalone']};"
        "由 formal/Export.lean 用 Lean 自己的 pretty-printer 与 collectAxioms 导出,"
        "陈述编译进 papers/p1-kv-certificates/THEOREMS.md 与 papers/p1-kv-certificates/arxiv/theorems.tex",
        headline=True,
        src="papers/p1-kv-certificates/theorems.json:n_total")
    add("理论", "公理并集", "、".join(th["axioms_union"]),
        "collectAxioms 机器给出,无 sorryAx",
        src="papers/p1-kv-certificates/theorems.json:axioms_union")

    # ---- T-E bootstrap CI(评审 B-3)----
    ci = json.load(open(os.path.join(OUT, "te_ci.json"), encoding="utf-8"))["configs"]
    def fmt_ci(c): return f"[{c[0]:+.1f},{c[1]:+.1f}]"
    add("RULER门控CI", "fp8 门控配对差", f"+{ci['fp8_gate5']['diff_vs_none']:.1f} " + fmt_ci(ci['fp8_gate5']['diff_ci95']),
        "逐样本配对 vs 满血,150 样本,bootstrap 10⁴;risk-ranked τ=5")
    add("RULER门控CI", "int8/kivi 门控配对差", "0.0",
        "两配置与满血在每个样本上判分完全一致(配对差恒 0)")
    add("RULER门控CI", "fp8 裸跑配对差", f"{ci['fp8_raw']['diff_vs_none']:.1f} " + fmt_ci(ci['fp8_raw']['diff_ci95']),
        "崩塌统计显著")
    add("RULER门控CI", "kivi 裸跑配对差", f"{ci['kivi_raw']['diff_vs_none']:.1f} " + fmt_ci(ci['kivi_raw']['diff_ci95']),
        "静默尾部从轶事升级为统计显著;复跑均值 71.3(原切片 73.7,run-to-run 漂移)")

    # ---- 128k 覆盖率 P38(评审 B-4)----
    p38 = json.load(open(os.path.join(OUT, "p38_long_coverage.json"), encoding="utf-8"))["results"]
    def _p38(S):
        rs = [r for r in p38 if r["S"] == S]
        return min(r["jkv"] for r in rs), max(r["jkv"] for r in rs), max(r["jkv_tanh"] for r in rs)
    lo8, hi8, _ = _p38(8192)
    lo32, hi32, _ = _p38(32768)
    lo128, hi128, t128 = _p38(131072)
    add("128k覆盖率", "8k 区间", f"{lo8*100:.1f}–{hi8*100:.1f}%",
        "Qwen2.5-1.5B,两域拼接流,P17 同配置,δ=10⁻²,KV-head any-of-G")
    add("128k覆盖率", "32k 区间", f"{lo32*100:.1f}–{hi32*100:.1f}%", "同上")
    add("128k覆盖率", "128k 区间", f"{lo128*100:.1f}–{hi128*100:.1f}%",
        "同上,YaRN factor=4(厂商官方长上下文配置);tanh 参考塌到 ≤0.2%", headline=False)
    add("128k覆盖率", "128k tanh 上限", f"{t128*100:.1f}%", "128k 下确定性 tanh 界的最好域")
    add("128k覆盖率", "违约", f"0",
        "三档 × 两域全部零违约(soundness 第四条独立证据链)")

    # ---- 双侧收紧界 P37(评审 A-2b 验证:负结果)----
    p37 = json.load(open(os.path.join(OUT, "p37_twosided_bound.json"), encoding="utf-8"))["results"]
    r8 = p37["rtn-8bit"]
    add("双侧界", "有效档中位反松", f"{1/r8['median_shrink_old_over_two']:.1f}×",
        "rtn-8bit(有效档)双侧聚合形/旧 e-form 的中位比;双侧形在小 c 区二阶劣化")
    add("双侧界", "有效档胜出比例", f"{r8['two_wins_frac']*100:.1f}%", "rtn-8bit,双侧形更紧的单元占比")
    add("双侧界", "coverage 变化", f"{r8['coverage_tau02_old']:.3f}→{r8['coverage_tau02_meter']:.3f}",
        "τ=0.2,min(1,旧,双侧,逐token min) 相对旧界;实质无变化")
    add("双侧界", "min形数值soundness",
        f"0/{sum(v['n'] for v in p37.values()):,}",
        "全部六量化器单元 meter ≥ 真 TV 的违约计数")
    add("双侧界", "饱和档最大缩减", f"{max(v['median_shrink_old_over_two'] for v in p37.values()):,.0f}×",
        "rtn-2bit 中位缩减(但仍远高于 1,meter 恒饱和,无操作点变化)")

    # ---- δ 曲线(三模型族;固定原始 6-7B 三族口径,H200 阶梯(14B-72B)单列 H200战役组)----
    _p17_three = [f for f in sorted(glob.glob(os.path.join(OUT, "p17_*.json")))
                  if not any(t in f for t in ("_14b", "_32b", "_72b"))]
    vals, tanh, red, dd, v_d2 = [], [], [], [], []
    for f in _p17_three:
        rows = json.load(open(f, encoding="utf-8"))
        doms = {}
        for r in rows:
            doms.setdefault(r["domain"], []).append(r)
        for _, rs in doms.items():
            for dk in ("jkv_0.0001", "jkv_0.001", "jkv_0.01", "jkv_0.05"):
                vals.append(statistics.mean(r[dk] for r in rs))
            s = statistics.mean(r["jkv_0.01"] for r in rs)
            v_d2.append(s)
            t = statistics.mean(r["jkv_tanh"] for r in rs)
            tanh.append(t)
            red.append(((1 - t) - (1 - s)) / (1 - t) * 100)
            dd.append((statistics.mean(r["jkv_0.05"] for r in rs)
                       - statistics.mean(r["jkv_0.0001"] for r in rs)) * 100)
    # 评审 A-3:头条区间必须取单一工作点(δ=10⁻²),不得跨 δ 档拼接极值
    add("δ曲线", "subG 覆盖率区间", f"{min(v_d2)*100:.1f}–{max(v_d2)*100:.1f}%",
        "三模型族(Qwen2.5-7B/Mistral-7B/Yi-1.5-6B)×三域,固定 δ=10⁻²,域内均值,KV-head any-of-G", headline=True)
    add("δ曲线", "tanh 覆盖率区间", f"{min(tanh)*100:.1f}–{max(tanh)*100:.1f}%", "同上,确定性 tanh 界", headline=True)
    add("δ曲线", "换页率相对下降", f"{min(red):.1f}–{max(red):.1f}%", "δ=10⁻² 下相对 tanh", headline=True)
    add("δ曲线", "δ 收紧 500× 最大损失", f"{max(dd):.1f}pp", "5×10⁻²→10⁻⁴,六个模型×域组合的最大值", headline=True)
    add("δ曲线", "模型族数", "3", "Qwen2.5-7B(8k)/Mistral-7B(8k)/Yi-1.5-6B(4k,受其位置上限)", headline=True,
        src="derived:三族枚举,清单即口径")

    # ---- RULER ----
    R = ruler()
    if R:
        w = R["witcert__0.40__fraction0.200"][0]
        e = R["expected_attention__0.40__fraction0.200"][0]
        b = R["no_press__0.00__fraction0.200"][0]
        add("RULER-4k", "WitCert 均分", f"{w:.2f}", "13 任务均值,Qwen2.5-7B,1300 样本,字节预算 46.5%", headline=True)
        add("RULER-4k", "满血均分", f"{b:.2f}", "no_press", headline=True)
        add("RULER-4k", "EA 均分", f"{e:.2f}", "expected_attention,其 compression_ratio=0.401 = 丢 40.1% token", headline=True)
        add("RULER-4k", "差值", f"+{w-e:.2f}", "配对 bootstrap p≈0.12,**不显著**;两列预算量纲不同", headline=True)
        add("RULER-4k", "字节节省", "46.5%", "K 145.1B + V 129B = 274.1B vs fp16 512B", headline=True,
        src="derived:1-274.1/512,字节数见口径")

    # ---- LongBench-E ----
    lb = J("longbench_e_summary.json")
    add("LongBench-E", "长度桶数", f"{lb['buckets']}", "5 任务 × 3 桶;lcc_e 三配置全部运行失败未计入", headline=True)
    add("LongBench-E", "WitCert 跌幅>1pp 桶数", f"WitCert {lb['witcert_drops_gt1pp']}/{lb['buckets']}", "")
    add("LongBench-E", "EA 跌幅>1pp 桶数", f"EA {lb['ea_drops_gt1pp']}/{lb['buckets']}", "")
    add("LongBench-E", "2wikimqa 4-8k WitCert", "−3.23", "唯一实质下降")
    add("LongBench-E", "2wikimqa 4-8k EA", "−7.79", "同桶,我们的 2.4 倍")
    add("LongBench-E", "对我方不利处", "+4.36", "EA 在 hotpotqa_e 8k+ 明显更好(如实列出)")

    # ---- 质量消融 ----
    p23 = J("p23_real_v_S8192.json")["results"]
    add("质量", "抖动增益(质量中性)", f"{p23['RTN']['rel_err']/p23['dither']['rel_err']:.2f}×",
        "真实 V,Qwen2.5-7B/8k;抖动本身质量中性", headline=True)
    add("质量", "离群旁路增益", f"{p23['RTN']['rel_err']/p23['RTN+离群']['rel_err']:.2f}×", "同上")

    # ---- 单卡 kernel ----
    p21 = J("p21_quad_S32768.json")
    add("kernel", "四联报覆盖率", f"{p21['coverage_witcert']*100:.1f}%", "S=32768,nq=24,τ=0.2,合成 K")
    add("kernel", "四联报时延", f"{p21['latency_ms']['WitCert-INT8']:.3f}", "ms")
    add("kernel", "四联报加速比",
        f"{p21['latency_ms']['WitCert-INT8']/p21['latency_ms']['fp16']:.2f}×",
        "**kernel 级;不可作为部署代价引用**,端到端见 serving 表", headline=True)
    add("kernel", "四联报误差", f"{p21['rel_err']['WitCert-INT8']:.1e}".replace("e-03", "e-3"), "")
    p24 = J("p24_packed_S32768.json")
    add("kernel", "真实显存节省",
        f"{(1-p24['measured_bytes']['packed']/p24['measured_bytes']['fp16'])*100:.1f}%",
        "torch.cuda.memory_allocated 实测,8k/32k/128k 一致", headline=True)
    p26 = J("p26_fallback_S32768.json")
    add("kernel", "回退后误差", f"{p26['err_fb_after']:.2e}".replace("e-04", "e-4"), "回退前 9.18e-3")
    add("kernel", "有效位宽", f"{p26['effective_bits']:.2f}", "门控后,净省 31.5%")
    add("kernel", "块级换页流量降低", f"{J('p27_blockpagein_S32768.json')['reduction']:.0f}×",
        "中位 1/128 块即回落阈值")
    p28 = J("p28_longcot.json")
    add("kernel", "长 CoT KV 头覆盖率", f"{max(r['cov_kv_head'] for r in p28):.3f}",
        "R1-Distill-7B 真实生成,n=4;与常规长文 0.792 持平")
    p29 = J("p29_sr_S32768.json")
    assert all(r["cov_sr"] == 0.0 for r in p29), "SR 覆盖率不再全为 0,论文否决结论需重写"
    add("kernel", "随机舍入分支覆盖率", "0.000", "所有离群预算下恒为 0 ⟹ 被数据否决")
    for r in J("p31_outlier_budget_S32768.json")["rows"]:
        add("kernel", f"m={r['m_pairs']} 覆盖率(实机)", f"{r['coverage']:.3f}", "S=32768,τ=0.2,合成 K")
    p30 = J("p30_e2e_m16.json")
    add("端到端解码", "m=16 覆盖率", f"{p30['cert_coverage']:.3f}", "HF Cache,28 层,真实 query")
    add("端到端解码", "m=16 KL", f"{p30['kl']:.2e}".replace("e-03", "e-3"), "teacher-forcing")
    add("端到端解码", "m=16 top-5 重叠", f"{p30['top5_overlap']:.3f}", "")
    add("端到端解码", "m=16 最大证书", f"{p30['cert_max']:.3f}", "距 τ=0.2 有 2.3× 余量")

    # ---- scale 粒度 ----
    p32 = J("p32_scale_granularity_S8192.json")["results"]
    A = [v for k, v in p32.items() if k.startswith("A")][0]
    g32 = [v for k, v in p32.items() if "32通道组" in k][0]
    add("scale粒度", "论文口径误差", f"{A['rel_err']:.3e}".replace("e-02", "e-2"), "逐块逐通道,分页不兼容")
    add("scale粒度", "G=32 相对论文", f"{g32['rel_err']/A['rel_err']:.2f}×", "逐 token 逐 32 通道组,分页兼容且更准", headline=True)

    # ---- 离群旁路 ----
    for tag, mn in (("Qwen2.5-7B-Instruct", "Qwen"), ("Yi-1.5-6B-Chat", "Yi")):
        p = J(f"p33_outlier_{tag}.json")
        add("离群旁路", f"{mn} 通道稳定性 Jaccard", f"{p['jaccard_mean']:.3f}", "前半/后半 token 各选 top-8 对")
        add("离群旁路", f"{mn} m=0 覆盖率", f"{p['coverage_by_m']['0']['coverage']:.3f}", "S=4096,τ=0.2,逐 token 组 scale")
        add("离群旁路", f"{mn} m=4 覆盖率", f"{p['coverage_by_m']['4']['coverage']:.3f}", "已排除 amax")
        add("离群旁路", f"{mn} m=16 覆盖率", f"{p['coverage_by_m']['16']['coverage']:.3f}", "")
    add("离群旁路", "只旁路不排除 amax 的降幅", "8%", "证书上界只降 8% ⟹ 必须同时排除")
    add("离群旁路", "旁路+排除 amax 的降幅", "32×", "合成对照:证书最大 2.378 → 0.0731")

    # ---- RULER 级门控评测(T-E) ----
    te = J("te_ruler.json")["rows"]
    add("RULER门控", "满血 avg", f"{te['none']['avg']:.1f}", "6 难任务×25样本=150条,7B,SGLang观测台", headline=True)
    add("RULER门控", "fp8 裸跑 avg", f"{te['fp8_raw']['avg']:.1f}", "灾难级;零分样本 114/150", headline=True)
    add("RULER门控", "fp8 门控 avg", f"{te['fp8_gate5']['avg']:.1f}",
        f"τ_gate=5;零分回到 {te['fp8_gate5']['n_zero']}(=满血);PCIe {te['fp8_gate5']['fallback']['mib']/1024:.1f} GiB 一次性", headline=True)
    add("RULER门控", "kivi 裸跑 avg", f"{te['kivi_raw']['avg']:.1f}",
        f"尾部损伤:零分 {te['kivi_raw']['n_zero']} vs 满血 {te['none']['n_zero']}(9 个静默失败)", headline=True)
    add("RULER门控", "kivi 门控 avg", f"{te['kivi_gate02']['avg']:.1f}",
        f"9 个静默失败全部消除;PCIe {te['kivi_gate02']['fallback']['mib']/1024:.1f} GiB")
    add("RULER门控", "int8 门控 PCIe", f"{te['int8_gate5']['fallback']['mib']/1024:.1f} GiB",
        "安全方案流量比 fp8 少 3.4×(增量持久修复;选择性在规模上成立)")
    add("RULER门控", "增量修复流量降幅", "41×", "整请求重换页 bug 修复:fp8_gate02 1267→30.9 GiB")

    # ---- 机理与第二方法(T-F E1 / T-G) ----
    e1 = J("tf_e1_layers.json")
    n_loss = sum(1 for k, v in e1.items() if k.startswith("L") and v < e1["none"] - 1e-6)
    add("机理", "单层污染有损层数", f"{n_loss}/28",
        "kivi2 逐层单独污染,needle 1.5B;全部无损 ⟹ 无关键层,失败纯来自跨层累积", headline=True)
    add("机理", "全层污染 acc", f"{e1['all']:.3f}", "对照:单层全部 1.0")
    tg = J("tg_kvquant.json")["result"]
    add("机理", "kvquant4 覆盖率", f"{tg['coverage']:.5f}",
        "第二个真实方法复刻:needle 满分但逐步认证不可得,与 KIVI 互证")

    # ---- 重复性(3 不相交切片,450 唯一样本) ----
    rp = J("te_repeats.json")["agg"]
    add("重复性", "满血三切片均值", f"{rp['none']['avg_mean']:.1f}", "79.3/87.2/83.4;所有方向性结论三切片复现")
    add("重复性", "fp8裸跑三切片均值", f"{rp['fp8_raw']['avg_mean']:.1f}", "22.8/27.8/28.7 —— 灾难在每个切片复现")
    add("重复性", "门控距满血差距", "0.4", "fp8/kivi 门控三切片均值与满血差 ≤0.4;kivi 额外零分(+9~+12)全消")

    # ---- T-H 界形对照 ----
    th2 = J("th_gate_bounds.json")["rows"]
    add("界形对照", "det门 fp8 PCIe", f"{th2['det_tanh']['fp8']['pcie_gib']:.1f} GiB",
        "τ=0.99,同见证输入;e-form 同 τ 为 30.8 GiB ⟹ Tier-A 档 det-tanh 门便宜 2.5×")
    add("界形对照", "det门 int8 PCIe", f"{th2['det_tanh']['int8']['pcie_gib']:.1f} GiB",
        "e-form 同 τ 27.3 GiB(5.2×);两界形质量均恢复(79.3–80.1)。受控抖动器档 e-form 反超")

    # ---- SGLang serving ----
    sv = J("sglang_serving_matrix.json")["models"]
    for mn in ("1.5B", "7B"):
        v = sv[mn]
        f_, w, o = v["fp16"], v["witcert"], v["witcert_outlier"]
        add("serving", f"{mn} fp16 容量", f"{f_['capacity']:,}", "page_size=16,mem_frac 0.88,batch16×64tok")
        add("serving", f"{mn} WitCert 容量", f"{w['capacity']:,}", f"{w['capacity_ratio']:.2f}×")
        add("serving", f"{mn} 容量倍率", f"{w['capacity_ratio']:.2f}×", "同显存预算")
        add("serving", f"{mn} 吞吐倍率", f"{w['tp_ratio']:.2f}×", "3 次取中位数")
        add("serving", f"{mn} 覆盖率(无旁路)", f"{w['coverage']:.3f}", "生产遥测,抽样 stride 32")
        add("serving", f"{mn} 旁路容量", f"{o['capacity']:,}", f"{o['capacity_ratio']:.2f}×")
        add("serving", f"{mn} 旁路覆盖率", f"{o['coverage']:.3f}", "m=4 固定通道集")
        add("serving", f"{mn} 证书边际(带旁路)",
            f"{abs(v['cert_marginal_with_outlier'])*100:.1f}%", "")
    add("serving", "证书边际(1.5B 无旁路)", "6.4%", "短序列 decode;长序列微基准是 +0.35%")
    add("serving", "证书边际(7B 无旁路)", "16.2%", "同上")
    add("serving", "kernel 级证书成本", "+0.35%", "S=4096 微基准,量化存储路径")
    add("serving", "fp16 路径证书成本", "+11.9%", "SGLang decode kernel,存储未量化时")
    add("serving", "写路径 Triton 加速", "82×", "torch 版在 decode 循环里会主导时延(63.7→575.9 tok/s)")
    # 补丁行数走冻结产物(2026-08-06 发布纪律:integration/ 私有不随
    # 发布,行数以 integration_stats.json 为准 —— monorepo 里由
    # tools/freeze_integration_stats.py 现算冻结,公开仓直读产物)
    _ps = J("integration_stats.json")
    _pl = [_ps["lines"][f] for f in (
        "witcert_sglang.patch", "witcert_sglang_extend.patch",
        "witcert_sglang_serving.patch")]
    add("serving", "补丁行数", f"{_pl[0]} / {_pl[1]} / {_pl[2]}",
        "decode / extend / serving 接线,三份可提 PR;行数由 "
        "freeze_integration_stats 现算冻结进产物,不手抄")

    # ---- H200 战役(P0/P1,2026-07-29/30,8×H200 143GB)----
    p40 = J("p40_adaptive_stress.json")
    add("H200战役", "自适应违约(自由解码)", f"0/{p40['n_req_free']}",
        "Qwen2.5-7B,50 prompts×20 seeds,prefill 4096,decode 64,δ_req=1e-2", headline=True)
    add("H200战役", "自适应监测 cell 总数", f"{p40['n_cells_total']:,}",
        "每步×每层×每 q-head,free+forced 双模式;数值容差 1e-9,零容差残余全部 |边际|≤1.8e-13")
    add("H200战役", "自适应轨迹分歧率", f"{p40['diverged_frac_free']*100:.1f}%",
        f"free 组与 fp16 基线分歧,中位第 {p40['div_step_median']} 步;rule-of-three 上界 0.3%")
    add("H200战役", "最坏 TV/界比", f"{p40['max_ratio_B_free']:.3f}", "全部 informative cell 上取最大")
    p41 = J("p41_rc_full_system.json")
    add("H200战役", "RC 对照 Δppl", f"+{p41['lm_ctx8192']['dppl_rc_mean']:.4f}",
        "Llama-3.1-8B,ctx8192×8 配对窗;WitCert 同框 +%.4f" % p41['lm_ctx8192']['dppl_wc_mean'])
    add("H200战役", "RC 页入", f"{p41['ruler_4096']['rc_pagein_mb_per_req']:.0f} MB/请求",
        "RC 快路径常态促升页入(RULER-4096);WitCert 认证态为 0,且无 CPU 常驻副本")
    p43 = J("p43_packed_gate_e2e.json")
    add("H200战役", "packed 门控闭环质量", f"{p43['gate02_fixed_avg']:.2f}",
        f"RULER 6 难任务×25,τ=0.2;= packed 无门控 {p43['packed_avg']:.2f},fp16 {p43['fp16_avg']:.2f}",
        headline=True)
    _pd = p43["per_req_pagein_mib_dist"]
    add("H200战役", "门控逐请求页入 P50/P95", f"{_pd['P50']/1024:.2f} / {_pd['P95']/1024:.2f} GiB",
        "串行差分口径,150 请求;packed 修复非持久化固有成本")
    p44s = J("p44_serving_rows.json")["results"]
    _rt = [p44s[k]["kv_capacity"] / p44s[k[:-8] + "_fp16"]["kv_capacity"]
           for k in p44s if k.endswith("_witcert")]
    assert _rt and all(abs(r - _rt[0]) < 5e-4 for r in _rt), f"容量比跨配置不一致:{_rt}"
    add("H200战役", "容量比跨规模/并行不变", f"{_rt[0]:.3f}×",
        "7B/14B/32B(单卡)、70B(tp4)、7B(tp2/tp4) 全部 1.8824×;结构决定", headline=True,
        src="p44_serving_rows.json:kv_capacity witcert/fp16 三对现算,断言一致")
    add("H200战役", "70B tp4 容量", f"{p44s['70b_tp4_witcert']['kv_capacity']:,}",
        f"fp16 {p44s['70b_tp4_fp16']['kv_capacity']:,};吞吐比 {p44s['70b_tp4_witcert']['tok_per_s_median']/p44s['70b_tp4_fp16']['tok_per_s_median']:.2f}×")
    p45 = J("p45_native128k.json")
    _p45rows = p45["results"]
    _nd = {r["S"]: r["jkv"] for r in _p45rows if r["domain"] == "needle"}
    add("H200战役", "原生 128k 覆盖率(needle)",
        f"{_nd[8192]*100:.1f} / {_nd[32768]*100:.1f} / {_nd[131072]*100:.1f}%",
        "Llama-3.1-8B 原生窗口(无 YaRN)@8k/32k/128k,零违约——随长度平坦;"
        "1.5B+YaRN 档的下滑为 YaRN 外推特有", headline=True)
    p48 = J("p48_pareto.json")["curves"]
    add("H200战役", "前沿最不利工况点", f"{p48['witcert']['512']['output_throughput']/p48['fp16']['512']['output_throughput']:.2f}×",
        "c=512 短请求(1024in/256out):witcert %.0f vs fp16 %.0f tok/s;单流 0.78×,tp2 反超"
        % (p48['witcert']['512']['output_throughput'], p48['fp16']['512']['output_throughput']))
    p46 = J("p46_longctx_serving.json")["cells"]
    add("H200战役", "长上下文代价比例平稳", "1.84×→1.89×",
        "Llama-8B witcert TTFT 中位比 @8k→32k(TPOT 1.47×→1.49×),不随 S 恶化;"
        "qwen7b cert-on 30k TPOT 异常已列工程项")
    add("H200战役", "72B 覆盖率", "25.5–29.8%",
        "jkv@δ=1e-2 三域,零违约;显著低于 7–32B(54–81%),削弱性结果如实报告(与 Yi 同型)")

    # ---- R8 异构架构尺子(2026-07-30,论文2 平台篇;paper=2 不参与论文1 守卫)----
    def add2(n, v, scope):
        add("R8平台", n, v, scope, paper=2)

    p50 = J("p50_latent_v2lite.json")["schemes"]
    add2("latent 尺子 cells", f"{p50['latent-rtn8']['n']:,}",
         "V2-Lite S=4096,9 文档×3 域,7 层采样×头步 2(MLA latent 数学验证)")
    add2("latent int8 双 Tier 违约", f"{p50['latent-rtn8']['viol_A']}/{p50['latent-rtn8']['n']:,}",
         "Tier A 零违约;Tier A soundness 由此承担,在线观测只给量级与紧度")
    add2("latent dither Tier B 覆盖率",
         f"{p50['latent-dither8']['covB']['0.2']*100:.1f}%", "τ=0.2;τ=0.5 时 100%")
    p51 = J("p51_glm_serving_phase1.json")["summary"]
    add2("GLM 生产 fp8 相对残差", f"{p51['fp8_rel_pct']:.2f}%",
         "GLM-5.2 W4AFP8 生产栈在线观测,78 层×8 rank,相对 latent 范数")
    add2("GLM latent 层间异质", f"{p51['heterogeneity_ratio']:.0f}×",
         "latent 范数最大/最小之比,异构策略论据")
    p52 = J("p52_corruption_sentinel.json")["summary"]
    add2("坏块哨兵假阳性率", f"{p52['false_positive_rate']:.0%}",
         f"{p52['verify_passes_total']} 次基线校验零告警;注入坏块被逐槽位定位(论文2 引言动机实验)")
    p53 = J("p53_topk.json")
    add2("DSA top-1 认证率", f"{p53['cert_rate_r']['0.001/r1']*100:.2f}%",
         f"GLM-5.2,η=1e-3,{p53['n_layers']} 层×{p53['rows_total']:,} 行;逐层区间 90.0–100%")
    add2("DSA 严格集合认证率", f"{p53['cert_rate_k']['0.001']*100:.2f}%",
         "top-2048 集合口径过强(负结果);认证内翻转恒 0")
    p54 = J("p54_v4flash_quant.json")
    add2("V4 条目二次量化 int8 残差", f"{p54['aggregate']['rel_int8']['mean']*100:.3f}%",
         f"{p54['n_layers']}/43 层×{p54['effective_rows']:,} 行;与 GLM 生产 fp8 的 0.93% 同量级")
    add2("V4 带范数见证保守度", f"{p54['aggregate']['tight_int8']['mean']:.2f}×",
         "W/‖Δ‖,B=16,int8;逐层 2.73–3.86")
    p58 = J("p58_v4_topk.json")
    add2("V4 页级 top-1 认证率", f"{p58['cert_rate_r']['0.001/r1']*100:.2f}%",
         "η=1e-3,21 个 C4 索引层;V4 选压缩页,与 DSA 的 token 级量纲不可直接比")
    # p56 有两种产出口径(本仓库两条并行工作线),字段名不同,兼容读取
    p56 = J("p56_kda_contraction.json")
    _sm = p56.get("summary")
    if _sm:                                   # 双 rank 合并口径(commit 93afe28)
        _lo, _hi = _sm["a_mean_range"]
        _h = _sm["half_life_range_steps"]
        _n = _sm["near_one_frac_range"]
        add2("KDA 收缩因子 ā", f"{_lo:.3f}–{_hi:.3f}",
             f"Kimi-Linear-48B 代理,{_sm['n_kda_layers']}/20 KDA 层全部 ā<1;"
             f"半衰期 {_h[0]:.1f}–{_h[1]:.1f} 步;逐层标量近似口径")
        add2("KDA 近 1 通道占比", f"{_n[0]*100:.1f}–{_n[1]*100:.1f}%",
             "P(a_t≥0.999) 逐层区间;这些通道上界退化为线性累积 ⇒ 界须逐通道给,"
             "属紧度限制而非 soundness 问题")
    else:                                     # 单 rank 逐层口径
        add2("KDA 收缩因子 ā", f"{p56['a_mean']['median']:.4f}",
             f"{p56['n_kda_layers']}/20 KDA 层;半衰期中位 "
             f"{p56['half_life_steps']['median']:.1f} 步;逐层标量近似口径")
        add2("KDA 近 1 通道占比", f"{p56['p_ge_0999']['median']*100:.1f}%",
             f"P(a_t≥0.999) 中位,最大 {p56['p_ge_0999']['max']*100:.1f}%")
    if os.path.exists(os.path.join(OUT, "p52_sentinel_sla.json")):
        _sla = J("p52_sentinel_sla.json")
        _ad = _sla["all_detected"]
        add2("哨兵单块检出概率", f"{_ad['per_block_detect_prob']:.3f}",
             f"有效轮数 n≈{_ad['n_effective']}(注入 rank 注入后的校验轮数,非全 rank 总和),"
             f"每轮抽 128/1024;超几何闭式")
        add2("哨兵检出 2/4 的概率", f"{_ad['P_at_most_observed']:.3f}",
             "'至多检出 2 个'的概率;实测 2/4 属正常涨落,是抽样受限而非能力上限")
        _row1 = next(r for r in _sla["sla_table"]["rows"] if r["B"] == 1)
        _knob = " / ".join(str(_row1[f"r{rx}"]["0.01"]) for rx in (32, 64, 128, 256))
        add2("哨兵检出延迟旋钮", f"{_knob} 轮",
             "B=1、δ=1e-2 时 r=32/64/128/256 所需轮数;检出延迟与抽样预算近似成反比")
    if os.path.exists(os.path.join(OUT, "p55_v4flash_attribution.json")):
        p55 = J("p55_v4flash_attribution.json")
        _pa4 = p55["pool_path"]["pool_attn"]["block4"]
        _pa128 = p55["pool_path"]["pool_attn"]["block128"]
        add2("V4 池化粒度", f"4 与 128 并存({_pa4['n_layers']}/{_pa128['n_layers']} 层)",
             "同一模型内两种 hc 池化块大小;1/block 差 32 倍,跨层比较必须用归一化量")
        add2("V4 池化集中度(block4)", f"{_pa4['top1_norm_median']:.4f}",
             f"top1_norm 中位(0=完全均摊,1=完全主导);有效 token {_pa4['eff_tokens_median']:.2f}/4;"
             "经验观测,不称证书")
        _sel = p55["select_path"]
        if _sel.get("b_S"):
            add2("选择段契约 b_S", f"{_sel['b_S']['0.001']:.5f}",
                 "Lemma S2:(a_S,b_S)=(1,½(m_out+m_in));η=1e-3,V4 C4 索引器 21 层中位;"
                 "选择段**加性不放大**上游误差")
        # p55 的 end_to_end 是被度量类型检查废掉的那次组合(相对见证 + TV 直接相加,
        # 不属任何度量)—— 不再产出冻结数字;合法重建见 p76 块
        add2("V4 三路自检违反", "0",
             "见证违约 + 认证内翻转 + 池化不变量违反合计;存储侧 sound / 选择侧条件认证 / 池化侧无认证")
    if os.path.exists(os.path.join(OUT, "p68_adaptive_sampling.json")):
        _ad = J("p68_adaptive_sampling.json")["phases"]
        add2("自适应采样覆盖地板兜底", str(_ad["high_load_c32"]["forced_by_floor"]),
             "高负载阶段触发次数;0 表示 every 一直在安全区。覆盖两阶段均 43 层未塌 —— "
             "地板存在的意义是覆盖塌陷没有免费警报,不是它常被触发")
    if os.path.exists(os.path.join(OUT, "p66_concurrent_overhead.json")):
        _cc = J("p66_concurrent_overhead.json")
        _w = _cc["worst_case"]
        add2("并发下探针开销(EVERY=64)",
             f"吞吐 {_w['on_e64']['throughput_worst_pct']:+.1f}% / "
             f"TTFT P99 {_w['on_e64']['ttft_p99_worst_pct']:+.1f}%",
             "并发 1/8/32/64,每档 3 次重复取中位;**均在逐档噪声以内** —— "
             "只能给上界不能给点估计;关图口径")
        add2("并发下探针开销(EVERY=16)",
             f"吞吐 {_w['on_e16']['throughput_worst_pct']:+.1f}% / "
             f"TTFT P99 {_w['on_e16']['ttft_p99_worst_pct']:+.1f}%",
             "**双双超出噪声,是可主张的真实退化**;故常开生产档位取 EVERY=64 不取 16")
    if os.path.exists(os.path.join(OUT, "p71_graph_safe.json")):
        _gs = J("p71_graph_safe.json")
        _rt = _gs["replay_test"]
        add2("图重放期累加(graph-safe)",
             f"Δ元素 {_rt['delta_elements']} / Δn_calls {_rt['delta_n_calls']}",
             f"**带图运行**(未加 disable-cuda-graph)。CUDA graph 重放不执行 Python,"
             f"故 n_calls 只涨 prefill 的 {_rt['expected_n_calls_prefill_only']} 次,"
             f"而设备累加器涨满 {_rt['expected_decode_replays']} 次 decode 重放 + prefill;"
             f"两者与预期分毫不差,静置后不再变化。"
             f"**判据是累加器在重放期继续增长,不是带图下的开销** —— 开销数字仍是关图口径")
    if os.path.exists(os.path.join(OUT, "p72_graph_overhead.json")):
        _go = J("p72_graph_overhead.json")
        _w1 = _go["worst_case"]["e1"]
        add2("带图全速探针开销(EVERY=1)",
             f"吞吐 {_w1['throughput_worst_pct']:+.1f}% / "
             f"TTFT P99 {_w1['ttft_p99_worst_pct']:+.1f}%",
             "**带图口径**,单卡 Qwen2.5-7B + gqa-kv,并发 1/8/32 各 3 次重复取中位。"
             "EVERY=1 是唯一被验证过在重放期有完整覆盖的档位,故这是带图成本的**上界**。"
             "**与论文关图数字栈不同,不可相减**")
    if os.path.exists(os.path.join(OUT, "p71_graph_safe_e64.json")):
        _e64 = J("p71_graph_safe_e64.json")
        add2("带图下 EVERY=64 的重放增量", str(_e64["replay_test"]["delta_elements"]),
             f"跑完整轮负载后累加器增量为 {_e64['replay_test']['delta_elements']},"
             f"判定 {_e64['verdict']} —— **采样门是 capture 时求值一次的 Python 分支**,"
             "开图后 EVERY 的语义从'每 N 次采一次'变成'这张图采不采'。"
             "该档层覆盖仍显示 28/28(起服期陈旧样本),验收门原本会放行,"
             "故新增设备侧执行计数与 --expect-replay 才能拦住")
    if os.path.exists(os.path.join(OUT, "p73_capture_select.json")):
        _cs = J("p73_capture_select.json")
        _v = _cs["verdicts"]
        add2("带图最省档探针开销(1 层 × 单量化档)",
             f"吞吐 {_v['s28q1']['throughput_worst_pct']:+.2f}% / "
             f"TTFT P99 {_v['s28q1']['ttft_p99_worst_pct']:+.2f}%",
             "**带图口径**,capture 时按层号选择性插桩(确定性旋钮),**未用融合算子**。"
             "重放累加与层覆盖对账均通过 —— 不是'探针不在图里'的免费。"
             "仍超出 c=32 档的噪声下界,故仅靠减插桩点**压不进噪声**;"
             "把 kernel 数压到 1 之后的同覆盖档见'带图常开档'")
        _win = _v.get("f28q1", {})
        add2("带图常开档(融合算子,1/28 层)",
             f"吞吐 {_win.get('throughput_worst_pct', 0):+.2f}% / "
             f"TTFT P99 {_win.get('ttft_p99_worst_pct', 0):+.2f}%",
             "**三条判据同时满足**:重放累加为真、退化在逐档噪声下界内、层覆盖与声明一致。"
             "**成立的是'1 层常开'不是'全层常开'** —— 同为融合算子,全 28 层仍要 "
             f"{_v.get('fall', {}).get('throughput_worst_pct', 0):+.2f}%")
        add2("融合算子在全层覆盖下的收益",
             f"{-52.81 / _v.get('fall', {}).get('throughput_worst_pct', -1):.1f}×",
             "同为 28/28 层带图运行:torch 版 -52.81%,融合版 "
             f"{_v.get('fall', {}).get('throughput_worst_pct', 0):+.2f}%。"
             "**kernel 数是成本的驱动量**,这是把它从几十压到 1 的直接收益")
        add2("带图插桩成本的标度",
             f"{_v['s14']['throughput_worst_pct'] / _v['s14q1']['throughput_worst_pct']:.2f}×",
             "同为 2 层,量化档从 2 减到 1 的吞吐成本之比 —— "
             f"另:同为单量化档,层数 2->1 的比为 "
             f"{_v['s14q1']['throughput_worst_pct'] / _v['s28q1']['throughput_worst_pct']:.2f}×。"
             "**成本 ≈ 层数 × 每次执行的 kernel 数,两个旋钮各自近似线性且可乘**;"
             "故开图后的成本驱动量是 kernel 数,不是采样率")
    if os.path.exists(os.path.join(OUT, "p74_fused_equiv.json")):
        _fq = J("p74_fused_equiv.json")
        add2("融合见证算子的数值等价", f"{max(c['worst_rel_err'] for c in _fq['cases']):.1e}",
             "融合 Triton 版与 torch 版在四组形状上的最大相对误差(mean/min/max);"
             "**viol 计数与样本数完全相等** —— viol 是 soundness 自检,不容许近似。"
             "舍入取 away-from-zero 而非 torch 的银行家舍入,仅在精确 .5 处不同(测度为零)")
        add2("融合见证算子单调用耗时",
             f"{_fq['latency_us']['fused_1quant']:.1f} us",
             f"对 torch 单量化档 {_fq['speedup']['vs_torch_1quant']:.1f}×、"
             f"双量化档 {_fq['speedup']['vs_torch_2quant']:.1f}×;"
             "**第一版用 grid=(1,) 想避开原子操作,反而 346us 比 torch 还慢** —— "
             "拿并行度换原子操作是亏本买卖,改按行分块 + 三维分带归约才拿到这个数")
    if os.path.exists(os.path.join(OUT, "p78_full_composition.json")):
        _fc = J("p78_full_composition.json")
        _s = _fc["summary"]
        add2("全链逐层注意力 TV 界(typical)", f"{_s['tv_total_typ_median']:.3f}",
             f"存储(窗口+压缩页均见证)+ 经验桥换算的选择误差,均 attn_dist:tv;"
             f"c4 层({_s['n_c4_layers']} 层)中 {_s['n_nonvacuous_typ']} 层非空洞。"
             "**组合档位 empirical**(选择段经经验桥,最弱段决定);c128 层页侧未见证分开报")
        add2("经验桥系数 ρ", f"{_s['rho_median']:.2f}",
             f"注意力口径遗漏质量 / 选择器口径遗漏质量,截断集上实测;中位 {_s['rho_median']:.2f}"
             f" 接近 1(尾部 {_s['rho_max']:.2f},折算取 ρ_max 保守)。"
             "**选择器质量是注意力质量的可用代理** —— 外推到 η 扰动换手集是均质性假设")
        add2("选择误差项(注意力单位)", f"{_s['b_S_attn']:.5f}",
             f"ρ_max × b_S;empirical 档。**不是瓶颈** —— 主导是存储见证"
             f"(centry max {_s['wit_centry_max']:.3f} 比 swa 还大)")
        add2("C4 截断遗漏质量", f"{_s['trunc_mass_median']:.3f}",
             f"top-512 截断遗漏的假想全压缩注意力质量,中位(最大 {_s['trunc_mass_max']:.3f});"
             "**架构设计量而非误差** —— 稀疏选择是 V4 的模型语义,不计入误差链;"
             "本负载 decode 采样口径")
    if os.path.exists(os.path.join(OUT, "p80_request_ledger.json")):
        _rl = J("p80_request_ledger.json")
        _wp = _rl["working_point_coverage"]; _bp = _rl.get("working_point_bound")
        add2("请求级账本:保留优先工作点",
             f"保留 {100*_wp['entry_retention']:.1f}% / 每步 TV 界 {_wp['tv_bound_per_step_max']:.2f}",
             f"**离线重放原型,非在线闭环**;条目级确定性证书(W ≤ {_wp['W_thr']:.2f})+ "
             f"假想局部回退;n={_rl['n_entries_total']:,} 条目,c4 {_rl['n_layers']} 层。"
             "保留率=压缩收益保留,不称 certified coverage(其每步界空洞须一并说)。"
             "非空洞界与 ≥80% 保留不能同时成立(int8 代理见证无重尾)。Σδ=0;"
             "Lean: ledger_sound/telescope_sum/two_layer_budget_le/unknown_length_price")
        if _bp:
            add2("请求级账本:界优先工作点",
                 f"TV ≤ {_bp['tv_bound_per_step_max']:.2f} / 保留 {100*_bp['entry_retention']:.1f}%",
                 f"W ≤ {_bp['W_thr']:.2f} 的条目走压缩,其余逐条目精确读取(局部回退)。"
                 "构造性:被读条目全 ≤ 阈值 ⟹ 每步 TV 界成立;"
                 "探针为采样审计口径,生产化=逐读取检查(gating 补丁)")
    if os.path.exists(os.path.join(OUT, "p82_online_ledger.json")):
        _ol = J("p82_online_ledger.json")
        _os = _ol["summary"]
        add2("在线影子账本:逐读取事件数", f"{_os['events_total']:,}",
             f"{_ol['n_requests']} 条串行请求,逐请求账本在线建立;每 decode 步 × c4 层 × "
             f"全部被选中页条目(非抽样)。**只读影子**:超阈条目照常被使用,"
             "回退为假想(would_*);真实回退与概率预算消费是 A9 终局")
        add2("在线影子账本:逐请求保留率中位", f"{_os['retention_median']:.3f}",
             f"W_thr={_ol['W_thr']:.2f};min {_os['retention_min']:.3f} / "
             f"max {_os['retention_max']:.3f};假想页入中位 "
             f"{_os['would_pagein_mib_median']:.0f} MiB/请求 —— 这就是把回退做实的"
             "代价预告,与 p80 离线曲线一致")
    if os.path.exists(os.path.join(OUT, "p88_same_request.json")):
        _sr = J("p88_same_request.json")
        _ss = _sr["summary"]
        add2("贪心解码重启稳定性地板", _ss["stability_off_vs_off2"],
             "off vs off2(零干预、两次独立起服):n=8 中即有分歧 —— **一切输出级"
             "对比的噪声地板**。p83 的'策略保护输出'对比缺此地板,已由本口径修正:"
             "干预效应真实(all 臂重度分歧),但 policy/dither 与基线的差异在 n=8 下"
             "处于地板附近,不可分辨;不称 protective,称 consistent with")
        add2("写策略接受率(精度粗化)", f"{100*_ss['policy_acceptance_rate']:.1f}%",
             "**掩码条目占比/策略接受率 —— 非物理压缩收益**(字节数不变,是值域粗化;"
             "物理收益需打包另案)。窗口 nope 侧")
        add2("抽签前授权:预授权率与保守代价",
             f"{100*_ss['dither_preauth_rate']:.1f}% / {_ss['preauth_conservatism']:.1f}×",
             "授权只看先验半径 u_e(均值项+尾项),**不看随机实现**;保守代价 = "
             "policy 事后接受率 / dither 预授权率 —— 先验尾项的价格,如实报")
        _av = _ss["auth_violations_observed"].split("/")
        add2("预授权违约审计", f"{_av[0]}/{int(_av[1]):,}",
             "事件 E_i='授权条目实现 W 超预授半径' 的运行时验证:全部预授条目实测"
             "违约计数;δ_i 自此**绑定真实概率事件**(五审 P0-1 修复),"
             "假设 R3′ 逐事件计数留痕")
        add2("同请求闭环 Σδ", f"{_ss['delta_spent_dither']:.7f}",
             "读侧逐请求账本与写侧真实策略**同一次运行**;逐请求快照差分出写侧逐请求"
             "账目(每请求一行:读侧检查数/写侧压缩与回退增量/δ 增量/违约);"
             "Σδ < δ_req=0.01(望远镜),每份 δ_i 绑定事件")
    if os.path.exists(os.path.join(OUT, "p86_full43_composition.json")):
        _f43 = J("p86_full43_composition.json")
        _s43 = _f43["summary"]
        add2("全 43 层存储项结构落账",
             f"dense 2 / c4 {_f43['families']['c4']} / c128 {_f43['families']['c128']}",
             "41 个有页层的页侧全部见证(写后读回口径);L0/L1 为 ratio=0 稠密层,"
             "无压缩页是架构事实非测量省略。c128 无索引器稠密读,无选择项")
        add2("c128 层逐层 TV 界(worst)", f"{_s43['tv_worst_median_c128']:.3f}",
             f"中位;c4 侧为 {_s43['tv_worst_median_c4']:.3f}(含经验桥选择项)。"
             f"c128 页见证 max {_s43['w_cw_max_c128']:.3f} 比 c4({_s43['w_cw_max_c4']:.3f})"
             "干净约一倍 —— 粗压缩层反而更好认证;43 层中 "
             f"{_s43['n_nonvacuous_worst_43']} 层 worst 界 < 1")
    if os.path.exists(os.path.join(OUT, "p96_identity_v2.json")):
        _iv = J("p96_identity_v2.json")
        _vs = _iv["summary"]
        add2("真实生命周期请求身份:user 请求独立预算 max Σδ",
             f"{_vs['max_delta_spent']:.6f}",
             f"七审 F2 重做:边界 = 写路径 extend 转换(先于任何写入,无一批偏移),"
             f"UID 永不复用;{_vs['n_user_requests']} 个 user 请求账户(warmup/boot "
             f"{_vs['n_warmup_accounts']} 个分开计)全部独立 ≤ δ_req=0.01;"
             "单元测试 R5 覆盖复用+更长初始长度等漏检情形。串行 batch=1 口径")
        add2("SR 随机流:跨请求零复用",
             "(nonce,UID,layer,k)",
             "七审关键修复:种子四元组含永不复用的请求 UID —— 不同请求不共享随机数,"
             "条件随机性对自适应流量成立(Lean 超鞅前提可实例化);四元组跨 rank 一致,"
             f"8-rank 账目{'逐字段一致' if _vs['replica_coupling_identical'] else '不一致'}(耦合保持)")
        _eg = _vs["eprocess_global"]
        add2("全局 e-process(跨请求不重置)",
             f"峰值 {_eg['log_M_max']:.2f} vs 阈值 {_eg['threshold']:.2f}",
             f"持续全服务漂移哨兵:{_eg['n_factors']:,} 因子跨全部请求单调累积"
             f"(终值 {_eg['log_M_final']:.1f}),越阈 {'0 次' if not _eg['crossed'] else '发生'};"
             "log_M_max 仅含更新后值(不以初始化 0 冒充峰值)。κ=0.5 预注册,δ_e=1%")
    if os.path.exists(os.path.join(OUT, "p97_owner_closure.json")):
        _oc = J("p97_owner_closure.json")
        _os2 = _oc["summary"]
        add2("owner 版本化:复用压力下闭环",
             f"复用 {_os2['slot_reuses_total']:,} 次 / 读未写 {_os2['read_unwritten_total']} 次",
             "G2:认证写逐槽位记录写入者 UID,复用 = 写时 owner 变更(压缩池无独立 "
             "free 路径,复用必先重写,flags 恒新鲜 —— 设计即约束,无需 allocator "
             "hook);不同 prompt 各账户 1,000+ 次复用下 foreign=0,自有闭环精确。"
             "口径:串行页回收 + 前缀共享;**请求中途驱逐**负载未覆盖")
        add2("foreign 读归属(前缀共享)",
             f"{_os2['foreign_reads_total']:,}(重复 prompt 占 {_os2['foreign_reads_dup_prompt']:,})",
             "读到别的请求写入的条目按 owner 分账 —— 重复 prompt 94% 读命中前缀缓存"
             "(owner=首写请求),不再误记入本请求闭环;写侧 δ 由 owner 请求承担,"
             "读侧检查确定性 δ=0,跨请求共享不破坏预算语义")
    if os.path.exists(os.path.join(OUT, "p98_concurrent_identity.json")):
        _ci = J("p98_concurrent_identity.json")
        _cs = _ci["summary"]
        add2("并发逐行身份:8 路并发独立预算",
             f"max Σδ = {_cs['max_delta_spent']:.6f}",
             f"G1:批内多请求混行下逐条目 UID 归属(decode 计划 [bs,16] 批行对齐 / "
             f"prefill 逐 query token 前缀和分段);{_cs['n_user_requests']} 个 user 账户"
             "各自独立 ≤ δ_req=0.01;每账户独立 (nonce,uid,layer,k) 随机流;"
             f"8-rank 耦合{'逐字段一致' if _cs['replica_coupling_identical'] else '不一致'}")
        add2("并发归属守卫:foreign 读",
             f"{_cs['foreign_reads_user']}/{_cs['read_events_user']:,}",
             "owner 分账做成归属正确性的**运行时守卫**:首验 foreign 3.4% 触发,"
             "定位出 c4/c128 两池共用键空间的 owner 污染(串行同刻单 uid 掩盖,"
             "并发暴露 —— 最早分配者低号槽位与 c128 数值区重叠);键按压缩比分离后"
             "8,228,864 次读 foreign=0、读未写=0")
    if os.path.exists(os.path.join(OUT, "p99_concurrent_identity.json")):
        _hz = J("p99_concurrent_identity.json")
        _hg = _hz.get("gate", {})
        add2("硬化验收门(machine-decidable)",
             f"{sum(_hg.get('criteria', {}).values())}/{len(_hg.get('criteria', {}))} PASS",
             "八/九审收口:**全零硬判据**(foreign/unwritten/violations/归属回退/"
             "身份 fail-closed 拦截/e-process 有数且未越阈/独立预算/账户数/8 份 rank 快照齐且耦合),"
             "判读脚本退出码承载 —— CI/发布包可直接依赖;p99 为 fail-closed 身份+"
             "违约恢复+generation 计数全部在线后的端到端复验(拦截 0/恢复 0 = "
             "硬化机制在正常流量下惰性,只在异常时接管)")
    if os.path.exists(os.path.join(OUT, "q1_packed_int4.json")):
        _q1 = J("q1_packed_int4.json")
        add2("低比特案例:packed INT4 独立池字节(GPU 实测)",
             f"{_q1['pool_bytes_gpu']['fp8_pool_MiB']:.1f}→{_q1['pool_bytes_gpu']['int4_pool_MiB']:.1f} MiB",
             "**独立池格式收益,非服务进程总显存**(FP8 原池仍是 FlashMLA 真实输入);"
             "64Ki token,memory_allocated 差分 −38.4%")
    if os.path.exists(os.path.join(OUT, "p105_wc_q2e.json.rank0")):
        _p5 = J("p105_wc_q2e.json.rank0")
        _pc = _p5.get("packshadow_check") or {}
        add2("低比特案例:serving 内 pack↔unpack 自洽",
             f"{_pc.get('n_checked',0):,} 检查 / unwritten {_pc.get('n_unwritten',0)} / rel {100*_pc.get('rel_mean',0):.1f}%",
             "真实 360B/token packed 池在 serving 内三跑稳定;工具证伪两条设计假设:"
             "①槽位渐进演化(只改 pack 时机误差 18×)②python 读路径非真实消费者"
             "(FlashMLA kernel 内直吃 FP8,三探 packedread=null)。"
             "**不声称**:INT4 已被 attention 消费/HBM 已降/吞吐/质量;"
             "rel_max≈4.9 归因演化期重读(近零范数假设被数据否决)")
    if os.path.exists(os.path.join(OUT, "p90_cert_adjudication.json")):
        _aj = J("p90_cert_adjudication.json")
        add2("四证书裁决:非空洞 43 层",
             " / ".join(f"{k} {v}" for k, v in _aj["nonvacuous_43"].items()),
             "认证对象=掩 2 位 SR(θ/Δ 由捕获字节离线重建);中心化 tanh(osc/4) 使全部"
             "43 层非空洞(现行形 0/43);soundness 自检:全部半径零被击穿"
             "(顶部夹取的确定性平移与零权重分支 log(0) 两个 bug 均被自检抓出后修复)")
        add2("四证书裁决:tightness(半径/实测中位)",
             " / ".join(f"{k} {v:.1f}×" for k, v in _aj["tightness_median_ratio"].items()),
             "64 次 SR 实现为地面真值;现行形 15.8× 松 → 中心化/精确 MGF 2.0×(8 倍收紧)。"
             "**门1(≥30/43)过;门2(retention@0.2 ≥1.5×)败** —— 全族 retention=0,"
             "τ=0.2 在掩 2 位粒度不可达;按预注册规则 E4 重 kernel 跳过。"
             "部署分野:写时预授权 x 在手 θ 免费(精确 MGF 零元数据);读时需 θ ≈+39% 存储")
    if os.path.exists(os.path.join(OUT, "wc_e2.json.rank0")):
        _e2 = J("wc_e2.json.rank0")["certified_write"]
        _n2 = _e2["n_compressed"] + _e2["n_fallback_exact"]
        add2("SR 精确半径:预授权率", f"{100*_e2['n_compressed']/max(1,_n2):.1f}%",
             "真随机舍入(条件均值恰零)+ **variance-aware two-point bounded-difference "
             "radius**(精确的是两点方差 θ(1−θ)Δ² 与夹取项;整体半径含 Jensen 与 "
             "Hoeffding 放缩,该组合尚未 Lean 化 → F3);同阈值 7.8%→67.5%(8.7×)")
        add2("SR 预授权违约审计", f"{_e2['n_auth_violations']}/{_e2['n_authorized']:,}",
             "全部预授条目实测违约计数;δ_i 绑定事件'授权条目实现 W 超预授半径'")
    if os.path.exists(os.path.join(OUT, "p92_eprocess_compare.json")):
        _ep = J("p92_eprocess_compare.json")
        add2("e-process 累计收紧", f"{_ep['ratio']:.0f}×",
             f"T={_ep['T']:,} 事件,**双侧**累计对象(公平口径,各付 δ/2):"
             f"{_ep['eprocess_anytime_radius_two_sided']:.0f}σ vs 望远镜求和 "
             f"{_ep['telescoping_cumulative_radius']:,.0f}σ。**失败事件不同**:"
             "望远镜界'任一逐事件半径被突破'(条目授权用),e-process 界有符号累计和 —— "
             "后者是累计误差的可形式化路线,**尚未接入逐条目授权账本**(F4)")
    if os.path.exists(os.path.join(OUT, "p93_e4_overhead.json")):
        _e4 = J("p93_e4_overhead.json")
        add2("认证写路径开销(未融合)", f"{_e4['overhead_ratio_median']:.2f}×",
             f"逐请求墙钟中位 off {_e4['arms']['off']['wall_median']:.1f}s → "
             f"dither {_e4['arms']['dither']['wall_median']:.1f}s;关图 + python 路径口径 —— "
             "融合 kernel 被 E1 门2 按预注册规则挡下,此为**未优化实现的诚实成本**;"
             "物理字节不变(精度粗化非物理压缩)")
    if os.path.exists(os.path.join(OUT, "p65_kda_logsum.json")):
        _k = J("p65_kda_logsum.json")
        add2("KDA 对数收缩 E[log a]", f"{_k['E_log_a']['median']:.4f}",
             f"正确判据(乘积口径)的逐层中位;均值口径 log(ā) 为 "
             f"{_k['log_of_mean']['median']:.4f} —— **均值把收缩算弱了 2.6×**;"
             f"全 {_k['n_layers']} 层 E[log a] 均为负")
        add2("KDA 误差半衰期", f"{_k['half_life_steps']['logsum_median']:.2f} 步",
             f"对数和判据;均值口径会报成 {_k['half_life_steps']['mean_median']:.2f} 步。"
             "口径:E[log a] 是 (通道,时刻) 聚合,**不是逐通道 Σlog a_t** —— "
             f"近 1 通道占比中位 {_k['p_near_one']['median']:.4f},其上不收缩")
    # p64 已被度量类型检查判废(superseded):它把相对见证与 TV 直接相加,
    # 得到的量不属于任何度量。**superseded 的产物不再产出冻结数字** —— 死数字
    # 留在 canon 里就是给未来的自己埋引用陷阱;其替代是 p76(桥接后的合法组合)。
    if os.path.exists(os.path.join(OUT, "p76_bridged_composition.json")):
        _c = J("p76_bridged_composition.json")
        _s = _c["summary"]
        add2("桥接后逐层注意力 TV 界(typical)", f"{_s['tv_typ_median']:.3f}",
             f"**窗口存储项**,精确形 ½(e^{{2ε}}−1),{_s['n_nonvacuous_typ']}/{_s['n_layers']} 层非空洞;"
             "ε = scale·‖q‖·W(打分桥 Cauchy–Schwarz × softmax 桥 e-form 实例化),"
             "q 范数与 W 均为采样分布统计量。**压缩页条目量化项未见证;"
             "选择段 b_S 在 selector_dist:tv,不相加**。每段 proof 指向已证 Lean 定理")
        add2("桥接后逐层注意力 TV 界(worst)", f"{_s['tv_worst_median']:.3f}",
             f"全 rank 逐层最大值口径,{_s['n_nonvacuous_worst']}/{_s['n_layers']} 层非空洞;"
             "Cauchy–Schwarz 取 Δk 与 q 对齐的最坏情形 —— sound 但悲观,这是证书的代价")
        add2("桥接系数 scale·max‖q‖", f"{_s['qn_max_all']:.4f}",
             "q 过 rmsnorm 后 scale·‖q‖ ≈ 1,打分桥几乎不放大;界的主导项是绝对见证 W"
             f"(全层最大 {_s['wit_max_all']:.3f})。p75 实采,43 层 × 8 rank")
        add2("请求级 TV 聚合预算", f"{_s['request_budget_sum_worst']:.1f}",
             f"{_s['n_layers']} 层求和(worst 口径),>> 1 —— **空洞,如实报**。"
             "跨层求和是聚合预算而非端到端输出距离(各层是不同的分布);"
             "被类型检查废掉的 0.982 连界都不是,不能拿本数字与它比较")
    if os.path.exists(os.path.join(OUT, "p61_platform_matrix.json")):
        pm = J("p61_platform_matrix.json")
        add2("平台覆盖:过验收门的探针行",
             f"{pm['n_probe_rows_verified']} 行 / {pm['n_unique_models']} 模型",
             "架构族 %s;适配器 %d/注入点 %d;收录门槛=通过覆盖->量纲->soundness 验收门"
             % ("、".join(pm["families_covered"]), pm["adapters"], pm["injection_points"]))
        gl = next((r for r in pm["matrix"] if r["path"] == "mla"), None)
        if gl:
            add2("GLM latent 二次量化 int8 残差", f"{gl['rel_int8_median']*100:.3f}%",
                 f"统一框架,{gl['layers_covered']}/{gl['layers_expected']} 层,"
                 f"采样 EVERY={gl['sample_every']};保守度中位 {gl['tight_int8_median']:.2f}×")
    if os.path.exists(os.path.join(OUT, "p60_probe_overhead.json")):
        p60 = J("p60_probe_overhead.json")
        worst = max(max(d.values()) for d in p60["overhead_pct"].values())
        add2("探针开销(最不利档)", f"{worst:+.2f}%",
             p60["caliber"])

    # ---- 代际失效因果链 + 生产集成(2026-08-02,case study 收口;paper=2)----
    try:
        b118 = J("p118_ruler_base_off.json")
        pk_on = J("p118_ruler_packed_on.json")
        pk_off = J("p118_ruler_packed_off.json")
        add2("matched 五臂:radix 掩蔽",
             f"{pk_on['acc']:.3f} vs {pk_off['acc']:.3f}",
             f"packed 臂 radix on/off;base 两侧均 {b118['acc']:.3f} —— 0.800 是"
             "缓存命中口径非质量,confounding 由同 launcher 五臂排除")
        p119o = J("p119_freshness_packonce.json")
        p119e = J("p119_freshness_everyread.json")
        add2("freshness 判别:pack-once vs every-read",
             f"{p119o['acc']:.3f} vs {p119e['acc']:.3f}",
             "20 篇 niah,两臂仅差 PACK_EVERY_READ 且读写双路径均有执行计数;"
             "everyread 恢复 ⇒ 代际一致性是主因(诊断配置,不作唯一归因)")
        q5 = J("q5_pack_roundtrip.json")
        add2("生产 pack 路径本征误差",
             f"{q5['B_pool']['mean']:.4f}",
             f"真实 c4 条目 {q5['n_rows']:,} 行/{q5['n_layers']} 层,纯核/池路径/"
             f"覆盖写三档同值,max {q5['B_pool']['max']:.4f},rope 段恰 0")
        v124 = J("p124_verdict.json")
        on124 = J("p124c_conc_inval_on.json")
        off124x = J("p124_conc_inval_off.json")
        add2("代际失效因果确认(第二次)",
             v124["verdict"],
             f"六条机器判据全过:inval_on acc={on124['acc']:.3f} / "
             f"inval_off acc={off124x['acc']:.3f}(并发 4 口径,两臂仅差失效开关,"
             "n_invalidate 双向证据);不覆盖 abort/radix-on/HiSparse/MTP/PD")
        dc4 = on124["snapshot"]["decode_shadow_all_ranks"]["c4"]
        off124 = J("p124_conc_inval_off.json")
        dc4off = off124["snapshot"]["decode_shadow_all_ranks"]["c4"]
        add2("解码侧影子:修复后 c4 保真度",
             f"{dc4['rel_mean']:.4f}",
             f"内核即将读到的 packed vs 此刻 FP8 源(重填前,1/32 采样,n="
             f"{dc4['n_checked']:,});本征 {q5['B_pool']['mean']:.4f},对照臂 "
             f"{dc4off['rel_mean']:.4f}({dc4off['rel_mean']/dc4['rel_mean']:.0f}×)"
             " —— centry 站点同指标对本失效零区分度(双写者构造性跨代比较)")
        a2 = J("p112a2_swapA.json")
        add2("阶段 A 池装配(规范寻址)",
             f"acc {a2['acc']:.3f}",
             "c128_kv_pool→SwappedC128Pool,20 层 remap(rank 快照 installed=1),"
             "与 noswap 臂 acc/解码侧 rel 一致 —— 结构性换池零质量代价")
        bf = J("p112b2_fp8.json"); bs = J("p112b2_stageb.json")
        dh = sum(bf["hbm_mib_per_gpu"]) - sum(bs["hbm_mib_per_gpu"])
        est_mib = 398478880 * 8 / 1048576
        add2("stage-B HBM 实测差分",
             f"{dh:,} MiB",
             f"FP8 全量 c128 池→窗口环(ring_pages=257)+ packed 常驻,8 卡 "
             f"aggregate 总差;核算 {est_mib:,.0f} MiB,aggregate 相差 "
             f"{abs(1 - dh / est_mib) * 100:.1f}%(逐卡 [566,346×7] 非均匀,"
             "有误差抵消,不得称逐卡吻合);c128 仅 ~0.3% 总 HBM(c4 是 32× "
             "大头),机制验证口径")
        add2("stage-B bring-up 质量(负结果)",
             f"{bs['acc']:.3f} vs {bf['acc']:.3f}",
             "6 篇 niah miss 3;HTTP n_error=0(产物 snapshot 空,**不能**声明"
             "内部 err/orphan/8rank installed —— 证据链洞,GPT review);后由 "
             "p112d 定谳为 decode 越界读环(寻址缺陷),环链是否无损未证明")
    except FileNotFoundError as e:
        print("[canon] 代际链产物缺失,跳过该组:", e)

    # ---- Q4/Q5/Q6 服务化收口(2026-08-03;paper=2)----
    try:
        q5r = J("q5e_refcons.json")
        rs5 = q5r["snapshot"]["pool_swap"]["ring_stats"]
        q5o = J("q5e_oracle.json")
        add2("参考消费者:packed 服务内消费",
             f"{rs5['n_ref_packed']:,} 行",
             f"三分类恒等式 {rs5['n_ref_resident']:,}+{rs5['n_ref_packed']:,}"
             f"+{rs5['n_ref_failed']}+{rs5['n_ref_skip_current']:,}="
             f"{rs5['n_ref_slots_total']:,}(rank0 聚合,逐调用去重槽);"
             f"oracle/refcons acc {q5o['acc']:.3f}/{q5r['acc']:.3f},tol=0;"
             "SMOKE 6 docs,环 129 页,覆盖随驱逐失效语义;不称无损")
        s1 = J("p112_serving_hbm.json")
        bt = s1["byte_truth"]
        add2("c128 单层 staging HBM 净差",
             f"{bt['measured_median_bytes']/1048576:.0f} MiB/GPU",
             f"实测中位 vs 字节净差 {bt['net_expected_bytes']/1048576:.1f} MiB"
             f"(freed {bt['staging1_freed_bytes']/1048576:.1f} − ring+packed "
             f"{(bt['ring_bytes']+bt['packed_bytes'])/1048576:.1f});全部 "
             "untyped_storage 直读;c128 仅 ~0.3% 总 HBM,机制口径")
        q4n = J("q4e_native.json"); q4s = J("q4e_c4staging.json")
        d4 = sorted(a - b for a, b in
                    zip(q4n["hbm_mib_per_gpu"], q4s["hbm_mib_per_gpu"]))
        sw4 = q4s["snapshot"]["pool_swap_c4"]
        net4 = (sw4["staging1_freed_bytes"] - sw4["ring_bytes"]
                - sw4["packed_bytes"]) / 1048576
        add2("c4 INT6 替换 HBM 净差",
             f"{d4[len(d4)//2]:,} MiB/GPU",
             f"native vs c4-staging1(formal 九门含 abort 扰动),字节净差 "
             f"{net4:,.0f} MiB(freed {sw4['staging1_freed_bytes']/2**30:.2f} "
             f"GiB=20/21 层,张量直读);双臂 acc {q4n['acc']:.3f}/"
             f"{q4s['acc']:.3f};SMOKE 探针配置(双图关),c128 保持原生 —— "
             "单独口径,不与 c128/图模式数字混算")
        g2 = J("q6g2_graphb2.json"); g2n = J("q6g2_native.json")
        pk6 = g2["snapshot"]["packed_kernel"]
        add2("packed kernel 包装器调用(双图开启)",
             f"{pk6['n_packed']:,} 次",
             f"**capture/eager 路径计数口径**(replay 执行录制内核,不回 "
             "host 计数;replay 图含 packed 分支由'capture 时所走分支'推得)"
             f":c128 INT4 {pk6['n_packed_c128']:,} + c4 "
             f"INT6 {pk6['n_packed_c4']:,};双臂 acc {g2['acc']:.3f}/"
             f"{g2n['acc']:.3f},negflag/uncovered=0,8 rank;fallback "
             f"{pk6['n_fallback']} 已归因无压缩缓存调用;SMOKE 单 seed 受限配置")
        k6 = J("q6k_graphb2.invalid.json"); k6n = J("q6c_native.json")
        add2("radix×stage-B 质量退化(复现)",
             f"{k6['acc']:.3f} vs {k6n['acc']:.3f}",
             "q6c 与 q6k(全链口径)两轮独立复现;radix-off 装配门显式拒绝"
             "该组合;吞吐坍缩为日志观察无独立 bench 产物,不作性能结论")
        pd1 = J("pd1_native.json"); pd2 = J("pd2_stageb.invalid.json")
        add2("PD 分离:native 通过 / stage-B 布局缺口",
             f"{pd1['acc']:.3f} / {pd2['acc']:.3f}",
             "单机双进程(prefill TP4+decode TP4+mini-lb,mooncake);native "
             "针跨进程答对=KV 真传输;stage-B 0/6 且 n_error=0 —— 传输完成、"
             "内核被调、服务不崩但语义错:布局契约未被传输层承载(runtime "
             "contract 必要性的实证);TP4×2 口径")
        h6a = J("q6h_rep1.json"); h6b = J("q6h_rep2.json")
        td = sum(1 for x, y in zip(h6a["items"], h6b["items"])
                 if x.get("out") != y.get("out"))
        add2("重复运行答案稳定性",
             f"acc {h6a['acc']:.3f}/{h6b['acc']:.3f}",
             f"同配置串行重复对(带路由),逐题文本 {td}/{len(h6a['items'])} "
             "不同(上游非确定推理;dsv4 不在 deterministic 白名单)——"
             "只主张答案稳定性,不主张逐字确定性")
        lat = J("p113_serving_latency.json")
        r = lat["ratio_staging1_over_native"]
        add2("staging1 服务时延比值",
             f"吞吐×{r['output_throughput_tok_s']:.3f} 成本×"
             f"{r['cost_gpu_s_per_output_token']:.3f}",
             "bench_serving random seed42 N=32;探针配置非生产;staging1 为"
             "逐调用全量物化未优化实现 = 上界开销;仅臂间可比,**不构成**"
             "token 成本降低声明")
        fu = J("p115_fused_unpack.json")
        rf = fu["ratio_fusedk_over_refcons"]
        add2("fused 解包核服务比值(含脚手架)",
             f"吞吐×{rf['output_throughput_tok_s']:.3f} 成本×"
             f"{rf['cost_gpu_s_per_output_token']:.3f}",
             "fusedk 臂含逐调用 ensure_fresh+Python 逐槽审计(非共模诊断"
             "脚手架)—— 比值是上界**不是纯核开销**;功能面独立成立"
             "(597,981 槽审计 stale=0,质量锚 1.000)")
        n12 = J("q6n_graphb2.json")
        m42 = J("q6m42_graphb2.json"); m43 = J("q6m43_graphb2.json")
        add2("图模式 12-doc 探针(负结果)",
             f"{n12['acc']:.3f}",
             f"无 MTP、12 docs、WORKERS=4:doc 稳定漏针(15@0.5/16@0.85/"
             f"18@0.5);MTP 双 seed 同树 {m42['acc']:.3f}/{m43['acc']:.3f}"
             "(漏针集与之高度重叠)—— 6-doc SMOKE 口径掩盖的系统性残差,"
             "cudagraph 图内消费声明由此限定为 6-doc;十四轮判别链定谳:"
             "预填压缩上下文读为必要通道且依赖 c4 环驻留(见判别链条目)")
        # ---- Q11 九轮判别链(2026-08-04):驱逐介导 + 分池双解离 + 写链无罪 ----
        o12 = J("q6o_graphb2.json")
        p3a = J("q6p3_ring1025.json"); p3b = J("q6p3_ring513.json")
        p5a = J("q6p5_c128small.json"); p5b = J("q6p5_c4small.json")
        p6a = J("q6p6_c4evict.json")
        fr = p6a["snapshot"]["pool_swap"]["ring_stats"]["n_forensic_rows"]
        fr4 = p6a["snapshot"]["pool_swap_c4"]["ring_stats"]["n_forensic_rows"]
        add2("判别:零驱逐环下 W4 并发全对",
             f"{p3a['acc']:.3f}",
             f"ring1025(n_evict=0)12-doc W4 = {p3a['acc']:.3f};同轮对照 "
             f"ring513 = {p3b['acc']:.3f}(n_evict=256)—— 腐坏驱逐介导;"
             f"串行 W1(q6o)= {o12['acc']:.3f} 但 n_evict=0,'串行干净'是"
             "混杂变量(未曾行使驱逐),真不变量 = 在飞请求页被驱逐 ⇔ 腐坏")
        add2("判别:分池双解离定位 c4",
             f"{p5a['acc']:.3f} / {p5b['acc']:.3f}",
             f"c128 独驱(256 次)acc={p5a['acc']:.3f} 全对 vs c4 独驱"
             f"(353 次)acc={p5b['acc']:.3f} —— c4 池驱逐特异;c128 驱逐"
             "无害此后又两轮复现(q6p6/q6p7 均 1.000)")
        add2("判别:eager 写链取证无罪",
             f"{fr + fr4:,}",
             f"烘定-存储逐行对账 {fr:,}(c128)+{fr4:,}(c4)行,mismatch=0,"
             "形状漂移=0 —— build→forward 间环槽从未易主;图内 finalize 撞"
             "已释放页计数亦为 0;fallback 调用全为无压缩缓存口径"
             "(extra_k_cache=None),与 c 池无关")
        p14a = J("q6p14_ablate.json"); p14c = J("q6p14_control.json")
        add2("判别:预填 extra 必要性消融(收口轮)",
             f"{p14a['acc']:.3f} / {p14c['acc']:.3f}",
             f"extra_topk_length 清零(执行自证 5,125 次)acc={p14a['acc']:.3f}"
             f" 全灭 vs 对照 {p14c['acc']:.3f}(c4 剂量带内)—— 预填压缩"
             "上下文读是任务必要通道;叠加坐标域错乱(kv-cache 坐标索引环"
             "缓冲,c4 越界 66 倍)与 14 轮驱逐剂量不变量,机制闭环:该通道"
             "数据依赖 c4 环驻留。修法=预填 extra 物化适配层(roadmap);"
             "短期口径=环≥活工作集(唯一已证干净配置)")
        q2g = J("q6q2_graphb2.json"); q3g = J("q6q3_graphb2.json")
        q2n = J("q6q2_native.json"); q3n = J("q6q3_native.json")
        add2("MTP 零驱逐口径重测(改判 PASS)",
             f"{q2g['acc']:.3f}/{q2n['acc']:.3f} 与 {q3g['acc']:.3f}/{q3n['acc']:.3f}",
             "EAGLE 1步,12-doc W4,双环 1025(quality_caliber 零驱逐自证):"
             "seed42/43 双臂全对 —— 每轮为同轮双臂受控对照,无跨轮归因需求;"
             "历史残差(q6i3 5/6、q6m 14/24)由 14 轮判别链归因为 c4 驱逐"
             "机制混杂,非 MTP 特异。矩阵 mtp 改判 PASS")
        qa_n = J("q10a_native.json"); qa_g = J("q10a_graphb2.json")
        qa_nb = J("q10a_native_bench.json"); qa_gb = J("q10a_graphb2_bench.json")
        add2("Q10 终表数据轮(零驱逐口径,未优化物化路径)",
             f"{qa_gb['output_throughput']:.1f} vs {qa_nb['output_throughput']:.1f} tok/s",
             f"bench(32 请求,2048/128):output 吞吐 ×"
             f"{qa_gb['output_throughput']/qa_nb['output_throughput']:.3f},"
             f"TTFT mean {qa_gb['mean_ttft_ms']/1000:.1f}s vs "
             f"{qa_nb['mean_ttft_ms']/1000:.1f}s;HBM ~"
             f"{max(qa_g['hbm_mib_per_gpu'])/1024:.1f} vs ~"
             f"{max(qa_n['hbm_mib_per_gpu'])/1024:.1f} GiB/GPU(大环口径反超,"
             "节省被环撑大抹掉)—— 真实代价口径,收益主张不作。**质量锚 "
             f"{qa_g['acc']:.3f}(11/12):零驱逐反例** —— bench 前置使逻辑"
             "槽复用,环页仍驻留即喂前任请求行(needle_0 输出他请求暗号),"
             "'零驱逐⇒干净'必要非充分;引用本表必须带此口径")
        qa_m = J("q11a_matfix.json"); qa_c = J("q11a_control.json")
        _tr = qa_m["snapshot"]["pool_swap_c4"]["ring_stats"]["n_translate_read"]
        add2("判别:读侧物化修正无效(伤害通道重开)",
             f"{qa_m['acc']:.3f} / {qa_c['acc']:.3f}",
             f"强制 sparse 预填分支 + stageb-r:n_translate_read={_tr:,} "
             "执行自证、orphan=0(非驻留全命中 packed),c4 驱逐 355 次 —— "
             f"acc={qa_m['acc']:.3f} 与对照 {qa_c['acc']:.3f} 同带同族。"
             "'预填 extra 错读是驱逐伤害媒介'证伪;与消融合并:通道必要、"
             "预填寻址非媒介 —— 修错了相位(后由 q11t2 定谳:伤害消费者"
             "= decode 侧 extra 读,见下两条)")

        # ---- Q11 结案(2026-08-05,46 轮):持久态全同 + decode 读侧定谳 ----
        def _pair_stats(wa, wb, merge_gen):
            import collections as _c
            def _m(w):
                m = _c.defaultdict(list)
                for k, v in w.items():
                    kk = ":".join(k.split(":")[:2]) if merge_gen else k
                    vs = v[1] if isinstance(v[1], list) else [v[1]]
                    m[kk].extend(vs)
                return {k: sorted(v) for k, v in m.items()}
            ma, mb = _m(wa), _m(wb)
            rel = []
            for k in set(ma) & set(mb):
                va, vb = ma[k], mb[k]
                if len(va) != len(vb):
                    continue
                rel += [abs(x - y) / max(abs(x), abs(y), 1e-6)
                        for x, y in zip(va, vb)]
            rel.sort()
            n = len(rel)
            return (rel[n // 2],
                    100.0 * sum(1 for r in rel if r > 0.1) / n, n)
        c4f = _pair_stats(
            J("q11p_rep1.json")["snapshot"]["stageb_oracle"]["w"],
            J("q11p_rep2.json")["snapshot"]["stageb_oracle"]["w"], True)
        c4x = _pair_stats(
            J("q11q_ring129.json")["snapshot"]["stageb_oracle"]["w"],
            J("q11q_ring257.json")["snapshot"]["stageb_oracle"]["w"], True)
        swf = _pair_stats(
            J("q11r2_rep1.json")["snapshot"]["swa_oracle"]["w"],
            J("q11r2_rep2.json")["snapshot"]["swa_oracle"]["w"], False)
        swx = _pair_stats(
            J("q11s_ring129.json")["snapshot"]["swa_oracle"]["w"],
            J("q11s_ring257.json")["snapshot"]["swa_oracle"]["w"], False)
        add2("判别:持久态跨臂全同(v6 内容对拍)",
             f"c4 中位 {c4x[0]:.4f}(底噪 {c4f[0]:.4f});"
             f"SWA 中位 {swx[0]:.4f}(底噪 {swf[0]:.4f})",
             f"L1+有序多重集度量,同配置重复对先定底噪再判读:c4(q11q,"
             f"{c4x[2]:,} 配对元素)>10% 键 {c4x[1]:.1f}% vs 底噪 "
             f"{c4f[1]:.1f}%;SWA(q11s,{swx[2]:,} 元素)>10% "
             f"{swx[1]:.1f}% vs 底噪 {swf[1]:.1f}% —— 腐坏臂(0.500)与"
             "健康臂(1.000)的 c4 与 SWA 写入内容均在底噪带内,持久态"
             "全同;SWA 行是预填残差流的函数,故预填计算两臂等价,伤害"
             "只能在 decode 瞬态。前代哈希差判读(q11l)以坏度量作废")
        s129 = J("q11s_ring129.json")["snapshot"]
        t2f = J("q11t2_r129fix.json"); t2c = J("q11t2_r129ctl.json")
        _ntr = s129["pool_swap_c4"]["ring_stats"]["n_translate_read"]
        _xm = s129["sm120_route"]["extra_max_ring-c4"]
        _ev = t2f["snapshot"]["pool_swap_c4"]["ring_stats"]["n_evict"]
        _dtr = t2f["snapshot"]["packed_kernel"]["n_dtr_calls"]
        add2("判别:decode extra 读侧未翻译定谳(干预确认,46 轮收官)",
             f"{t2f['acc']:.3f} vs {t2c['acc']:.3f}",
             f"机制:decode extra 读以逻辑槽直索物理坐标的后备存储(读侧翻译"
             f"执行数 n_translate_read={_ntr};请求逻辑条目上界 "
             f"{_xm:,} > ring129 物理容量 8,256)—— 零驱逐+单请求时物理"
             "=逻辑恒等映射侥幸成立,驱逐回收/多请求共环即读到他条目"
             "内容且 written/代际审计不报警(q10a 零驱逐反例同机制)。"
             f"干预:非驻留读折 -1({_dtr:,} 调用)= **诚实截断被逐条目**"
             f"(~26% extra 行被弃,微缩仍全对;后续判别定谳:重映射对请求"
             f" id ≈恒等,疗效即截断本身,且时机敏感 —— 驱逐落进预填窗口时"
             f"截断亦致 0.500),W1 ring129(驱逐 {_ev})"
             f"{t2c['acc']:.3f}→{t2f['acc']:.3f},对照臂翻译计数确认为空,"
             "驱逐/调用数两臂全同。**范围口径与开放问题**:W1 微缩干预"
             "有效性确认(驱逐晚于预填时);W4 规模剂量分层四对+方差基线"
             "判定 —— 剂量随机、同剂量两臂共曲线(截断与腐坏等价伤害);"
             "九轮穷尽仪器化(选择流含底噪/服务内容/档完整性/键控/topk)"
             "全部跨臂等价后质量分裂依旧,**伤害微观机制最后一公里开放**;"
             "生产可证语义 = 零驱逐 + 请求隔离(三轮 W4 零驱逐全对)")
        # ---- p111 双池 PPL(2026-08-06,tp4 同口径配对)----
        import math as _m
        _pb = J("p111_lp_base.json"); _pp = J("p111_lp_packed.json")
        if isinstance(_pb, dict): _pb = _pb["docs"]
        if isinstance(_pp, dict): _pp = _pp["docs"]
        _nb = sum(x["n_tok"] for x in _pb); _np = sum(x["n_tok"] for x in _pp)
        _ppl_b = _m.exp(-sum(x["sum_lp"] for x in _pb) / _nb)
        _ppl_p = _m.exp(-sum(x["sum_lp"] for x in _pp) / _np)
        add2("双池 PPL 配对(teacher-forcing)",
             f"{_ppl_b:.4f} vs {_ppl_p:.4f}",
             f"tp4 同口径双臂(此前 base 沿用 tp8 的 p108,跨 tp logprob "
             f"有真实差,已重制):natural+code 24 篇 × 4k 前缀,"
             f"{_nb:,} token 逐 token logprob;packed(INT4/INT6 消费)"
             f"ΔPPL {100*(_ppl_p/_ppl_b-1):+.2f}% —— PPL 口径中性。"
             "flagship dual_pool_ppl 由 PENDING 转有数;激活双向门口径:packed 臂 "
             "消费计数为正,base 臂以 manifest 证明 packed 开关/适配器缺席")
    except FileNotFoundError as e:
        print("[canon] Q4/Q5/Q6 产物缺失,跳过:", e)
    return C


def main():
    C = build()
    pj = os.path.join(ROOT, "papers", "p1-kv-certificates", "canon.json")
    json.dump({"generated_by": "tools/make_canon.py", "entries": C},
              open(pj, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    lines = ["# 冻结的数字与口径清单(CANON)", "",
             "**本文件由 `tools/make_canon.py` 从 `experiments/out/*.json` 自动生成,请勿手改。**", "",
             "`tests/test_paper_claims.py` 用它同时核查中文稿、`papers/p1-kv-certificates/arxiv/main.tex`、",
             "`papers/p1-kv-certificates/abstract_en.md` —— 任何一份文档漏写或写错其中一个数字都会 FAIL。",
             "这是为了解决『代码比论文成熟得快、多份文档相互矛盾』的问题。", ""]
    cur = None
    for e in C:
        if e["group"] != cur:
            cur = e["group"]
            lines += [f"## {cur}", "", "| 量 | 值 | 口径 |", "|---|---|---|"]
        lines.append(f"| {e['name']} | **{e['value']}** | {e['scope']} |")
    lines.append("")
    open(os.path.join(ROOT, "papers", "p1-kv-certificates", "CANON.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"生成 {len(C)} 条:papers/p1-kv-certificates/CANON.md + papers/p1-kv-certificates/canon.json")


if __name__ == "__main__":
    main()
