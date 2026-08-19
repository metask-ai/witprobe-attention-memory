# -*- coding: utf-8 -*-
"""旗舰一键 gate:把散在 experiments/out/ 的判据产物聚合成一份分层报告。

分层(每项 = {value, source_file, gate, caliber}):
  A 机制:并发身份隔离 / e-process 闭环 / fail-closed / 生产 kernel 消费 / 写入见证;
  B 质量:配对 PPL / 输出地板 / Q0 量化门 / 真实条目往返 / 双池质量 / RULER;
  C 性能:池字节 GPU 实测 / 加权节省核算 / HBM 服务差分 / 吞吐·TTFT·ITL / 融合解包核;
  D 路径矩阵:p110 六路径(cudagraph radix mtp hisparse tpcouple pd)统一判读转录。

gate 语义(三态,显式区分"没测"与"测挂了"):
  PASS    判据满足;
  FAIL    产物在但判据不满足,或产物损坏(损坏 = FAIL,不许静默降级成 PENDING);
  PENDING 产物尚不存在 —— 旗舰声明需要但还没测。每个 PENDING 都带 how_to_obtain
          (哪个 launcher / 待写),并汇总进报告的 telemetry_pending 段。

尚未落地的产物按"前向契约"评估:约定文件名落盘后,顶层 gate.passed(或 passed)
布尔即被采信;文件在但无该字段 → FAIL(产物不合契约)。契约文件名见 FUTURE 表。

总 gate = 全 PASS。PENDING 显式列出,不算 PASS 也不算 FAIL。
退出码:0 全 PASS;1 有 FAIL;2 无 FAIL 但有 PENDING;125 拒绝覆盖他人产物。

python3 experiments/flagship_gate.py             # 打印报告
python3 experiments/flagship_gate.py --json      # 另写 experiments/out/flagship_gate.json
python3 experiments/flagship_gate.py --out-dir D # 测试用:换产物目录
"""
import argparse
import json
import math
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GENERATED_BY = "flagship_gate.py"

#: 质量地板:同拓扑配对 PPL 相对差上限(%)。p109 清洁对照实测 0.188%,
#: 地板取 1% 是旗舰声明的质量红线,不是拟合现值。
PPL_FLOOR_PCT = 1.0

#: p99 并发身份 gate 的 12 项判据(全部为 true 才算机制成立)。
P99_CRITERIA = (
    "n_user_requests==8", "all_budgets<=delta_req", "foreign_reads==0",
    "read_unwritten==0", "auth_violations==0", "uid_group_fallbacks==0",
    "identity_fail_closed==0", "eprocess_has_factors", "eprocess_logM_present",
    "eprocess_not_crossed", "n_ranks_seen==8", "replica_coupling_identical",
)

#: 前向契约:旗舰声明需要、当前尚无产物的遥测。文件落盘后本脚本自动采信
#: 顶层 gate.passed / passed。how_to_obtain 是获取方式(launcher 或待写)。
FUTURE = {
    "hbm_serving": {
        "file": "p112_serving_hbm.json",
        "how_to_obtain": "待写 launcher(建议 p112_serving_hbm.sh,起服方式沿 "
                         "p111_dual_pool.sh):同负载双臂起服,KV 池字节与总 HBM 差分",
        "caliber": "服务口径 HBM 差分;q1 的 -38.4% 是池字节微基准,总量节省依赖池占比",
    },
    "serving_latency": {
        "file": "p113_serving_latency.json",
        "how_to_obtain": "待写 launcher(建议 p113_serving_latency.sh):bench_serving "
                         "双臂对照,吞吐/TTFT/ITL/cost_per_successful_output_token",
        "caliber": "在线服务时延与吞吐,packed vs FP8 同拓扑同负载配对",
    },
    "fused_unpack": {
        "file": "p115_fused_unpack.json",
        "how_to_obtain": "待写 launcher(kernel 主线产出后):融合 Triton 解包核服务实测;"
                         "q1 的 4.57× 是 torch 参考对生产 FP8 核的未融合上界",
        "caliber": "解包核开销,融合实现 vs 生产 FP8 dequantize 核",
    },
}

#: D 层六路径与 p110_sixpath.json summary 的键一一对应。
SIXPATH_ITEMS = ("cudagraph", "radix", "mtp", "hisparse", "tpcouple", "pd")
SIXPATH_HOW = ("tools/run_remote.sh hgx experiments/launchers/p110_sixpath_%s.sh + "
               "--pull 后本地 python3 experiments/p110_sixpath.py")


def _load(out_dir, name):
    p = os.path.join(out_dir, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except ValueError:
        return "CORRUPT"


def _item(value, source_file, gate, caliber, how_to_obtain=None):
    d = {"value": value, "source_file": source_file, "gate": gate, "caliber": caliber}
    if how_to_obtain:
        d["how_to_obtain"] = how_to_obtain
    return d


def _pending(source_file, caliber, how_to_obtain):
    return _item(None, source_file, "PENDING", caliber, how_to_obtain)


def _gate(ok):
    return "PASS" if ok else "FAIL"


def _q0_ref(out_dir):
    """Q0 离线扫描给出的各池**理论**重构误差(生产选型那一档)。

    门限对表用:阈值必须来自"正确实现下该指标应当取什么值",不是拍一个宽区间。
    2026-08-02 教训:本项此前只查覆盖(n_unwritten==0),rel 只显示不判定,
    于是 c4 实测 0.135(理论 0.0239)照样 PASS,陈旧槽污染畅通无阻。
    """
    d = _load(out_dir, "q0_sweep.json")
    if not d or d == "CORRUPT":
        return {}
    want = {"c4": "C_had64_int6", "c128": "C_had64_int4"}
    ref = {}
    for pool, name in want.items():
        for row in ((d.get("pools", {}).get(pool, {}) or {}).get("validation") or []):
            if row.get("name") == name and "rel_l2_mean" in row:
                ref[pool] = float(row["rel_l2_mean"])
    return ref


# ---------------------------------------------------------------- A 机制

def layer_a(out_dir):
    items = {}
    how99 = "tools/run_remote.sh hgx experiments/launchers/p99_hardened.sh"
    p99 = _load(out_dir, "p99_concurrent_identity.json")
    if p99 is None:
        for k, cal in (("identity_isolation", "并发下逐条目归属隔离"),
                       ("identity_fail_closed", "身份不明条目 fail-closed"),
                       ("eprocess_loop", "e-process 闭环"),
                       ("replica_coupling", "副本耦合一致性"),
                       ("privacy_budget", "逐请求隐私预算")):
            items[k] = _pending("p99_concurrent_identity.json", cal, how99)
    elif p99 == "CORRUPT" or "gate" not in p99 or "criteria" not in p99.get("gate", {}):
        items["identity_isolation"] = _item(
            "产物损坏或缺 gate.criteria", "p99_concurrent_identity.json", "FAIL",
            "损坏产物按 FAIL 处理,不降级 PENDING")
    else:
        cr = p99["gate"]["criteria"]
        sm = p99.get("summary", {})
        cal_base = "8 路并发终态快照,tp8+ep8,hgx;判据键沿 p99 gate.criteria 原文"
        iso_keys = ("foreign_reads==0", "read_unwritten==0", "auth_violations==0",
                    "uid_group_fallbacks==0", "n_user_requests==8")
        items["identity_isolation"] = _item(
            "foreign_reads=%s read_unwritten=%s auth_violations=%s" % (
                sm.get("foreign_reads_user"), sm.get("read_unwritten_user"),
                sm.get("auth_violations")),
            "p99_concurrent_identity.json",
            _gate(all(cr.get(k) is True for k in iso_keys)),
            cal_base + ";foreign 读是归属正确性的运行时守卫,错位立即非零")
        items["identity_fail_closed"] = _item(
            "identity_fail_closed=0", "p99_concurrent_identity.json",
            _gate(cr.get("identity_fail_closed==0") is True),
            cal_base + ";0 = 无请求因身份不明触发 fail-closed 回退")
        ep = sm.get("eprocess_global", {})
        items["eprocess_loop"] = _item(
            "n_factors=%s log_M_max=%s crossed=%s" % (
                ep.get("n_factors"), ep.get("log_M_max"), ep.get("crossed")),
            "p99_concurrent_identity.json",
            _gate(all(cr.get(k) is True for k in (
                "eprocess_has_factors", "eprocess_logM_present", "eprocess_not_crossed"))),
            cal_base + ";log_M_max 未越界即 e-process 未穿越")
        items["replica_coupling"] = _item(
            "replica_coupling_identical=%s n_ranks_seen=8" % sm.get(
                "replica_coupling_identical"),
            "p99_concurrent_identity.json",
            _gate(all(cr.get(k) is True for k in (
                "n_ranks_seen==8", "replica_coupling_identical"))),
            cal_base + ";每账户独立随机流,批流 rank 间一致才耦合")
        items["privacy_budget"] = _item(
            "max_delta_spent=%s" % sm.get("max_delta_spent"),
            "p99_concurrent_identity.json",
            _gate(cr.get("all_budgets<=delta_req") is True),
            cal_base + ";全部请求预算 <= delta_req")
        missing = [k for k in P99_CRITERIA if k not in cr]
        if missing:
            items["identity_criteria_complete"] = _item(
                "缺判据键:%s" % missing, "p99_concurrent_identity.json", "FAIL",
                "p99 gate.criteria 应含 12 项;键缺失说明产物 schema 变了,先 git log 对账")

    p98 = _load(out_dir, "p98_concurrent_identity.json")
    if p98 is None:
        items["identity_first_run"] = _pending(
            "p98_concurrent_identity.json", "并发身份首采(p99 为加固复跑)",
            "tools/run_remote.sh hgx experiments/launchers/p98_concurrent_identity.sh")
    elif p98 == "CORRUPT" or not p98.get("gate", {}).get("criteria"):
        items["identity_first_run"] = _item(
            "产物损坏或缺 gate.criteria", "p98_concurrent_identity.json", "FAIL",
            "损坏产物按 FAIL 处理")
    else:
        cr98 = p98["gate"]["criteria"]
        n_ok = sum(1 for v in cr98.values() if v is True)
        items["identity_first_run"] = _item(
            "criteria %d/%d" % (n_ok, len(cr98)), "p98_concurrent_identity.json",
            _gate(p98["gate"].get("passed") is True and n_ok == len(cr98)),
            "首采与 p99 加固复跑构成两次独立并发身份验收")

    p106 = _load(out_dir, "p106_packed_decode.json")
    how106 = "tools/run_remote.sh hgx experiments/launchers/p106_packed_decode.sh"
    if p106 is None:
        items["kernel_consumption"] = _pending(
            "p106_packed_decode.json", "生产路径对 packed 条目的真实消费", how106)
        items["packshadow_witness"] = _pending(
            "p106_packed_decode.json", "packed 影子读回见证", how106)
    elif p106 == "CORRUPT" or not p106.get("gate", {}).get("criteria"):
        items["kernel_consumption"] = _item(
            "产物损坏或缺 gate.criteria", "p106_packed_decode.json", "FAIL",
            "损坏产物按 FAIL 处理")
    else:
        cr = p106["gate"]["criteria"]
        rt = p106.get("sm120_route", {})
        mech_keys = ("route_forced", "entry_gt0", "packed_rows_gt0", "err_none",
                     "selfcheck_stable")
        items["kernel_consumption"] = _item(
            "n_packed_rows=%s n_entry=%s n_gather=%s" % (
                rt.get("n_packed_rows"), rt.get("n_entry"), rt.get("n_gather")),
            "p106_packed_decode.json",
            _gate(all(cr.get(k) is True for k in mech_keys)),
            "sm120 参考路由强制生效 + packed 行真实消费(零触发即败);"
            "参考路径性能非生产 kernel,fail-closed 到 FP8;"
            "p109 口径下生产 packed kernel 覆盖 1040/2278 调用,余走 fail-closed")
        ps = p106.get("packshadow_check") or {}
        ref = _q0_ref(out_dir)
        # 优先按池判定:c4/c128 各自对表自己的理论值。历史产物无分列字段时回退聚合。
        by_pool = p106.get("packshadow_by_pool") or {}
        rel_mean, rel_max = ps.get("rel_mean"), ps.get("rel_max")
        # 参考值按该产物实际打包的池选;产物未声明池时取较宽者并在 value 里
        # 把两个理论值都列出来,避免"用最松的界蒙混"而读者看不见
        hint = json.dumps(p106.get("what", ""), ensure_ascii=False) + \
            json.dumps(p106.get("caliber", ""), ensure_ascii=False)
        if by_pool:
            over_pool = [k for k, v in by_pool.items()
                         if k in ref and v.get("rel_mean", 0) > 1.5 * ref[k]]
            pools = sorted(by_pool)
            adm = 1.5 * max((ref[k] for k in pools if k in ref), default=0) or None
        else:
            over_pool = []
            pools = [k for k in ("c4", "c128") if k in ref and k in hint] or list(ref)
            adm = 1.5 * max(ref[k] for k in pools) if pools else None
        # 容许区间对表(2026-08-02 新增,见 docs/research/witcert-rrd/ASSESSMENT §2.1):
        #   rel>=1 = 重构比零向量还远,正确量化器物理上做不出 → 数据无效,判 FAIL;
        #   rel_mean > 1.5x 理论值 → 判 FAIL(此前只查覆盖,污染数据照样 PASS)。
        impossible = rel_max is not None and rel_max >= 1.0
        over = bool(over_pool) if by_pool else (
            adm is not None and rel_mean is not None and rel_mean > adm)
        items["packshadow_witness"] = _item(
            "n_checked=%s n_unwritten=%s rel_mean=%s rel_max=%s (理论 %s,容许 ≤%s)" % (
                ps.get("n_checked"), ps.get("n_unwritten"), rel_mean, rel_max,
                ref or "q0_sweep 缺失", ("%.4f" % adm) if adm else "未知"),
            "p106_packed_decode.json + q0_sweep.json",
            _gate(bool(ps) and ps.get("n_unwritten") == 0
                  and ps.get("n_checked", 0) > 0 and not impossible and not over),
            "影子读回逐条目对账:read_unwritten 必须为 0(读到未写条目 = 池身份错);"
            "rel 必须落在理论容许区间内 —— rel≥1 判数据无效(陈旧/跨代条目),"
            "rel_mean 超理论 1.5× 判失真异常,阈值溯源 q0_sweep.json 而非拍定")
    return items


# ---------------------------------------------------------------- B 质量

def _ppl_from_lp(rows):
    if isinstance(rows, dict):                 # 2026-08-06 富化 schema
        rows = rows["docs"]
    n = sum(r["n_tok"] for r in rows)
    return math.exp(-sum(r["sum_lp"] for r in rows) / n)


#: base 臂必须可证缺席的 packed 开关(评审 P1:缺失计数不当零)
_PACKED_SWITCHES = ("WITCERT_PACKED_C4", "WITCERT_PACKED_C128",
                    "WITCERT_PACKED_KERNEL")


def _lp_activation(lb, lp):
    """路径激活双向门 v2(评审 P1 收紧):
    packed 臂:n_packed_c4>0 ∧ n_packed_c128>0;
    base 臂:计数**显式为零**,或计数缺失但 manifest 同时证明
      ①witcert_env 无任何 packed 开关 ②adapters_status 无 packed 适配器
      (缺计数且无缺席证明 = 仪器未装载嫌疑,不当零);
    两臂 code 必须为真实指纹(unset/缺失 → provisional)。
    旧 schema(裸列表)→ None = provisional。"""
    if not (isinstance(lb, dict) and isinstance(lp, dict)):
        return None
    mp, mb = lp.get("manifest") or {}, lb.get("manifest") or {}
    if (mp.get("code") in (None, "", "unset")
            or mb.get("code") in (None, "", "unset")):
        return None
    pk, bs = mp.get("packed_kernel") or {}, mb.get("packed_kernel") or {}
    if not ((pk.get("n_packed_c4") or 0) > 0
            and (pk.get("n_packed_c128") or 0) > 0):
        return False
    b4, b128 = bs.get("n_packed_c4"), bs.get("n_packed_c128")
    if b4 == 0 and b128 == 0:
        return True                        # 显式零
    if b4 is None and b128 is None:
        env = mb.get("witcert_env") or {}
        ads = " ".join(mb.get("adapters_status") or [])
        absent = (not any(env.get(k) == "1" for k in _PACKED_SWITCHES)
                  and "packed" not in ads and "stageb" not in ads)
        return True if absent else None    # 有缺席证明才认
    return False


def layer_b(out_dir):
    items = {}
    q = _load(out_dir, "p109_quality_clean.json")
    if q is None:
        items["ppl_clean"] = _pending(
            "p109_quality_clean.json", "同拓扑配对 PPL(旗舰质量红线 ≤%.1f%%)" % PPL_FLOOR_PCT,
            "tools/run_remote.sh hgx experiments/launchers/p109_kernel_quality.sh + "
            "p109b_tp4_base.sh")
    elif q == "CORRUPT" or "delta_pct" not in q:
        items["ppl_clean"] = _item("产物损坏或缺 delta_pct", "p109_quality_clean.json",
                                   "FAIL", "损坏产物按 FAIL 处理")
    else:
        items["ppl_clean"] = _item(
            "delta_pct=%.4f%% (fp8 %.6f -> packed %.6f)" % (
                q["delta_pct"], q.get("ppl_fp8_tp4", float("nan")),
                q.get("ppl_packed_tp4", float("nan"))),
            "p109_quality_clean.json", _gate(q["delta_pct"] <= PPL_FLOOR_PCT),
            "同拓扑(tp4)同语料同文档配对,拓扑混杂已消除;packed kernel 覆盖 46%%,"
            "余 fail-closed —— 引用该数字必须带覆盖口径;红线 ≤%.1f%%" % PPL_FLOOR_PCT)

    p106 = _load(out_dir, "p106_packed_decode.json")
    if p106 is None:
        items["output_floor"] = _pending(
            "p106_packed_decode.json", "贪心解码输出过重启地板",
            "tools/run_remote.sh hgx experiments/launchers/p106_packed_decode.sh")
    elif p106 == "CORRUPT" or not p106.get("gate", {}).get("criteria"):
        items["output_floor"] = _item("产物损坏或缺 gate.criteria",
                                      "p106_packed_decode.json", "FAIL",
                                      "损坏产物按 FAIL 处理")
    else:
        cr = p106["gate"]["criteria"]
        items["output_floor"] = _item(
            "route_vs_fp8base=%s packed_vs_route=%s" % (
                (p106.get("outputs") or {}).get("route_vs_fp8base"),
                (p106.get("outputs") or {}).get("packed_vs_route")),
            "p106_packed_decode.json",
            _gate(all(cr.get(k) is True for k in
                      ("route_vs_base_at_floor", "packed_vs_route_at_floor"))),
            "p88 口径:贪心解码、完全一致率、地板 = 同配置重启对照;n=16 方向性")

    q0 = _load(out_dir, "q0_sweep.json")
    if q0 is None:
        items["q0_quant_gate"] = _pending(
            "q0_sweep.json", "Q0 量化候选扫描收口(shadow)",
            "tools/run_remote.sh hgx experiments/launchers/p100_q0_capture.sh + "
            "p101_q05_capture.sh 采集后本地 python3 experiments/q0_sweep.py")
    elif q0 == "CORRUPT" or "verdict" not in q0 or "hbm_saving" not in q0:
        items["q0_quant_gate"] = _item("产物损坏或缺 verdict/hbm_saving",
                                       "q0_sweep.json", "FAIL", "损坏产物按 FAIL 处理")
    else:
        oc = q0["verdict"].get("optimal_config", {})
        items["q0_quant_gate"] = _item(
            {"c4": oc.get("c4"), "c128": oc.get("c128")}, "q0_sweep.json",
            _gate(bool(oc.get("c4")) and bool(oc.get("c128"))),
            "shadow 口径(非物理);calibration/validation/hidden 层分裂防过拟合;"
            "否决路线(c4 INT4/INT5、INT2 全族、1-bit RQ、varnorm、池级 PCA)保留在 verdict")

    q1 = _load(out_dir, "q1_packed_int4.json")
    how_q1 = "hgx 上按 q1_packed_int4.json 的 source(p101 raw 采集)复算," \
             "python3 experiments/q1_packed_int4.py"
    if q1 is None:
        items["q1_roundtrip"] = _pending(
            "q1_packed_int4.json", "真实条目 packed INT4 物理往返", how_q1)
    elif q1 == "CORRUPT" or "roundtrip" not in q1:
        items["q1_roundtrip"] = _item("产物损坏或缺 roundtrip", "q1_packed_int4.json",
                                      "FAIL", "损坏产物按 FAIL 处理")
    else:
        rt = q1["roundtrip"]
        items["q1_roundtrip"] = _item(
            "rel_l2_mean=%.4f rel_l2_p95=%.4f cov_W<=4=%s (n=%s)" % (
                rt.get("rel_l2_mean", float("nan")), rt.get("rel_l2_p95", float("nan")),
                rt.get("cov_W_le_4"), rt.get("n_entries")),
            "q1_packed_int4.json", _gate(rt.get("cov_W_le_4") == 1.0),
            "真实 c128 条目 GPU 往返,与 shadow had64_int4 同一数学;见证覆盖 W<=4 须 100%")

    lb = _load(out_dir, "p111_lp_base.json")
    lp = _load(out_dir, "p111_lp_packed.json")
    how111 = "tools/run_remote.sh hgx experiments/launchers/p111_dual_pool.sh(lane A)"
    if lb is None or lp is None:
        items["dual_pool_ppl"] = _pending(
            "p111_lp_base.json + p111_lp_packed.json",
            "双池 packed(c4 INT6 + c128 INT4)配对 PPL —— 旗舰主配置的质量验收", how111)
    elif lb == "CORRUPT" or lp == "CORRUPT":
        items["dual_pool_ppl"] = _item("产物损坏", "p111_lp_*.json", "FAIL",
                                       "损坏产物按 FAIL 处理")
    else:
        pb, pp = _ppl_from_lp(lb), _ppl_from_lp(lp)
        delta = (pp / pb - 1.0) * 100.0
        act = _lp_activation(lb, lp)
        verdict = ("PASS" if (delta <= PPL_FLOOR_PCT and act is True)
                   else "FAIL" if act is False or delta > PPL_FLOOR_PCT
                   else "PENDING")
        items["dual_pool_ppl"] = _item(
            "delta_pct=%.4f%% (base %.6f -> packed %.6f) activation=%s"
            % (delta, pb, pp, act),
            "p111_lp_base.json + p111_lp_packed.json",
            verdict,
            "聚合式与 p109_quality 相同;红线 ≤%.1f%%;**激活双向门**:"
            "packed 臂 n_packed_c4>0∧c128>0、base 臂两者为零(此前发生过 "
            "PPL 测错路径,无激活证据只算 provisional=PENDING)"
            % PPL_FLOOR_PCT,
            how_to_obtain=how111 if act is None else None)

    items["ruler_flagship"] = _ruler_matched(out_dir)
    return items


def _ruler_matched(out_dir):
    """RULER niah:**matched control 双臂**(同 launcher、同配置、radix 一律 off)。

    2026-08-02 口径:radix=on 时三个配置同为 0.800 —— 那是缓存命中率,不是质量
    (p116/p118 已证)。故本项只认 radix=off 的配对。生产 packed 臂优先取 p119
    (带开关生效计数器),回退 p118。诊断臂(every-read 全量重填)不作生产证据。
    """
    base = _load(out_dir, "p118_ruler_base_off.json")
    # 候选按新到旧;**崩溃臂(n_error>0)不是测量**,跳过并披露,而不是让一次
    # 崩溃顶掉同配置的有效测量。全部无效才判 FAIL。
    pk_file, pk, skipped = None, None, []
    # p124c = 代际失效修复默认开启后的 packed 臂(并发 4 口径,acc 1.000,
    # p124_verdict 六条全过);p119/p118 是修复前的臂,仅当新产物缺失时回退
    for cand in ("p124c_conc_inval_on.json", "p119_freshness_packonce.json",
                 "p118_ruler_packed_off.json"):
        d = _load(out_dir, cand)
        if d is None:
            continue
        if d != "CORRUPT" and (d.get("n_error") or 0) > 0:
            skipped.append("%s(n_error=%s)" % (cand, d.get("n_error")))
            continue
        pk_file, pk = cand, d
        break
    if pk_file is None:
        pk_file = "p119_freshness_packonce.json"
    src = "p118_ruler_base_off.json + " + pk_file
    how = "tools/run_remote.sh hgx experiments/launchers/p119_freshness.sh"
    cal = ("长上下文检索任务分,packed vs FP8 同配置配对(radix=off);"
           "**当前的 ΔPPL 实验(p108)替代不了本项** —— 它取 prefill 侧 "
           "input logprob,而 packed 池只在 decode 被消费,走的不是同一条路径,"
           "因此验证不了本次压缩路径;这是实验设计走错路径,不是 PPL 这个指标"
           "天生不敏感(TinyKG 10664)")
    if base is None or pk is None:
        return _pending(src, cal, how)
    if base == "CORRUPT" or pk == "CORRUPT":
        return _item("产物损坏", src, "FAIL", "损坏产物按 FAIL 处理")
    ab, ap = base.get("acc"), pk.get("acc")
    if ab is None or ap is None:
        return _item("产物缺 acc 字段", src, "FAIL", "不合契约")
    # 带请求错误的臂不是质量测量:分母含失败项时 acc 只是"没答上来"的比例。
    # 2026-08-02 实例:一轮 p119 被人工中止,孤儿客户端把 19 个连接错误连同
    # acc=0.05 写成了产物 —— 若按 acc 读就会把一次中止当成质量结论。
    if (base.get("n_error") or 0) > 0:
        return _item("base 臂 n_error=%s(共 %s)—— 该臂无效" % (
            base.get("n_error"), base.get("n")), src, "FAIL",
            cal + ";**含请求错误的臂不作质量测量**,先查该臂服务是否崩溃")
    if skipped:                       # 跳过了崩溃臂就必须说出来
        cal += ";已跳过无效臂:" + "、".join(skipped)
    ev = ((pk.get("snapshot") or {}).get("all_ranks") or {})
    dc4 = (((pk.get("snapshot") or {}).get("decode_shadow_all_ranks") or {})
           .get("c4") or {})
    note = ""
    if "n_repack" in ev:
        note = " | n_invalidate=%s" % ev.get("n_invalidate")
        if dc4:
            # 数据有效性按**解码侧**判(内核实际读到的东西);centry 侧混合对账
            # 对 c4 无效(双写者,构造性跨代比较,p120 定谳),只展示不判定
            note += " | 解码侧 c4 rel_mean=%s rel_max=%s" % (
                round(dc4.get("rel_mean", -1), 5), round(dc4.get("rel_max", -1), 4))
            if (dc4.get("rel_max") or 0) >= 1.0:
                return _item("acc %.3f vs base %.3f%s" % (ap, ab, note), src, "FAIL",
                             cal + ";**解码侧 rel_max≥1 判数据无效**(内核读到"
                             "跨代条目),结论不得用于理论归因")
        elif (ev.get("rel_max") or 0) >= 1.0:  # 无解码侧数据的旧产物才用 centry 兜底
            return _item("acc %.3f vs base %.3f%s" % (ap, ab, note), src, "FAIL",
                         cal + ";centry 侧 rel_max≥1(无解码侧数据,保守判无效)")
    return _item("packed acc=%.3f vs base acc=%.3f (Δ=%.3f)%s" % (ap, ab, ap - ab, note),
                 src, _gate(ap >= ab - 0.05), cal + ";红线:packed 不低于 base 5 个点")


# ---------------------------------------------------------------- C 性能

def _future_item(out_dir, key):
    spec = FUTURE[key]
    d = _load(out_dir, spec["file"])
    if d is None:
        return _pending(spec["file"], spec["caliber"], spec["how_to_obtain"])
    if d == "CORRUPT":
        return _item("产物损坏", spec["file"], "FAIL", "损坏产物按 FAIL 处理")
    passed = d.get("gate", {}).get("passed") if isinstance(d.get("gate"), dict) \
        else d.get("passed")
    if not isinstance(passed, bool):
        return _item("产物在但无 gate.passed/passed 布尔,不合前向契约",
                     spec["file"], "FAIL", spec["caliber"])
    return _item(d.get("summary", "gate.passed=%s" % passed), spec["file"],
                 _gate(passed), spec["caliber"])


def layer_c(out_dir):
    items = {}
    q1 = _load(out_dir, "q1_packed_int4.json")
    how_q1 = "hgx 复跑 python3 experiments/q1_packed_int4.py(见其 source 字段)"
    if q1 is None:
        items["pool_bytes_gpu"] = _pending(
            "q1_packed_int4.json", "packed INT4 池字节 GPU 实测", how_q1)
    elif q1 == "CORRUPT" or "pool_bytes_gpu" not in q1:
        items["pool_bytes_gpu"] = _item("产物损坏或缺 pool_bytes_gpu",
                                        "q1_packed_int4.json", "FAIL",
                                        "损坏产物按 FAIL 处理")
    else:
        pb = q1["pool_bytes_gpu"]
        ok = (pb.get("measured_saving", 0) > 0
              and abs(pb.get("measured_saving", 0)
                      - pb.get("format_saving_nominal", -1)) < 1e-9)
        items["pool_bytes_gpu"] = _item(
            "saving=%.4f (fp8 %s MiB -> int4 %s MiB, %s token)" % (
                pb.get("measured_saving", float("nan")), pb.get("fp8_pool_MiB"),
                pb.get("int4_pool_MiB"), pb.get("n_tokens")),
            "q1_packed_int4.json", _gate(ok),
            "torch.cuda.memory_allocated 差分,hgx;实测 = 名义才 PASS(有水分即败);"
            "池字节口径,非总 HBM,非服务集成")
        ub = q1.get("unpack_bench") or {}
        items["unpack_upper_bound"] = _item(
            "ratio_int4_over_fp8=%.2f (torch 参考,未融合上界)" % ub.get(
                "ratio_int4_over_fp8", float("nan")),
            "q1_packed_int4.json",
            # 红线(主张级):未融合上界超过 10× 时,即使融合拿到 ~3× 也回不到
            # 平价,方向即已死。原判据只有 ratio>0,任何正数都过。
            _gate(0 < ub.get("ratio_int4_over_fp8", 0) <= 10.0),
            "CUDA event 微基准;int4 为 torch 参考实现 vs 生产 FP8 Triton 核;"
            "红线 0<ratio≤10 —— 只证明上界存在且在可挽救范围内,"
            "融合核实测见 fused_unpack(PENDING)")

    q0 = _load(out_dir, "q0_sweep.json")
    if q0 is None:
        items["weighted_pool_saving"] = _pending(
            "q0_sweep.json", "双池内存加权节省核算",
            "同 B 层 q0_quant_gate 的获取方式")
    elif q0 == "CORRUPT" or "verdict" not in q0:
        items["weighted_pool_saving"] = _item("产物损坏或缺 verdict", "q0_sweep.json",
                                              "FAIL", "损坏产物按 FAIL 处理")
    else:
        mw = q0["verdict"].get("memory_weighted", {})
        sv = mw.get("compressed_pool_saving")
        items["weighted_pool_saving"] = _item(
            "compressed_pool_saving=%s (weights: %s)" % (sv, mw.get("weights")),
            "q0_sweep.json",
            # 红线(主张级,非拟合现值):加权节省 <10% 时这条路线不值得做;
            # >100% 物理不可能 = 核算口径错,判数据无效。原判据只有 sv>0,
            # 逐字段变异杀不掉 —— 2026-08-02 由 tests/test_gate_falsifiable.py 抓出。
            _gate(isinstance(sv, (int, float)) and 0.10 <= sv < 1.0),
            "**压缩池字节**口径的核算值,非实测;红线 10%≤saving<100%;"
            "总 HBM 还含 swa 窗口池/稠密层/激活,总量节省依赖池占比 —— "
            "服务实测见 hbm_serving(PENDING)")

    items["hbm_serving"] = _future_item(out_dir, "hbm_serving")
    items["serving_latency"] = _future_item(out_dir, "serving_latency")
    items["fused_unpack"] = _future_item(out_dir, "fused_unpack")
    return items


# ---------------------------------------------------------------- D 路径矩阵

def layer_d(out_dir):
    items = {}
    six = _load(out_dir, "p110_sixpath.json")
    if six == "CORRUPT":
        items["sixpath_matrix"] = _item("产物损坏", "p110_sixpath.json", "FAIL",
                                        "损坏产物按 FAIL 处理")
        return items
    summary = (six or {}).get("summary", {})
    for it in SIXPATH_ITEMS:
        g = summary.get(it)
        how = SIXPATH_HOW % it
        if it == "pd":
            how = ("需双机拓扑(prefill/decode 两套服务 + KV 传输后端),单机双 lane "
                   "不构成 PD;experiments/launchers/p110_sixpath_pd.sh 为入口占位")
        cal = "p110 统一判读转录:服务健康 ∧ n_packed>0 ∧ err=None ∧ 输出过地板"
        if six is None or g is None:
            items[it] = _pending("p110_sixpath.json", cal + "(尚无判读产物)", how)
        elif g == "PASS":
            items[it] = _item("PASS(p110 判读)", "p110_sixpath.json", "PASS", cal)
        elif g in ("PENDING", "SKIP"):
            items[it] = _pending(
                "p110_sixpath.json",
                cal + ("(p110 判 SKIP:单机不构成 PD,旗舰口径记 PENDING)"
                       if g == "SKIP" else "(p110 判 PENDING)"), how)
        else:
            checks = ((six.get("matrix") or {}).get(it, {}) or {}).get("checks", {})
            items[it] = _item(
                "FAIL(p110 判读;未过:%s)" % [k for k, v in checks.items() if not v],
                "p110_sixpath.json", "FAIL", cal)
    return items


# ---------------------------------------------------------------- 汇总

LAYERS = (("A", "机制:身份 / 闭环 / fail-closed / kernel 消费", layer_a),
          ("B", "质量:PPL / 输出地板 / 量化门 / RULER", layer_b),
          ("C", "性能:HBM / 吞吐(服务实测多为 PENDING)", layer_c),
          ("D", "路径矩阵:p110 六路径", layer_d))


def build_report(out_dir=OUT):
    layers, pending, counts = {}, [], {"PASS": 0, "FAIL": 0, "PENDING": 0}
    for lid, title, fn in LAYERS:
        items = fn(out_dir)
        layers[lid] = {"title": title, "items": items}
        for name, it in items.items():
            counts[it["gate"]] += 1
            if it["gate"] == "PENDING":
                pending.append({"layer": lid, "item": name,
                                "expected_file": it["source_file"],
                                "claim": it["caliber"],
                                "how_to_obtain": it["how_to_obtain"]})
    overall = "FAIL" if counts["FAIL"] else (
        "PENDING" if counts["PENDING"] else "PASS")
    exit_code = {"PASS": 0, "FAIL": 1, "PENDING": 2}[overall]
    return {
        "what": "旗舰一键 gate:A 机制 / B 质量 / C 性能 / D 路径矩阵 分层聚合",
        "generated_by": GENERATED_BY,
        "machine": "local(聚合;数字源产物各带 machine/stack)",
        "caliber": [
            "gate 三态:PASS / FAIL / PENDING;PENDING = 产物不存在(没测),"
            "不算 PASS 也不算 FAIL,单独退出码 2",
            "总 gate = 全 PASS;任一 FAIL 即 FAIL;损坏产物按 FAIL 不按 PENDING",
            "PPL 红线 ≤%.1f%%(声明红线,非拟合现值;p109 清洁对照实测 0.188%%)" % PPL_FLOOR_PCT,
            "PENDING 项按前向契约评估:约定文件落盘后采信顶层 gate.passed/passed;"
            "契约文件名见 telemetry_pending.expected_file",
            "本报告只聚合不复算(p111 配对 PPL 除外,聚合式与 p109_quality 相同);"
            "引用任何数字须回到 source_file 带其原口径",
        ],
        "layers": layers,
        "telemetry_pending": pending,
        "summary": {"overall": overall, "n_pass": counts["PASS"],
                    "n_fail": counts["FAIL"], "n_pending": counts["PENDING"]},
        "exit_code": exit_code,
    }


def write_json(rep, out_dir=OUT, force=False):
    """写 flagship_gate.json;拒绝覆盖非本脚本生成的产物。返回 True=已写。"""
    dst = os.path.join(out_dir, "flagship_gate.json")
    if os.path.exists(dst):
        try:
            old = json.load(open(dst, encoding="utf-8"))
        except ValueError:
            old = {}
        if old.get("generated_by") not in (None, GENERATED_BY) and not force:
            print("拒绝覆盖:%s 由 %r 生成(非本脚本);确认后加 --force"
                  % (dst, old.get("generated_by")))
            return False
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("写出", dst)
    return True


def print_report(rep):
    for lid, _, _ in LAYERS:
        lay = rep["layers"][lid]
        print("[%s] %s" % (lid, lay["title"]))
        for name, it in lay["items"].items():
            print("  %-26s %-7s %s" % (name, it["gate"],
                                       "" if it["value"] is None else it["value"]))
    if rep["telemetry_pending"]:
        print("\ntelemetry 缺口(旗舰声明需要但尚无产物):")
        for p in rep["telemetry_pending"]:
            print("  [%s/%s] 期望产物 %s" % (p["layer"], p["item"], p["expected_file"]))
            print("      获取:%s" % p["how_to_obtain"])
    s = rep["summary"]
    print("\n总 gate: %s(PASS %d / FAIL %d / PENDING %d);退出码 %d"
          % (s["overall"], s["n_pass"], s["n_fail"], s["n_pending"], rep["exit_code"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="旗舰一键 gate")
    ap.add_argument("--json", action="store_true", help="写 out/flagship_gate.json")
    ap.add_argument("--force", action="store_true", help="覆盖非本脚本生成的产物")
    ap.add_argument("--out-dir", default=OUT, help="产物目录(测试用)")
    args = ap.parse_args(argv)
    rep = build_report(args.out_dir)
    print_report(rep)
    if args.json and not write_json(rep, args.out_dir, args.force):
        sys.exit(125)
    sys.exit(rep["exit_code"])


if __name__ == "__main__":
    main()
