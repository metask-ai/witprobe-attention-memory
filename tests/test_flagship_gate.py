# -*- coding: utf-8 -*-
"""flagship_gate 守卫:结构完整性(真实 out/ 产物)+ 三种退出码(fixture)。

T1 真实产物:build_report 结构完整 —— 每项必有 value/source_file/gate/caliber,
   gate ∈ {PASS,FAIL,PENDING},PENDING 必带 how_to_obtain,telemetry_pending
   与 PENDING 项一一对应,exit_code 与计数一致;
T2 fixture 全好(含前向契约产物 + p110 全 PASS)→ overall PASS,exit 0;
T3 fixture 判据翻转(p99 foreign_reads)→ FAIL,exit 1;损坏产物同样 FAIL;
T4 fixture 缺 p110 与前向契约产物 → 无 FAIL 有 PENDING,exit 2;
T5 generated_by 护栏:不覆盖他人产物(main --json 退出 125),--force 才覆盖。

python3 tests/test_flagship_gate.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import flagship_gate as fg  # noqa: E402

GATES = ("PASS", "FAIL", "PENDING")


def _w(d, name, obj):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def make_fixture(d):
    """最小'全绿'产物集:A/B/C 核心 + p111 双池 + 前向契约 + p110 全 PASS。"""
    ident = {
        "gate": {"criteria": {k: True for k in fg.P99_CRITERIA}, "passed": True},
        "summary": {"foreign_reads_user": 0, "read_unwritten_user": 0,
                    "auth_violations": "0/100", "max_delta_spent": 0.005,
                    "replica_coupling_identical": True,
                    "eprocess_global": {"n_factors": 100, "log_M_max": -0.5,
                                        "crossed": False}},
    }
    _w(d, "p99_concurrent_identity.json", ident)
    _w(d, "p98_concurrent_identity.json", ident)
    _w(d, "p106_packed_decode.json", {
        "gate": {"criteria": {k: True for k in (
            "route_forced", "entry_gt0", "packed_rows_gt0", "err_none",
            "route_vs_base_at_floor", "packed_vs_route_at_floor",
            "selfcheck_stable")}, "passed": True},
        "sm120_route": {"forced": True, "n_gather": 8, "n_packed_rows": 100,
                        "n_entry": 4},
        "packshadow_check": {"n_checked": 10, "n_unwritten": 0, "rel_mean": 0.07},
        "outputs": {"route_vs_fp8base": "11/16", "packed_vs_route": "11/16"}})
    _w(d, "p109_quality_clean.json",
       {"delta_pct": 0.2, "ppl_fp8_tp4": 1.35, "ppl_packed_tp4": 1.3527})
    _w(d, "q0_sweep.json", {
        "hbm_saving": {},
        "verdict": {"optimal_config": {"c4": "had64_INT6", "c128": "had64_INT4"},
                    "memory_weighted": {"compressed_pool_saving": 0.18,
                                        "weights": "c4=21/4, c128=20/128"}}})
    _w(d, "q1_packed_int4.json", {
        "roundtrip": {"rel_l2_mean": 0.07, "rel_l2_p95": 0.09, "cov_W_le_4": 1.0,
                      "n_entries": 10},
        "pool_bytes_gpu": {"measured_saving": 0.38, "format_saving_nominal": 0.38,
                           "fp8_pool_MiB": 36.5, "int4_pool_MiB": 22.5,
                           "n_tokens": 100},
        "unpack_bench": {"ratio_int4_over_fp8": 4.5}})
    lp_docs = [{"doc": "a.txt", "n_tok": 100, "sum_lp": -50.0}]
    _w(d, "p111_lp_base.json", {
        "docs": lp_docs,
        "manifest": {"arm": "base", "code": "fixturehash",
                     "witcert_env": {},
                     "adapters_status": [],
                     "packed_kernel": {"n_packed_c4": 0,
                                       "n_packed_c128": 0}}})
    _w(d, "p111_lp_packed.json", {          # delta = 0 → 过红线
        "docs": lp_docs,
        "manifest": {"arm": "packed", "code": "fixturehash",
                     "witcert_env": {"WITCERT_PACKED_C4": "1",
                                     "WITCERT_PACKED_C128": "1"},
                     "adapters_status": ["packed 已注入"],
                     "packed_kernel": {"n_packed_c4": 7,
                                       "n_packed_c128": 7}}})
    for spec in fg.FUTURE.values():
        _w(d, spec["file"], {"gate": {"passed": True}, "summary": "fixture"})
    _w(d, "p110_sixpath.json",
       {"summary": {k: "PASS" for k in fg.SIXPATH_ITEMS}, "matrix": {}})
    # ruler_flagship 双臂(6c59cc4 判定改解码侧后夹具一直缺失 → T2 恒
    # PENDING 的存量债):base + p124c 生产 packed 臂,解码侧 rel 合法
    _w(d, "p118_ruler_base_off.json", {"acc": 1.0, "n": 6, "n_error": 0})
    _w(d, "p124c_conc_inval_on.json", {
        "acc": 1.0, "n": 6, "n_error": 0,
        "snapshot": {"all_ranks": {"n_repack": 10, "n_invalidate": 5,
                                   "rel_max": 0.1},
                     "decode_shadow_all_ranks": {
                         "c4": {"n_checked": 10, "rel_mean": 0.067,
                                "rel_max": 0.2}}}})


def t1_structure_real():
    rep = fg.build_report()          # 真实 experiments/out/
    n = {"PASS": 0, "FAIL": 0, "PENDING": 0}
    pend_keys = set()
    for lid in ("A", "B", "C", "D"):
        assert lid in rep["layers"], "缺层 %s" % lid
        items = rep["layers"][lid]["items"]
        assert items, "层 %s 空" % lid
        for name, it in items.items():
            for k in ("value", "source_file", "gate", "caliber"):
                assert k in it, "[%s/%s] 缺字段 %s" % (lid, name, k)
            assert it["gate"] in GATES, "[%s/%s] 非法 gate %r" % (lid, name, it["gate"])
            assert it["caliber"], "[%s/%s] caliber 为空" % (lid, name)
            if it["gate"] == "PENDING":
                assert it.get("how_to_obtain"), \
                    "[%s/%s] PENDING 必带 how_to_obtain" % (lid, name)
                pend_keys.add((lid, name))
            n[it["gate"]] += 1
    tp = {(p["layer"], p["item"]) for p in rep["telemetry_pending"]}
    assert tp == pend_keys, "telemetry_pending 与 PENDING 项不一致:%s vs %s" % (tp, pend_keys)
    for p in rep["telemetry_pending"]:
        for k in ("expected_file", "claim", "how_to_obtain"):
            assert p.get(k), "telemetry_pending 缺 %s:%s" % (k, p)
    s = rep["summary"]
    assert (s["n_pass"], s["n_fail"], s["n_pending"]) == (n["PASS"], n["FAIL"], n["PENDING"])
    want = "FAIL" if n["FAIL"] else ("PENDING" if n["PENDING"] else "PASS")
    assert s["overall"] == want and rep["exit_code"] == {"PASS": 0, "FAIL": 1,
                                                         "PENDING": 2}[want]
    assert rep["generated_by"] == fg.GENERATED_BY
    print("T1 真实产物结构完整:PASS %d / FAIL %d / PENDING %d,overall=%s"
          % (n["PASS"], n["FAIL"], n["PENDING"], s["overall"]))


def _run_main(argv):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fg.main(argv)
    except SystemExit as e:
        return e.code, buf.getvalue()
    raise AssertionError("main 未退出")


def t2_exit0(d):
    make_fixture(d)
    rep = fg.build_report(d)
    bad = [(l, k, it["gate"]) for l, lay in rep["layers"].items()
           for k, it in lay["items"].items() if it["gate"] != "PASS"]
    assert not bad, "fixture 应全 PASS,非 PASS 项:%s" % bad
    assert rep["summary"]["overall"] == "PASS" and rep["exit_code"] == 0
    code, _ = _run_main(["--out-dir", d])
    assert code == 0, "全绿 fixture 退出码应为 0,得 %s" % code
    print("T2 全绿 fixture:overall=PASS,exit 0(%d 项)" % rep["summary"]["n_pass"])


def t3_exit1(d):
    make_fixture(d)
    p = json.load(open(os.path.join(d, "p99_concurrent_identity.json")))
    p["gate"]["criteria"]["foreign_reads==0"] = False
    _w(d, "p99_concurrent_identity.json", p)
    rep = fg.build_report(d)
    assert rep["layers"]["A"]["items"]["identity_isolation"]["gate"] == "FAIL"
    assert rep["summary"]["overall"] == "FAIL" and rep["exit_code"] == 1
    code, _ = _run_main(["--out-dir", d])
    assert code == 1, "判据翻转退出码应为 1,得 %s" % code
    # 损坏产物 = FAIL,不许静默降级 PENDING
    open(os.path.join(d, "q1_packed_int4.json"), "w").write("{broken")
    rep = fg.build_report(d)
    assert rep["layers"]["B"]["items"]["q1_roundtrip"]["gate"] == "FAIL"
    assert rep["layers"]["C"]["items"]["pool_bytes_gpu"]["gate"] == "FAIL"
    print("T3 判据翻转/损坏产物:overall=FAIL,exit 1")


def t4_exit2(d):
    make_fixture(d)
    os.remove(os.path.join(d, "p110_sixpath.json"))
    for spec in fg.FUTURE.values():
        os.remove(os.path.join(d, spec["file"]))
    rep = fg.build_report(d)
    assert rep["summary"]["n_fail"] == 0 and rep["summary"]["n_pending"] > 0
    assert rep["summary"]["overall"] == "PENDING" and rep["exit_code"] == 2
    # 六路径逐项 PENDING,pd 的获取方式必须点明双机拓扑
    for k in fg.SIXPATH_ITEMS:
        assert rep["layers"]["D"]["items"][k]["gate"] == "PENDING"
    assert "双机" in rep["layers"]["D"]["items"]["pd"]["how_to_obtain"]
    code, _ = _run_main(["--out-dir", d])
    assert code == 2, "无 FAIL 有 PENDING 退出码应为 2,得 %s" % code
    print("T4 缺产物 fixture:overall=PENDING,exit 2(%d 项 PENDING)"
          % rep["summary"]["n_pending"])


def t5_generated_by_guard(d):
    make_fixture(d)
    _w(d, "flagship_gate.json", {"generated_by": "someone_else.py"})
    rep = fg.build_report(d)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert not fg.write_json(rep, d), "应拒绝覆盖他人产物"
    assert "拒绝覆盖" in buf.getvalue()
    old = json.load(open(os.path.join(d, "flagship_gate.json")))
    assert old["generated_by"] == "someone_else.py", "拒绝后文件不应被动过"
    code, _ = _run_main(["--out-dir", d, "--json"])
    assert code == 125, "--json 撞他人产物退出码应为 125,得 %s" % code
    with contextlib.redirect_stdout(io.StringIO()):
        assert fg.write_json(rep, d, force=True), "--force 应放行"
    new = json.load(open(os.path.join(d, "flagship_gate.json")))
    assert new["generated_by"] == fg.GENERATED_BY
    with contextlib.redirect_stdout(io.StringIO()):
        assert fg.write_json(rep, d), "自己的产物应可直接覆盖"
    print("T5 generated_by 护栏:拒绝他人产物(exit 125)/ --force 放行 / 自产可覆盖")


def main():
    t1_structure_real()
    t_activation_semantics()
    for t in (t2_exit0, t3_exit1, t4_exit2, t5_generated_by_guard):
        d = tempfile.mkdtemp(prefix="flagship_gate_test_")
        try:
            t(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("ALL FLAGSHIP GATE TESTS PASSED")


def t_activation_semantics():
    """激活门 v2 语义(评审 P1):缺失计数不当零。"""
    docs = [{"doc": "a", "n_tok": 10, "sum_lp": -5.0}]
    def arm(code="h", counts=(0, 0), env=None, ads=None):
        pk = {}
        if counts is not None:
            pk = {"n_packed_c4": counts[0], "n_packed_c128": counts[1]}
        return {"docs": docs, "manifest": {
            "code": code, "packed_kernel": pk,
            "witcert_env": env or {}, "adapters_status": ads or []}}
    packed = arm(counts=(7, 7))
    # 显式零 → True
    assert fg._lp_activation(arm(counts=(0, 0)), packed) is True
    # 缺计数 + 缺席证明 → True
    assert fg._lp_activation(arm(counts=None), packed) is True
    # 缺计数 + env 含 packed 开关 → None(仪器未装载嫌疑)
    assert fg._lp_activation(
        arm(counts=None, env={"WITCERT_PACKED_C4": "1"}), packed) is None
    # code unset → None
    assert fg._lp_activation(arm(counts=(0, 0), code="unset"), packed) is None
    # 旧 schema → None
    assert fg._lp_activation(docs, docs) is None
    print("  激活门语义 5 例通过")


if __name__ == "__main__":
    main()
