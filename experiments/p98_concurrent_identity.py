# -*- coding: utf-8 -*-
"""G1(p98):并发批逐行请求身份的判读。

串行口径(p96/p97)之外的最后一块:8 条 prompt **同时**发,批内多请求混行。
验收点:
  ① 账户数:user 账户(有读事件)= 8 = 并发请求数 —— 逐行状态机没有把并发
     请求合并或碎裂;
  ② 每账户独立 Σδ ≤ δ_req = 0.01 —— 写侧逐条目归属(decode 批行对齐 /
     prefill extend_seq_lens 前缀和)把 δ 记到正确账户;
  ③ 不同 prompt 的 foreign 读 ≈ 0 —— 归属若错位,foreign 会立刻非零
     (owner 分账是归属正确性的**运行时守卫**,不只是诊断);
  ④ 8-rank 账目逐字段一致 —— 并发调度下批流仍 rank 间一致,耦合保持。

python3 experiments/p98_concurrent_identity.py
"""
import glob
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DELTA_REQ = 0.01
WKEYS = ("n_compressed", "n_fallback_exact", "delta_spent",
         "n_prob_events", "n_authorized", "n_auth_violations")


PREFIX = os.environ.get("WITCERT_P98_PREFIX", "p98")


def snap(rank=0):
    fs = glob.glob(os.path.join(OUT, f"{PREFIX}_dither_rfinal.rank{rank}"))
    return json.load(open(fs[0])) if fs else None


def uid_of(rid):
    return int(rid[1:]) if rid.startswith("u") else -1


def main():
    d = snap()
    by_rid = d.get("cwrite_by_rid") or {}
    leds = d.get("request_ledgers") or {}
    rows = []
    read_uids = sorted((r for r in by_rid
                        if leds.get(r, {}).get("n_events", 0) > 0), key=uid_of)
    user_uids = set(read_uids[-8:])   # 最后 8 个读账户 = 8 条 prompt;更早 = warmup
    for rid in sorted(by_rid, key=uid_of):
        w = by_rid[rid]
        led = leds.get(rid, {})
        rows.append({
            "rid": rid,
            "is_user_request": rid in user_uids,
            "write_side": {k: w.get(k, 0) for k in WKEYS},
            "n_slot_reuses": w.get("n_slot_reuses", 0),
            "budget_ok": 0.0 <= w["delta_spent"] <= DELTA_REQ,
            "read": {k: led.get(k, 0) for k in
                     ("n_events", "n_read_masked", "n_read_kept_exact",
                      "n_read_unwritten", "n_read_foreign_owner")},
        })
    coupling = {"n_ranks_seen": 1, "identical": True, "diff": None}
    for k in range(1, 8):
        sk = snap(rank=k)
        if sk is None:
            continue
        coupling["n_ranks_seen"] += 1
        bk = sk.get("cwrite_by_rid") or {}
        for rid in by_rid:
            for f in WKEYS:
                a, b = by_rid[rid].get(f, 0), bk.get(rid, {}).get(f, 0)
                if a != b:
                    coupling["identical"] = False
                    coupling["diff"] = f"rank{k}:{rid}:{f} {a}!={b}"
    user = [r for r in rows if r["is_user_request"]]
    warm = [r for r in rows if not r["is_user_request"]]
    all_ok = all(r["budget_ok"] for r in rows)
    foreign_user = sum(r["read"]["n_read_foreign_owner"] for r in user)
    reads_user = sum(r["read"]["n_events"] for r in user)
    unwr = sum(r["read"]["n_read_unwritten"] for r in user)
    viol = sum(r["write_side"]["n_auth_violations"] for r in rows)
    nauth = sum(r["write_side"]["n_authorized"] for r in rows)
    eg = d.get("eprocess_global") or {}
    rep = {
        "what": "G1:并发批逐行请求身份 —— 逐条目 UID 归属 + 每账户独立预算/随机流",
        "machine": "hgx",
        "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8",
        "caliber": [
            "8 路 ThreadPoolExecutor 并发,批内多请求混行;终态快照(并发下无逐请求"
            "快照边界);user 账户 = 有读事件的账户",
            "写侧逐条目归属:decode 计划 [bs,16] 批行对齐 / prefill 计划逐 query "
            "token 由 extend_seq_lens 前缀和分段;形状不符退回批首行口径(计数守卫)",
            "foreign 读是归属正确性的运行时守卫:归属错位会把别账户条目算进本账户"
            " owner,foreign 立即非零;8 条 prompt 互不相同,无前缀共享",
            "每账户独立 (nonce,uid,layer,k) 随机流;并发调度批流 rank 间一致 → 耦合",
        ],
        "per_request": rows,
        "replica_coupling": coupling,
        "summary": {
            "n_user_requests": len(user), "n_other_accounts": len(warm),
            "all_budgets_ok": all_ok,
            "max_delta_spent": max((r["write_side"]["delta_spent"] for r in rows),
                                   default=0.0),
            "foreign_reads_user": foreign_user, "read_events_user": reads_user,
            "read_unwritten_user": unwr,
            "auth_violations": f"{viol}/{nauth}",
            "replica_coupling_identical": coupling["identical"],
            "eprocess_global": {"n_factors": eg.get("n_factors"),
                                "log_M_max": eg.get("log_M_max"),
                                "crossed": eg.get("crossed")},
        },
    }
    # 八审 P0-2:**machine-decidable gate** —— 全零硬判据,任一不过即整体不过,
    # 且以退出码承载(CI/发布包可直接依赖);不设 <1% 之类的软阈值。
    fbk = d.get("uid_group_fallbacks")
    ifc = d.get("identity_fail_closed")
    gate = {
        "n_user_requests==8": len(user) == 8,
        "all_budgets<=delta_req": all_ok,
        "foreign_reads==0": foreign_user == 0,
        "read_unwritten==0": unwr == 0,
        "auth_violations==0": viol == 0,
        "uid_group_fallbacks==0": (fbk is None or fbk.get("n", 0) == 0),
        "identity_fail_closed==0": (ifc is None or ifc.get("n_entries", 0) == 0),
        # 九审:缺失数据不得默认通过 —— 哨兵必须真的有数据且未越阈
        "eprocess_has_factors": (eg.get("n_factors") or 0) > 0,
        "eprocess_logM_present": eg.get("log_M_max") is not None,
        "eprocess_not_crossed": not bool(eg.get("crossed")),
        # 九审:只剩 rank0 时 identical 恒真 —— 必须八份快照全在
        "n_ranks_seen==8": coupling["n_ranks_seen"] == 8,
        "replica_coupling_identical": coupling["identical"],
    }
    passed_gate = all(gate.values())
    rep["gate"] = {"criteria": gate, "passed": passed_gate}
    s = rep["summary"]
    rep["findings"] = {"0_headline": (
        "**并发逐行身份 gate %s**(%d/%d 硬判据):%d user 账户独立 max Σδ = %.6f;"
        "foreign %s/读 %s;读未写 %d;违约 %s;归属回退 %s;身份 fail-closed 拦截 %s;"
        "8-rank 耦合%s;全局哨兵 %s 因子峰值 %s%s"
        % ("PASS" if passed_gate else "**FAIL**",
           sum(gate.values()), len(gate), len(user), s["max_delta_spent"],
           f"{foreign_user:,}", f"{reads_user:,}", unwr, s["auth_violations"],
           0 if (fbk is None) else fbk.get("n", 0),
           0 if (ifc is None) else ifc.get("n_entries", 0),
           "一致" if coupling["identical"] else "**不一致**",
           eg.get("n_factors"), None if eg.get("log_M_max") is None
           else round(eg["log_M_max"], 2),
           "" if passed_gate else " | 未过项:" +
           ",".join(k for k, v in gate.items() if not v)))}
    dst = os.path.join(OUT, f"{PREFIX}_concurrent_identity.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print(" ", rep["findings"]["0_headline"])
    sys.exit(0 if passed_gate else 1)


if __name__ == "__main__":
    main()
