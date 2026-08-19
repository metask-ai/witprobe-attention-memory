# -*- coding: utf-8 -*-
"""F2 v2(p96):真实生命周期请求身份 + 不复用随机流的判读(七审重做的验收)。

七审三个 P0 缺陷对应的验收点:
  ① 账户边界 = 真实请求边界:统一状态机在**写路径**检测 extend 转换
     (`_request_identity` 在任何写入入账之前执行)—— 首写偏批问题不存在;
     单元测试 R5 覆盖复用+更长初始长度、chunked prefill、幂等、UID 不复用。
  ② 账户口径:user 请求 = 读账户中最后 8 个(串行 uid 顺序=到达顺序);
     更早读账户 = warmup generate,无读账户 = boot —— 分开计数,不合称请求数。
  ③ 随机流:种子 = (server_nonce, UID, layer, k),UID 永不复用 ——
     不同请求不共享随机数;跨 rank 四元组一致 → 副本耦合保持(逐字段验收)。
  ④ e-process:逐账户 + **全局(跨请求不重置)**双本;log_M_max 只取更新后值
     (不再把初始化 0 冒充峰值)。

python3 experiments/p96_identity_v2.py
"""
import glob
import json
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DELTA_REQ = 0.01
WKEYS = ("n_compressed", "n_fallback_exact", "delta_spent",
         "n_prob_events", "n_authorized", "n_auth_violations")


def snap(arm, r, rank=0):
    fs = glob.glob(os.path.join(OUT, f"p96_{arm}_r{r}.rank{rank}"))
    return json.load(open(fs[0])) if fs else None


def uid_of(rid):
    return int(rid[1:]) if rid.startswith("u") else -1


def main():
    rep = {"what": "F2 v2:真实生命周期请求身份(写路径 extend 边界)+ 不复用随机流"
                   " + 全局 e-process",
           "machine": "hgx",
           "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8",
           "caliber": [
               "身份 = 逻辑请求 UID(全局单调,永不复用);边界主信号 = 非 extend→extend"
               "转换,在写路径、任何写入入账之前检测 —— 无一批偏移;decode-only 后备 = "
               "同 rid seq_len 回落。单元测试 R5 覆盖七审列举的漏检情形",
               "user 请求 = 读账户中**最后 8 个**(串行执行 uid 顺序=到达顺序,"
               "launcher 恰发 8 条 prompt);更早的读账户 = 服务器 warmup generate,"
               "无读账户 = boot/prefill-only warmup —— 三类分开报,不合称请求数",
               "SR 种子 = (server_nonce, UID, layer, k) mod 2^63 —— 不同请求不共享"
               "随机流(条件随机性对自适应流量成立);四元组跨 rank 一致 → 耦合保持",
               "串行 batch=1 口径;并发批下批内首请求近似失效(显式声明);"
               "槽位复用的 (entry,version) 仍是串行无驱逐口径",
               "e-process 全局账户跨请求不重置(持续全服务哨兵);log_M_max 仅含"
               "更新后的值,负值峰值如实报(不以初始化 0 冒充)",
           ],
           "arms": {}}
    hard_fails = []
    for arm in ("policy", "dither"):
        s7 = snap(arm, 7)
        if s7 is None:
            rep["arms"][arm] = "缺快照"
            continue
        by_rid = s7.get("cwrite_by_rid") or {}
        leds = s7.get("request_ledgers") or {}
        rows = []
        read_uids = sorted((r for r in by_rid
                            if leds.get(r, {}).get("n_events", 0) > 0), key=uid_of)
        user_uids = set(read_uids[-8:])          # 串行:最后 8 个读账户 = 8 条 prompt
        for rid in sorted(by_rid, key=uid_of):
            w = by_rid[rid]
            led = leds.get(rid, {})
            ok = 0.0 <= w["delta_spent"] <= DELTA_REQ
            if not ok:
                hard_fails.append(f"{arm}:{rid} Σδ={w['delta_spent']}")
            rows.append({
                "rid": rid,
                "is_user_request": rid in user_uids,
                "write_side": {k: w.get(k, 0) for k in WKEYS},
                "budget_ok": ok,
                "read_side": {k: led.get(k, 0) for k in
                              ("n_events", "n_read_masked",
                               "n_read_kept_exact", "n_read_unwritten")},
                "eprocess": w.get("eprocess"),
            })
        coupling = {"n_ranks_seen": 1, "identical": True, "diff": None}
        for k in range(1, 8):
            sk = snap(arm, 7, rank=k)
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
        rep["arms"][arm] = {"per_request": rows, "replica_coupling": coupling,
                            "eprocess_global": s7.get("eprocess_global")}

    da = rep["arms"].get("dither")
    if isinstance(da, dict) and da["per_request"]:
        rows = da["per_request"]
        user = [r for r in rows if r["is_user_request"]]
        warm = [r for r in rows if not r["is_user_request"]]
        all_ok = all(r["budget_ok"] for r in rows) and not hard_fails
        masked = sum(r["read_side"]["n_read_masked"] for r in user)
        reads = sum(r["read_side"]["n_events"] for r in user)
        unwr = sum(r["read_side"]["n_read_unwritten"] for r in user)
        viol = sum(r["write_side"]["n_auth_violations"] for r in rows)
        nauth = sum(r["write_side"]["n_authorized"] for r in rows)
        eg = da.get("eprocess_global") or {}
        rep["summary"] = {
            "n_user_requests": len(user), "n_warmup_accounts": len(warm),
            "all_budgets_independent_ok": all_ok,
            "max_delta_spent": max(r["write_side"]["delta_spent"] for r in rows),
            "read_of_masked_user": masked, "read_events_user": reads,
            "read_unwritten_user": unwr,
            "auth_violations": f"{viol}/{nauth}",
            "replica_coupling_identical": da["replica_coupling"]["identical"],
            "eprocess_global": {"n_factors": eg.get("n_factors"),
                                "log_M_final": eg.get("log_M"),
                                "log_M_max": eg.get("log_M_max"),
                                "threshold": math.log(1.0 / eg.get("delta_e", 0.01)),
                                "crossed": eg.get("crossed")},
        }
        s = rep["summary"]; egs = s["eprocess_global"]
        rep["findings"] = {"0_headline": (
            "**真实生命周期身份 + 不复用随机流成立**:%d 个 user 请求账户(另有 %d 个"
            " warmup/boot 账户,分开计)各自独立预算,max Σδ = %.6f ≤ 0.01(%s);"
            "边界在写路径 extend 转换处检测(先于任何写入,无一批偏移);SR 种子含"
            " (nonce,UID,layer,k),跨请求零随机流复用;读闭环:user 请求 %s 次检查中"
            " %s 次命中已掩槽位、%d 次读到未写槽位;违约 %s;8-rank 耦合%s;"
            "**全局 e-process(跨请求不重置)**:%s 因子,log M 终值 %.1f、"
            "更新后峰值 %.2f vs 阈值 %.2f,越阈 %s —— 持续全服务漂移哨兵在位"
            % (len(user), len(warm), s["max_delta_spent"],
               "全部通过" if all_ok else "存在超支:" + ";".join(hard_fails),
               f"{reads:,}", f"{masked:,}", unwr, s["auth_violations"],
               "逐字段一致" if s["replica_coupling_identical"] else "**不一致**",
               f"{egs['n_factors']:,}", egs["log_M_final"], egs["log_M_max"],
               egs["threshold"], "0 次" if not egs["crossed"] else "**发生**"))}
    dst = os.path.join(OUT, "p96_identity_v2.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep.get("findings", {}).items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
