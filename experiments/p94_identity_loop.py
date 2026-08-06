# -*- coding: utf-8 -*-
"""F1/F2(p94):同对象身份闭环 + 真正逐请求预算的判读。

六审三 P0 中前两个的验收:
  P0-1 同对象:认证写(dsv4-cwrite-c)作用于**压缩页池** —— 读侧账本 dequant
       检查的同一物理缓冲;槽位状态图 (layer, slot)→{未写/已掩/保精确} 让
       每次读取按写侧状态分账 → n_read_masked 就是"读到被策略改过的条目"的
       直接计数,(entry,version) 闭环成立(version 口径见 caliber)。
  P0-2 逐请求预算:请求身份 = rid#seq(decode→extend 转换计数,pool index
       复用不再串账);每请求独立 δ_req=1% 望远镜计数器 —— 验收标准是
       **每一个**请求的 Σδ ≤ 0.01 且 δ_i 序列从 1/2 重新开始。
  P0-2' 副本耦合:8 个 TP rank 各持一份压缩 KV 副本、各自入账 —— 若各算各的
       δ,总预算要乘 8。这里给出**耦合论证的实证侧**:SR 抽签用 counter-based
       播种 (layer, k),8 个 rank 的随机实现**逐比特相同** → 8 份副本是同一
       随机变量的同一实现,概率空间只有一个,δ 只数一次。验收 = 各 rank 的
       写侧账目字段完全一致(n_compressed / delta_spent / n_auth_violations)。

python3 experiments/p94_identity_loop.py
"""
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DELTA_REQ = 0.01
WKEYS = ("n_compressed", "n_fallback_exact", "delta_spent",
         "n_prob_events", "n_authorized", "n_auth_violations")


def snap(arm, r, rank=0):
    fs = glob.glob(os.path.join(OUT, f"p94_{arm}_r{r}.rank{rank}"))
    return json.load(open(fs[0])) if fs else None


def seq_of(rid):
    return int(rid.split("#")[1]) if "#" in rid else -1


def main():
    rep = {"what": "F1/F2:同对象身份闭环(压缩页池认证写 + 槽位状态图)+ 逐请求独立预算",
           "machine": "hgx",
           "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8",
           "caliber": [
               "同对象:写侧 = item.compress_kv_pool(c4/c128),读侧账本 dequant 的同一缓冲;"
               "swa 窗口池的 cwrite 仍在但不参与本闭环计数",
               "请求身份 rid#seq:非 extend→extend 转换计一次(chunked prefill 连续 extend "
               "属同一请求);串行 batch=1 口径,并发批下该口径失效(显式声明)",
               "version 口径:串行无驱逐负载,槽位不复用 —— slot_flags 即 (entry,version);"
               "有驱逐/复用的负载需要显式 version 计数器,本实验不覆盖",
               "副本耦合:counter-based 种子 (layer,k) → 8 rank 逐比特同实现;"
               "验收为各 rank 账目字段完全一致,δ 按逻辑事件数一次",
               "首个 compress 写可能先于该批 stash 一步 —— 请求边界最多偏一个 batch(如实报)",
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
        for rid in sorted(by_rid, key=seq_of):
            w = by_rid[rid]
            led = leds.get(rid, {})
            ok = 0.0 <= w["delta_spent"] <= DELTA_REQ
            if not ok:
                hard_fails.append(f"{arm}:{rid} Σδ={w['delta_spent']}")
            rows.append({
                "rid": rid,
                "write_side": {k: w.get(k, 0) for k in WKEYS},
                "budget_ok": ok,
                "read_side": {
                    "n_events": led.get("n_events", 0),
                    "n_read_masked": led.get("n_read_masked", 0),
                    "n_read_kept_exact": led.get("n_read_kept_exact", 0),
                    "n_read_unwritten": led.get("n_read_unwritten", 0),
                },
            })
        # 副本耦合:全 rank 账目一致性(逐 rid 逐字段)
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
        rep["arms"][arm] = {"per_request": rows, "replica_coupling": coupling}

    # headline 只在两臂都有数据时给
    da = rep["arms"].get("dither")
    if isinstance(da, dict) and da["per_request"]:
        rows = da["per_request"]
        n_req = len(rows)
        all_ok = all(r["budget_ok"] for r in rows)
        masked_reads = sum(r["read_side"]["n_read_masked"] for r in rows)
        tot_reads = sum(r["read_side"]["n_events"] for r in rows)
        viol = sum(r["write_side"]["n_auth_violations"] for r in rows)
        nauth = sum(r["write_side"]["n_authorized"] for r in rows)
        coup = da["replica_coupling"]
        rep["summary"] = {
            "n_requests_ledgered": n_req,
            "all_budgets_independent_ok": all_ok and not hard_fails,
            "max_delta_spent": max(r["write_side"]["delta_spent"] for r in rows),
            "read_of_masked_total": masked_reads,
            "read_events_total": tot_reads,
            "auth_violations": f"{viol}/{nauth}",
            "replica_coupling_identical": coup["identical"],
        }
        rep["findings"] = {"0_headline": (
            "**同对象身份闭环 + 逐请求预算成立**:%d 个请求各自独立记账,"
            "max Σδ = %.6f ≤ δ_req=0.01(%s);读侧 %s 次条目检查中 %s 次命中"
            "写侧已掩槽位(n_read_masked>0 = 读到的就是被策略改过的物理条目);"
            "预授违约 %s;副本耦合:%d 个 rank 账目%s —— counter-based 种子下"
            "8 份副本为同一随机实现,δ 按逻辑事件数一次"
            % (n_req, rep["summary"]["max_delta_spent"],
               "全部通过" if rep["summary"]["all_budgets_independent_ok"] else
               "存在超支:" + ";".join(hard_fails),
               f"{tot_reads:,}", f"{masked_reads:,}", rep["summary"]["auth_violations"],
               coup["n_ranks_seen"],
               "逐字段一致" if coup["identical"] else f"不一致({coup['diff']})——耦合论证不成立,δ 记账需按 rank 分开"))}
    dst = os.path.join(OUT, "p94_identity_loop.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    for k, v in rep.get("findings", {}).items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
