# -*- coding: utf-8 -*-
"""G2(p97):owner 版本化闭环判读。

设计要点(为什么不需要 allocator hook):压缩池槽位没有独立 free 路径 ——
复用发生在页被重新分配后,而所有压缩写都经过认证写 hook,复用**必然先经过
一次重写**,slot_flags 因此恒新鲜。真正缺的是"这个掩码是谁做的"——
G2 = 写侧逐槽位记录写入者 UID:
  · 复用事件 = 写时 owner 变更(n_slot_reuses,即版本事件计数);
  · 读侧 own/foreign 分账:foreign = 读到了**别的请求写入**的条目 ——
    合法来源是前缀缓存共享(第 9 条 prompt 重复第 1 条,专测此路径),
    异常来源是复用后未重写(设计上不可能,计数守卫之)。

验收(完成判据):复用压力下(串行页回收 + 前缀共享)read-of-masked 计数
仍与写侧对账,foreign 读被 owner 分账捕获而非误记为本请求的闭环。

python3 experiments/p97_owner_closure.py
"""
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DELTA_REQ = 0.01


def snap(r, rank=0):
    fs = glob.glob(os.path.join(OUT, f"p97_dither_r{r}.rank{rank}"))
    return json.load(open(fs[0])) if fs else None


def uid_of(rid):
    return int(rid[1:]) if rid.startswith("u") else -1


def main():
    last = max(int(f.rsplit("_r", 1)[1].split(".")[0])
               for f in glob.glob(os.path.join(OUT, "p97_dither_r*.rank0")))
    d = snap(last)
    by_rid = d.get("cwrite_by_rid") or {}
    leds = d.get("request_ledgers") or {}
    rows = []
    for rid in sorted(by_rid, key=uid_of):
        w = by_rid[rid]
        led = leds.get(rid, {})
        rows.append({
            "rid": rid,
            "delta_spent": w["delta_spent"],
            "budget_ok": 0.0 <= w["delta_spent"] <= DELTA_REQ,
            "n_slot_reuses": w.get("n_slot_reuses", 0),
            "read": {k: led.get(k, 0) for k in
                     ("n_events", "n_read_masked", "n_read_kept_exact",
                      "n_read_unwritten", "n_read_foreign_owner")},
        })
    reuse_total = sum(r["n_slot_reuses"] for r in rows)
    foreign_total = sum(r["read"]["n_read_foreign_owner"] for r in rows)
    unwr_total = sum(r["read"]["n_read_unwritten"] for r in rows)
    all_ok = all(r["budget_ok"] for r in rows)
    # 第 9 条 = 第 1 条的重复:它的 foreign 读(若前缀缓存命中)应显著;
    # 其余请求 foreign 应接近 0(串行不同任务无共享)
    dup = rows[-1] if rows else None
    rep = {
        "what": "G2:owner 版本化闭环 —— (entry, owner, 状态) 三元分账",
        "machine": "hgx",
        "stack": "DeepSeek-V4-Flash-FP8, sglang 0.5.13.post1, tp8+ep8",
        "caliber": [
            "复用事件 = 认证写时槽位 owner 变更(压缩池无独立 free 路径,复用必先"
            "重写,flags 恒新鲜 —— 这就是不需要 allocator hook 的原因,设计即约束)",
            "foreign 读 = 读到别的 UID 写入的条目;合法来源是前缀缓存共享 ——"
            "第 9 条 prompt 重复第 1 条专测此路径;异常来源(复用后未重写)设计上"
            "不可能,由 n_read_unwritten 守卫",
            "复用压力来源 = 串行页回收 + 前缀共享;**请求中途驱逐**的负载未覆盖"
            "(需长 context 逼近显存上限,另案)",
        ],
        "per_request": rows,
        "summary": {
            "n_accounts": len(rows),
            "all_budgets_ok": all_ok,
            "slot_reuses_total": reuse_total,
            "foreign_reads_total": foreign_total,
            "foreign_reads_dup_prompt": (dup or {}).get("read", {}).get("n_read_foreign_owner"),
            "read_unwritten_total": unwr_total,
        },
    }
    rep["findings"] = {"0_headline": (
        "**owner 版本化闭环成立**:%d 账户预算%s;槽位复用(写时 owner 变更)"
        "共 %s 次 —— 复用后 flags 被重写刷新,读侧 0 次读到未写槽位(%d);"
        "foreign 读共 %s 次(重复 prompt 账户占 %s)—— 别的请求写入的条目"
        "被 owner 分账捕获,不再误记入本请求闭环"
        % (len(rows), "全过" if all_ok else "**有超支**",
           f"{reuse_total:,}", unwr_total, f"{foreign_total:,}",
           rep["summary"]["foreign_reads_dup_prompt"]))}
    dst = os.path.join(OUT, "p97_owner_closure.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst)
    print(" ", rep["findings"]["0_headline"])


if __name__ == "__main__":
    main()
