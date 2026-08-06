# -*- coding: utf-8 -*-
"""R9-B3:graph-safe 遥测的判据产物。由 p71_graph_replay.sh 在远端调用。

**为什么这份产物必须存在**:B3 之前被我判成 BLOCKED,理由是"CUDA graph 捕获失败"。
那个判断是错的 —— 缺的不是设计,是**对照实验**。p70 的三档归因阶梯(裸栈 / 打补丁探针关 /
打补丁探针开,其余命令行完全相同)证明三档都能起服并完成捕获,探针档捕获耗时与显存
显著更高,说明探针算子确实被捕进了图。本文件给出比"能起服"强得多的判据。

判据(graph-safe 与否的分水岭):

    CUDA graph 重放**不执行任何 Python**。探针的 python 函数体只在 capture 时跑过一次。
    因此在 decode(唯一走图的阶段):
      · n_calls(python 侧计数)只应增长 **prefill** 那部分 = 请求数 × 层数
      · 设备张量累加器 n(被捕进图、每次重放 in-place 累加)应增长
        **prefill + 全部 decode 步** = 请求数 × 层数 × (1 + 输出 token 数)

    两者都对上,才叫"重放期在累加";只看"服务起来了"会把静默降级当成成功。

口径(随数字引用):
  · 单卡 Qwen2.5-7B + gqa-kv 单适配器;多适配器/多路径未验。
  · 判定的是**累加器在重放期继续增长**,不是"带图下的开销"—— 论文里的开销数字仍是
    关图口径实测,不因本文件改写(带图开销要另测,见 caliber 末条)。
  · flusher 延迟 60s 上岗以避开起服捕获窗口;取快照前必须等过这一窗口,
    否则会把"还没冲刷"误读成"探针没跑"(p70 C 档就是这么误判的)。

env: S1/S2/S3="OK <n_calls> <元素数> <槽位>", NREQ/NTOK/NLAYER
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def parse(env_name):
    parts = os.environ.get(env_name, "").split()
    if len(parts) != 4 or parts[0] != "OK":
        raise SystemExit(f"{env_name} 不可用: {os.environ.get(env_name)!r}")
    return {"n_calls": int(parts[1]), "elements": int(parts[2]), "slots": int(parts[3])}


def main():
    s1, s2, s3 = parse("S1"), parse("S2"), parse("S3")
    nreq = int(os.environ["NREQ"]); ntok = int(os.environ["NTOK"])
    nlayer = int(os.environ["NLAYER"])
    every = int(os.environ.get("EVERY", "1"))

    d_calls = s2["n_calls"] - s1["n_calls"]
    d_elems = s2["elements"] - s1["elements"]
    # 预期:python 只在 prefill 跑;设备累加器 prefill + decode 每步都跑
    exp_calls = nreq * nlayer
    exp_decode = nreq * ntok * nlayer
    exp_elems = exp_decode + exp_calls

    rep = {
        "what": "graph-safe 遥测:CUDA graph 重放期探针累加器是否继续增长",
        "sample_every": every,
        "machine": "hgx",
        "stack": ("Qwen2.5-7B-Instruct, sglang 0.5.13.post1, tp1, ctx 8192, "
                  "**未加 --disable-cuda-graph / --disable-piecewise-cuda-graph**"),
        "caliber": [
            "单卡单适配器(gqa-kv);多适配器与多路径共存下未验",
            "判定的是累加器在重放期继续增长,不是带图下的开销",
            "带图开销未测:论文中的开销数字仍是关图口径,不因本文件改写",
            "取快照必须等过 flusher 的 60s 延迟窗口,否则会把未冲刷误读成未运行",
        ],
        "attribution_ladder": {
            "note": ("p70:三档除单一变量外命令行完全相同。此前判 BLOCKED 是因为"
                     "**没做对照**,不是因为设计不足"),
            "A_bare_stack": "UP(捕获 4.17s / 0.37GB)",
            "B_patched_probe_off": "UP(捕获 4.03s / 0.37GB)",
            "C_patched_probe_on": "UP(捕获 7.58s / 0.62GB)",
            "reading": ("三档全过 -> 捕获不再失败;C 档捕获耗时与显存显著高于 A/B,"
                        "说明探针算子被捕进了图,而不是被静默跳过"),
        },
        "snapshots": {"s1_after_warmup": s1, "s2_after_load": s2, "s3_after_idle": s3},
        "workload_between_s1_s2": {"requests": nreq, "new_tokens_each": ntok,
                                   "layers": nlayer},
        "replay_test": {
            "delta_n_calls": d_calls, "expected_n_calls_prefill_only": exp_calls,
            "delta_elements": d_elems, "expected_elements": exp_elems,
            "expected_decode_replays": exp_decode,
            "calls_match": d_calls == exp_calls,
            "elements_match": d_elems == exp_elems,
            "idle_stable": s3 == s2,
        },
    }
    rt = rep["replay_test"]
    # **EVERY>1 时预期值不适用**:采样门是 capture 时求值一次的 python 分支,
    # 哪几张图里带探针取决于捕获当刻的计数器状态,不是 1/N 的确定关系。
    # 故此时判据退化为方向性的:**元素数增长而 n_calls 不增长 = 重放期仍在累加**。
    rt["replay_growth"] = d_elems - d_calls          # 只能来自重放
    if every == 1:
        rep["verdict"] = ("PASS" if (rt["calls_match"] and rt["elements_match"]
                                     and rt["idle_stable"]) else "FAIL")
    else:
        rep["verdict"] = ("PASS_DIRECTIONAL" if (rt["replay_growth"] > 0
                                                 and rt["idle_stable"]) else "NO_REPLAY_COVERAGE")
        rep["every_gt_1_note"] = (
            "EVERY=%d:精确预期不适用 —— 采样门在 capture 时求值一次,哪几张图带探针"
            "取决于捕获当刻的计数器状态。判据退化为方向性:重放增量 = Δ元素 - Δn_calls = %d"
            % (every, rt["replay_growth"]))
    rep["findings"] = {
        "0_headline": (
            "**graph-safe 成立**:带图运行下,python 侧 n_calls 只增长 %d(= %d 请求 × %d 层,"
            "即 prefill 那一次),而设备累加器增长 %d(= %d 次 decode 重放 + %d 次 prefill)。"
            "CUDA graph 重放不执行任何 Python,故元素数的增长**只能来自被捕进图、"
            "在重放期 in-place 累加的探针算子**。静置 20s 后快照不变,排除冲刷滞后造成的假增长。"
            % (d_calls, nreq, nlayer, d_elems, exp_decode, exp_calls)
            if rep["verdict"] == "PASS" else
            "判据未通过:Δn_calls=%d(预期 %d), Δ元素=%d(预期 %d), 静置稳定=%s"
            % (d_calls, exp_calls, d_elems, exp_elems, rt["idle_stable"])),
        "1_four_fixes": (
            "四项修复缺一不可,且每一项都是被这条路径逼出来的:"
            "①计数必须是设备张量 —— python int 会冻在 capture 值而 sum 继续长,均值被系统性算大;"
            "②探针路径不能有 RNG —— capture 期间会使流捕获失效;"
            "③flusher 必须避开捕获窗口 —— 回读累加器是 device-host 同步;"
            "④累加器不能在 capture 期分配 —— serving 每个 batch size 一张图,"
            "在图 A 的池里分配的累加器从图 B 访问即失效"),
        "2_what_this_does_not_show": (
            "**没有测带图下的开销**。论文报的 EVERY=64 档吞吐/尾延迟仍是关图口径实测,"
            "本文件不改写那些数字;带图开销需另起一轮同口径测量"),
        "3_methodology": (
            "此前把 B3 判成 BLOCKED,理由是'捕获失败且未能归因'。归因不了不等于修不好 —— "
            "**该做的是把变量拆开做对照(p70 三档),而不是继续打补丁,也不是就地收口**"),
    }
    # EVERY=1 那份是论文引用的产物,**不能被别的档位覆盖**(撤回残留是老陷阱)
    dst = os.path.join(OUT_DIR, "p71_graph_safe.json" if every == 1
                       else f"p71_graph_safe_e{every}.json")
    json.dump(rep, open(dst, "w"), ensure_ascii=False, indent=1)
    print("写出", dst, "->", rep["verdict"])
    for k, v in rep["findings"].items():
        print(" ", k, ":", v)


if __name__ == "__main__":
    main()
