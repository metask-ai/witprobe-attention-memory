# 论文2:异构注意力记忆的运行时可观测性

**唯一主张**:现代模型的"注意力记忆"已不再等同于普通 KV Cache。我们给出一套统一的
运行时观测契约,让连续缓存、latent cache、离散稀疏选择和递归状态都能被在线测量、
验证和定位。

候选标题:
- Beyond the KV Cache: Runtime Observability for Heterogeneous Attention Memory
- Runtime Contracts for Heterogeneous Attention Memory(走契约演算路线时)

**主轴是四类记忆对象,不是模型名单**:

| 记忆对象 | 代表模型 | 对应尺子 | 认证强度 |
|---|---|---|---|
| Dense KV | Qwen2.5-7B、Llama-3.1-8B | 残差带范数 → TV 风险 | sound |
| Latent KV | GLM-5.2、DeepSeek-V4-Flash | latent 残差见证 | sound |
| Sparse Selector | DSA(token 级)、C4(页级) | top-k margin 稳定性 | 条件认证 |
| Recurrent State | Kimi-Linear / K3 | 收缩因子与误差半衰期 | 观测(论文3 转证书) |

## 与另外两篇的边界

- **论文1(p1-kv-certificates)现在应冻结**:普通 KV 量化的 sound TV 证书 + 门控 + 修复。
  R8 的内容不要再塞进它的 arXiv v2,否则会冲散第一篇清晰的数学主线。
- **论文3(p3-recurrent-state)**:KDA 递归状态的完整误差传播证书。需要逐通道递归界、
  真实扰动传播、输出侧闭环三者齐备才独立成篇;在本篇里 KDA 只作为一种已支持的
  记忆对象报告收缩观测。

## 现有素材(全部已过验收门,数字见 canon 中 paper=2 的条目)

- 平台覆盖矩阵 `experiments/out/p61_platform_matrix.json`:7 条探针行 / 5 个唯一模型 / 4 个架构族
- 三路归因 `experiments/out/p55_v4flash_attribution.json`:存储 sound / 选择条件认证 / 池化经验
- 开销预算 `experiments/out/p60_probe_overhead.json`:EVERY=16 最不利 +5.45%
- latent 数学 `p50` / GLM 在线 `p51` / 坏块哨兵 `p52` / DSA 选择 `p53` / V4 选择 `p58` / KDA `p56`

## 已知欠账(写作时必须如实标注或补齐)

1. **哨兵检出是抽样受限的**:注入 4 个坏槽位实测检出 2 个,不能写成 4/4。需补检出率随
   抽样率的曲线、time-to-detect 分布,以及超几何闭式给出的检测延迟 SLA(TinyKG 10496)。
2. **开销口径不够顶会**:现为单请求、串行、关图下的相对增量。需并发 serving 的吞吐与
   P50/P99,以及 graph-safe 或异步遥测路径,否则会被判为研究探针而非常开生产观测层。
3. **池化侧没有 sound 契约**:V4 的链条 池化→选择→存储 里,池化是纯经验的。若能推出
   sound 界则整条异构链首次端到端 certified;推不出则如实报告"认证到池化边界为止",
   把它作为现代压缩架构的结构性认证断点(TinyKG 10499)。

## 写作顺序

先 `math_spec.md`(只定义对象、假设、计划证明的定理),且**第一节就写 V4 的 U/S/R 三算子
与端到端组合尝试** —— 让数学立刻接触数据。算不出端到端数字,就说明抽象层次还不对。
