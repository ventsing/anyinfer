# AnyInfer Dev 设计基线

> 状态：**v0.1 开发基线**。本文档定稿后，任何实现偏离必须先在 `DECISIONS.md` 追加一条 ADR 说明理由，不得静默改设计。
>
> 前身：`../anyinfer`（原型，PyTorch + vLLM wrapper）。本文档是对原型六轮架构评审的收敛结果。

---

## 0. 本文档的地位

三份文档分工，避免事实漂移：

| 文档 | 内容 | 变更方式 |
| --- | --- | --- |
| `DESIGN.md`（本文） | 当前生效的架构契约 | 改设计前先改本文 |
| `DECISIONS.md` | 决策记录（ADR），含被否决方案与理由 | 只追加，不修改历史条目 |
| `TODO.md` | 可勾选的任务队列（带完成判据与依赖） | 完成即勾选；与本文冲突时以本文为准 |
| `README.md` | 怎么构建、怎么跑里程碑 | 随代码更新 |

---

## 1. 定位与边界

### 1.1 做什么

面向**本地 RSI / RL rollout** 的 Ascend 原生推理引擎：纯 Rust 控制面 + Ascend C 自研算子的裸金属运行时，为递归自我改进闭环提供可被自动修改、可被自动裁决的算力底座。

### 1.2 不做什么（明确排除，避免范围蔓延）

- ❌ 通用云端 Web Serving（不追求多租户、不退化为 OpenAI 兼容网关）
- ❌ 多机分布式推理（本基线 v0.1–v0.4 只做单机；通信层单列，见 §9.3）
- ❌ 复用 vLLM 的执行路径（vLLM 仅作**差分测试的对照组**，不作为运行时依赖）
- ❌ 异构卡适配（只面向 `ascend910b` / `ascend910_93` 两个分支起步）

### 1.3 目标包络（**必须先钉死的变量**）

| 项 | 基线取值 | 影响 |
| --- | --- | --- |
| 卡数 | **1× 910B（64 GB）起步** | 无需 HCCL/TP/EP；所有设计可在此规模验证完 |
| 模型 | 7B–32B dense 或小 MoE；w8a8 / w4a8 | 权重 ≤ 35 GB，"秒级启动"成立 |
| 主 KPI | **p99 TTFT、采样吞吐（steps/s per host-core）、前缀命中率** | 决定奖励函数，见 §6 |
| 次 KPI | 抢占率、冷启动墙钟、reload 墙钟 | 不进主排序，但设下限门 |

> **未定项**：千亿级 MoE 与多卡是 v0.5 后的独立轨道。任何"为 V4 优化"的提案必须等这条轨道开启，不得提前污染单卡设计。

---

## 2. 从原型继承什么、切断什么

### 2.1 继承（这些是原型最有价值的资产，不要重写）

| 资产 | 位置（`../anyinfer`） | 用途 |
| --- | --- | --- |
| 位级校验方法论 | `native/acl_backend/reference.h` + `bench.cpp:196` | 算子验收第一道门 |
| 纯 POD C ABI 范式 | `native/acl_backend/hc_post.h` | FFI 边界的直接模板 |
| 度量纪律 | `src/native/scheduler.rs`（paired A/B）、`deploy/calibrate.py`（per-device roofline） | 奖励信号与同卡归一化 |
| 溯源纪律 | `deploy/operator_integrity.py`（哈希链 + 加载期自证） | 自举安全的基础设施 |
| 必需算子清单 | `operators/inventory.tsv`、`docs/operators.md` | 算子轨道的排期依据 |
| 参考实现 | `operators/vllm-ascend/`（2747 文件 / 41 MB，只读对照） | 差分测试对照组、算法参考 |

### 2.2 切断

- 不 vendoring `vllm-ascend`（41 MB / 2747 文件 / 58 万行源码）；只**只读引用**，需要时按文件摘录。
- 不保留 PyTorch / Python 运行时（v0.1 阶段仅用其读权重与当对照，v0.3 删除）。
- 不保留原型的 `reference/` vLLM 对比栈（那属于 RPM 度量，不属于本引擎）。

---

## 3. 核心决策摘要

细则见 `DECISIONS.md`。这里只列**会反向约束实现**的六条：

| # | 决策 | 一句话理由 |
| --- | --- | --- |
| D1 | 每步跨 FFI **一次**（`anyinfer_engine_step`），per-op 调用全在 C++ 内 | 消除边界开销的叙事必须自洽 |
| D2 | KV 走**设备侧 block table 间接寻址** | 图重放要求地址静态，而 CoW 让地址变化 |
| D3 | 图捕获边界**止于 top-k + logsumexp**；FSM 掩码走**固定设备位掩码缓冲** | 既保住异步流水，又不牺牲 grammar |
| D4 | `weight_version` 同时是 **prefix cache 的 epoch** | 换权重而不失效前缀 = 静默污染 |
| D5 | batch invariance **分级**：decode dense 路径严格，MoE 与 prefix 共享允许容差 | 前缀共享与绝对不变性在数学上冲突 |
| D6 | 自研算子**优先 Ascend C 裸指针**，只在必要时用 ACLNN + 指针重绑 | ACLNN 的 `aclTensor` 描述符会在捕获期固化指针 |

---

## 4. 执行模型

### 4.1 每步一次跨语言

```
Rust（控制面）                           C++（计算面）
────────────────────────────────────────────────────────
写 host 侧元数据 → 小 H2D            ┌── embed → layers × N → lm_head
                                     │   → +bitmask → logsumexp → top-k
anyinfer_engine_step(engine, step*) ─┘
读回 top-k + logsumexp（16 KB）
采样 / 记录 logprob / 推进 FSM
更新 block table 与 bitmask（下次 H2D）
```

- **禁止**在热路径上按 op 跨 FFI。prefill 的 per-op 序列在 C++ 内展开。
- **禁止**在热路径上分配。所有缓冲在 plan 阶段一次分配（原型 `tests/host_decode_allocations.rs` 的零分配断言是正确做法）。

### 4.2 捕获边界

```
[NPUGraph 捕获区]  embed → layers × N → final norm → lm_head
                   → logits += (mask ? 0 : -inf)   ← 读固定地址 bitmask
                   → logsumexp(fp32)
                   → top-k (k ≥ 256) values+indices
[动态区 / host]     归一化、采样、logprob 记录、FSM 推进、bitmask 生成
```

为什么不让图止于 logits：全量 logits 是 `vocab × batch × 4B`，128 k 词表 × batch 32 = **16.4 MB/步**，
PCIe 上 0.8–1.5 ms 并强制每步流同步，对 30 ms 步长是 3–5% 且打断预取下一步的能力。
bitmask（128 k bits = 16 KB）+ 设备侧 top-k/logsumexp 把它压到 **16 KB**。

**两条必须写死的护栏**：

1. `k` 必须够大（≥ 256）。若掩码把 top-k 内候选全灭，走**全量 logits 慢路径回退**，并统计其触发率。
2. `logsumexp` 必须 fp32 且归约顺序与训练端一致，否则 logprob 对不上（见 §6.2）。

### 4.3 动静结合

| 路径 | 下发方式 | 理由 |
| --- | --- | --- |
| Prefill（变长、chunked） | 逐 op 序列 + 缓存 `aclOpExecutor` | shape 每步变，图捕获无收益 |
| Decode 稳态 | NPUGraph 重放 | 固定 shape，下发开销是纯损失 |

### 4.4 图按 batch 桶 + padding 隔离

- decode 图按 `{1,2,4,8,…,max_num_seqs}` 分桶捕获；实际 batch 向上取桶、余行 padding。
- **padding 行必须两端隔离**（原型 vendored 代码 `attention_v1.py:370-396` 给出了生产做法）：
  - 读侧：`seq_lens` 传真实值，dummy 行取 KV 长度 1、block 行全 0；
  - 写侧：`slot_mapping` 只取 `[:num_actual_tokens]`，dummy 行**绝不落 KV**。
  - 违反任一条 → dummy 行写入 HBM → **轨迹永久污染**。
- 必须报告 padding 浪费比（本地小 batch 场景可能数个百分点，要报数不要猜）。

**硬约束：捕获区内只允许读固定地址的设备缓冲，不得依赖任何 step 级 host 标量。** 标量会在捕获时被固化（`graph_probe_test.cpp` 的 Arm 1 就是用来证明这一点的）。

---

## 5. 内存模型

### 5.1 双层：Arena + Block

```
aclrtMalloc 一次性申请（大页）┌──────────────────────────────────┐
                            │ block0 │ block1 │ … │ blockN     │  ← 物理层，O(1) 队列分配
                            └──────────────────────────────────┘
Rust 侧 Radix Tree  ── 逻辑层：前缀共享、CoW 分叉、引用计数
```

- **block size 是部署期参数，不是常量**：原型自己的 vLLM 路径就是 SoC 相关（A2/V4.1 用 128，其余 32）。硬编 16/32 会在某些 SoC 上让 attention 掉档。
- **引用计数必须覆盖祖先链**：后代持有全部祖先 block 的引用，否则共享系统提示词的中间节点会被淘汰 → 活跃请求被毁。

### 5.2 驱逐 / 抢占 / 换出

三级，缺一不可：

1. **驱逐**：只允许淘汰 `refcount == 0` 的**叶**节点，按逻辑 LRU。
2. **换出**：叶节点都被引用且 HBM 紧张 → KV 换到 pinned host 内存。
3. **抢占**：换出空间也不够 → 丢弃被抢占序列的 KV，重算 prefill。

**抢占率**定义（可测）：`Recompute Tokens / Total Decoded Tokens`，目标 < 5%。

### 5.3 weight_version 是 prefix cache 的 epoch

> 换权重后，旧 KV 是用旧权重算的。若新轨迹命中旧前缀，就会用**旧权重产生的 KV + 新权重算的后续层**——静默污染，且不报错。

规则：`weight_version` 变化 ⇒ Radix Tree 整体失效。代价必须计入 reload 预算：

```
T_reload = W / BW + T_reprompt
```

`T_reprompt` 是系统提示词重新 prefill 的时间（8k token × batch 32 是秒级，与 I/O 同量级）。
**结论：reload 的暂停不应被"隐藏"，而应被调度到自然屏障并计入预算。**

---

## 6. RL 契约（验收门）

### 6.1 分层 Oracle

| 层 | 判据 | 覆盖对象 |
| --- | --- | --- |
| L1 位级 | 构造精确输入，逐位相等 | 元素级算子：norm / rope / swiglu / dequant / hc_post |
| L2 FP64 参考 | 相对误差 + max_abs_error | GEMM / attention |
| L3 路由一致性 | expert 选择集合逐 token 相同 | MoE router / grouped GEMM |
| L4 端到端 | token 序列一致 + logprob 容差 | 全链路 |

**L2/L4 的 oracle**：`tools/toy/` 的 FP64 参考实现（`model.py`）+ `compare.py` 对拍器。
玩具模型与真实 decoder 同构（rmsnorm / qkv / RoPE / causal attention / swiglu / tied lm_head），
权重是 dyadic（`k*2^-7`，fp32 精确），所以"对不上"能直接定位到某一层、某一 token。

**RoPE 只能走 L2**：旋转角是超越数，`cos/sin` 的结果不可能是 dyadic，
因此含 RoPE 的层永远不满足 L1 的位级判据。这条在 `tools/toy/model.py` 里有测试断言。

**L1 前提**：输入必须是"乘积与求和可精确表示"的有界二进制有理数（原型 `reference.h:23-35` 的做法）。**没有这个前提，位级相等不成立，不许声称。**

**L1 的仲裁者**：`native/harness/exact.h` —— `product_is_exact` / `sum_is_exact` / `dot_is_exact`。
任何声称 L1 的测试必须先用它判定"这组输入下位级相等是否本来就成立"；判据不成立时应降级为 L2。
另有一条硬约束：**算子内部若有 split-K、atomic 累加、或按运行时数据决定归约阶数，则无论输入多精确都不存在"顺序无关"，不许声称 L1。**

### 6.2 logprob 保真（P0）

- 引擎必须输出 `(token, logprob[, top-k logprobs])`。
- 验收门（取代模糊的 KL 容差）：

```
max|Δlogprob| < 1e-3   ∧   clip_rate(PPO) < 1%
```

- `Δlogprob` 相对**训练端**计算，不是相对自己上一次。

### 6.3 batch invariance（分级）

- **严格**：decode 的 dense 路径（attention / GEMM / 归约）在 batch=1 与 batch=N 下逐位一致。
  实现手段：固定 tile、固定累加顺序、**不用 split-K、不用 atomic**。
- **允许容差**：MoE 路由置换、prefix 共享复用。
- 理由见 D5：被共享的前缀 KV 是在某次特定 batch shape 下算出来的；共享 = 注入"别人的归约顺序"。要么规范化 prefill shape，要么接受容差——本基线选后者。

### 6.4 确定性

同 seed + 同 prompt + 同 weight_version ⇒ **同轨迹**（token 序列逐位相同）。这是 RL 调试的底线，也是 L1 位级自检的前提。

### 6.5 Drain & Swap

1. 到达屏障 → 停止接纳新请求。
2. **跨 rank collective barrier**（单卡时为退化情形）。
3. 等待在飞轨迹结束，或按 PPO masking 截断并打标。
4. `aclrtSynchronizeStream` 确认无在飞 step。
5. 切换权重指针 → **递增 weight_version → 失效 prefix cache**。
6. 恢复接纳。

**绝不允许在一个 step 内混合新旧权重。**

---

## 7. MoE 特定约束

1. **容量只 pad、绝不 drop。** drop 会改变模型输出——那不是数值漂移，是策略变了，PPO 会把它当模型选了别的动作。
2. **容量上界 = `max_num_seqs × top_k`，只用于 decode。** prefill 的 token 数远超此值，fixed capacity 会导致显存爆炸或丢弃 → prefill 走动态 shape。
3. **反过来说，fixed capacity 是 MoE 能被图捕获的前提**（专家 GEMM 的 shape 必须静态）。这条因果关系要写进注释，否则后人会当成性能妥协而删掉。

---

## 8. 自举安全（RSI 成立的前提）

> 让 agent 改自己的底座，必须保证**改坏了还能启动**，且**运行期能回退**。

1. **双版本 kernel**：`kernels/current.so` 与 `kernels/last-known-good.so`。
2. **启动自检用位级复算，不是 1e-3 容差。** 同一代码、同一输入、同一 seed 必须**完全一致**；给容差等于把确定性 bug 藏起来。`1e-3` 是 L4 跨实现的尺子，两把尺子不许混用。
3. **运行期自检**：每 30–60 min 或每 N 次 reload，在空闲窗口跑一次位级自检；失败 → 熔断 → 切 stable → 上报。因为"跑三小时后偶发错"才是真贵的 bug。
4. **回退必须能在运行期触发**（step 边界重绑 kernel），不能只靠重启进程。
5. **不可自修改面**：测量代码、golden 语料、基准套件、构建脚本、回退逻辑对 agent 只读且哈希锁定。

---

## 9. 演进路线

### 9.1 引擎轨道

| 阶段 | 交付 | 硬门 |
| --- | --- | --- |
| **M-A（当前）** | 图捕获 + 间接寻址**判决**（见 §10） | `verdict == indirection_supported`，否则转 A'/B/C |
| **M-B** | 算子 harness（精确输入 + 位级/FP64 比对 + JSON 裁决 + 熔断） | 能自动判定"某算子改动是否可合入" |
| **M-C** | 玩具模型 + 逐层对拍导出（token / logprob / 每层激活） | L4 门可用，能定位到具体层 |
| **v0.1** | 单卡端到端生成（自研 scheduler + 自研 kernel 子集）+ D1 边界 | L1–L4 全绿 |
| **v0.2** | Arena + Block + Radix + CoW + 驱逐/换出/抢占 | CoW 正确 + 打满不崩 |
| **v0.3** | mmap + pinned staging + 并发 DMA；去 PyTorch；设备侧 top-k/logsumexp | 冷启动 ≤ 2 s（≤35 GB 权重） |
| **v0.4** | Drain & Swap + weight_version epoch + 双版本自检回退 | 故意注入坏 kernel 能自动回退 |

### 9.2 算子轨道（并行，与引擎轨道同等优先级）

按 trace 热点排序，一次只换一个算子。每个算子必须同时过：

```
L1/L2/L3 正确性门  +  微基准不劣于当前  +  端到端 L4 无回归
```

三条不齐，一律不合入。

### 9.3 通信层（独立轨道，v0.5+）

单机多卡（TP/EP）、HCCL 初始化、EP all-to-all 与 fixed capacity 的交互。**当前不实现，但不得在设计上堵死**（例如 block table 布局要能按 rank 分片）。

---

## 10. 里程碑 A：图捕获判决（本轮开发目标）

**要回答的问题**：CANN 图重放时，读「固定地址缓冲的内容」是否能反映每步 H2D 的新值？

**方法**：四臂测试（缺一不可）

| 臂 | 内容 | 失败意味着 |
| --- | --- | --- |
| Arm 0 | 不捕获，直接下发 | kernel 本身错，与捕获无关 |
| Arm 1 | 捕获 + **直接地址**（标量固化） | 捕获通路坏（流顺序/API） |
| Arm 2 | 捕获 + **间接寻址**，两次重放换表内容 | 主判决：折叠 → 转后备 |
| Arm 3 | 再次换表重放 | 可重入性 |

**关键防伪**：每次重放前用 `aclrtMemset(..., 0xEF, ...)` 写哨兵。
没有哨兵，"读了旧块"与"图根本没跑"不可区分 —— 两者都是旧值。

**裁决表**

| verdict | 含义 | 后续 |
| --- | --- | --- |
| `kernel_broken` | Arm 0 失败 | 查 kernel |
| `empty_graph` | 捕获到 0 节点 | 该下发路径不可捕获 → A' |
| `capture_path_broken` | 控制臂失败 | 查流顺序 / 捕获区内 sync |
| `content_folded` | 二次重放仍旧值 | **真折叠** → A'/B/C |
| `replay_not_executing` | 二次重放哨兵未动 | 重放机制问题 |
| `indirection_supported` | 主路径成立 | Phase 2 按现设计推进 |

**后备梯队**

| 层 | 机制 | 触发 | 代价 |
| --- | --- | --- | --- |
| A | 间接寻址（Ascend C 裸指针） | 主路径 | — |
| A' | 指针重绑（`graph_task_update` 等价物） | 复用 ACLNN 算子时 | 每步 update + event 排序；paged attention 的 workspace 需重查 |
| B | 静态扁平 KV（每序列预留 `max_seq_len` 连续区 + 独立共享前缀段） | A/A' 均失败 | 显存浪费、无细粒度共享 |
| C | 缓存 `aclOpExecutor` + 逐 op 下发 | 图捕获彻底不通 | 每 op 一次下发；30 ms 步长下可接受 |

**依据**：`../anyinfer/operators/vllm-ascend/vllm_ascend/compilation/updatable_graph.py` 是生产代码，
证明 CANN 提供节点参数重绑（`graph_task_update_begin/end`），即"不折叠、但 ACLNN 需要重绑"。

---

## 11. 工程预算（架构需求，不是细节）

| 项 | 预算 | 为什么是架构需求 |
| --- | --- | --- |
| 全量重建 | **≤ 10 min** | RSI 迭代频率 = 1/(构建+测量) |
| smoke 一轮 | **≤ 60 s** | agent 一天要试几十次 |
| 权重 reload | `W / BW`（不隐藏）+ reprompt | 见 §5.3 |
| 冷启动 | ≤ 2 s @ ≤35 GB | 决定"秒级启动"是否可宣称 |
| 每步 host/device 比 | 先测，再定投入 | 决定 host 优化的预算 |

---

## 12. 未决问题（每条都可用 <15 分钟验证）

| # | 问题 | 验证方式 |
| --- | --- | --- |
| Q1 | 本机 CANN 暴露哪套图 API（`aclrtStream*` / `aclmdlRI*`） | `grep` 头文件 |
| Q2 | `DataCopy` 的第二参数是元素数还是 32B 块数 | 查头文件注释 |
| Q3 | Ascend C kernel 能否被图捕获，还是只支持 ACLNN | 里程碑 A 的 Arm 1 |
| Q4 | 间接寻址在重放中是否生效 | 里程碑 A 的 Arm 2 |
| Q5 | top-k 回退（掩码清空 top-k）的真实触发率 | 玩具模型上统计 |
| Q6 | 图桶 padding 的浪费比 | 同上 |

---

## 附录：术语

- **Arena**：启动时一次性申请的 HBM 大块。
- **Block**：Arena 内的定长 KV 单元（size 为部署期参数）。
- **Radix Tree**：前缀索引结构；block 是物理单元，树是逻辑视图。
- **CoW**：写时复制；分叉轨迹共享未分裂节点。
- **Epoch**：`weight_version` 的别名，用于批量失效缓存。
- **桶（bucket）**：图捕获时的 batch 尺寸档位。
