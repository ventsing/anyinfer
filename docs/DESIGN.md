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
| 芯片 | **1× 910C**（`ascend910_93xx`，A3 家族） | 无需 HCCL/TP/EP；所有设计可在此规模验证完 |
| HBM | **待实测** — 走 `tools/device_probe.sh` | KV 池容量、并发上限、抢占率门槛全由它定，见 §5.2.4 |
| block_size | **待实测**：由 SoC 定，A2/V4.1 用 128，其余 32，见 §1.3.1 | 硬编 16/32 会在某些 SoC 上让 attention 掉档 |
| 模型 | 7B–32B dense 或小 MoE；w8a8 / w4a8 | 权重 ≤ 35 GB，"秒级启动"成立 |
| 主 KPI | **p99 TTFT、采样吞吐（steps/s per host-core）、前缀命中率** | 决定奖励函数，见 §6 |
| 次 KPI | 抢占率、冷启动墙钟、reload 墙钟 | 不进主排序，但设下限门 |

> **上面两条"待实测"是缺口，不是设计选择。** 本表此前写的是「1× 910B（64 GB）」——
> 那是基于一台当时并不存在的机器写下的假设，实机是 910C，HBM 与 block_size 都得从机器上读。
> **读出来之前，任何容量类数字（KV 池大小、图桶档位、抢占率门槛 5%）都不得引用**：
> 它们在 910B 上是猜的，在 910C 上连猜的依据都没有。
>
> 采集入口是 `tools/device_probe.sh`，产出 `device-probe.json`；A-01 的两项结论
> （`ANYINFER_GRAPH_API` 与 `ANYINFER_SOC`）也从那里出。

#### 1.3.1 SoC → 设备家族（决定 block_size 与编译期 SoC_NAME）

| SoC 字符串 | 家族 | block_size | 编译期 `-DANYINFER_SOC=` |
| --- | --- | --- | --- |
| `910b` / `ascend910b1`..`b4` | A2 | 128（V4.1 同） | `ascend910bN` |
| `910c` / `ascend910_93xx` | A3 | **待实测** | `ascend910_93xx` |
| `310p*` | 310P | 32 | `ascend310pN` |
| `ascend950*` | A5 | **待实测** | `ascend950xx` |

映射依据是 vendored 参考实现自己的判据
（`operators/vllm-ascend/vllm_ascend/device/hardware.py` 的 `_SOC_VERSION_TO_DEVICE_TYPE`，
在 `main` 分支上）—— **不是我推的**。它把 `910c` 与 `ascend910_9391/9381/9372/9392/9382/9362/9363`
一并归入 A3。A3 的 block_size 留空，因为参考实现里没有这条映射，只能在机器上量。

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

### 4.5 C ABI 边界（`native/abi/anyinfer.h`，D-01）

边界上只有 `AnyInferStep`（一 step 的全部输入输出）与 `AnyInferStatus`（失败信息，
含 `failed_op_id`）。后者是 RSI 闭环的关键：不能确定"哪一层的哪个算子算错了"，
误差就无法归因到具体的优化动作，闭环退化成"整体重跑一遍试试"。

三条自缚规则，都为了消除"编译期都过、运行时读到别的字段"这类最难查的错误：

| 规则 | 做法 |
| --- | --- |
| 不用 enum 表示状态码与哨兵 | `int32_t` + 具名常量 / 宏。enum 的底层类型是实现定义的，这层猜测没必要保留 |
| 结构体里零隐式 padding | 字段顺序排到首尾相接，`offsetof` 连续性被两侧测试断言；padding 字节未定义，会破坏"两次构造逐位相等" |
| sizeof 硬钉住 | 加字段必须同时改 C 常量、`abi_layout.txt`、Rust 定义。这是**刻意制造的摩擦** |

**验证链是一份产物、两边独立校验**：

```
Rust 定义 ──生成──> native/abi/abi_layout.txt <──解析校验── C 定义
```

`abi_layout.txt` 是纯文本，所以"布局变了"在 code review 里直接可见。
两侧各自对着它断言 size/align/offset 与"无 padding"，任何一侧漂移都被抓到。
ABI 只支持 LP64（指针 8 字节），32 位目标直接编译失败而不是悄悄错位。

### 4.6 步级 plan：分配只在 plan 期（`crates/anyinfer-plan`，D-02）

V4.1 上一个 decode step 约 30 ms，被 100+ 层计算填满；host 侧一次 `malloc` 只要几百纳秒。
**所以问题不是分配慢，而是它出现在热路径上的位置不可控**：分配失败时故障点落在"第 N 步的第 3 行
已经写了一半"的地方；分配器抖动把 p99 推向毛刺；新分配的缓冲可能没被 pin，H2D 退化成同步拷贝。

因此不去优化分配速度，而是**把分配从热路径上移除**：形状在 plan 期定死，缓冲一次要够，
稳态只做 `copy_from_slice`。plan 期恰好 11 次设备区域申请 + 9 次 host 缓冲分配，此后归零。

三条不可协商的规则：

| 规则 | 违反后果 |
| --- | --- |
| 所有缓冲长度恒等于桶形状 | 捕获图按固定形状录制，传短了就是越界读 |
| 每一步整块重写，不留残留 | 上一步的块号被这一步读到 = 读别人已释放的 HBM，不崩、但结果依赖历史 |
| 先全部校验，再全部写入 | "写一半才发现第 2 行非法"留下内部不自洽的 step，调用方可能已用了一半 |

两个减少误用面的设计：

- **`slot_mapping` 由块表推导**，不接受调用方另给一份。两份数据一旦不一致，KV 就写进错位置，极难查。
- **`DecodeRow` 没有 `position` 字段**：decode 一步只追加一个 token，`position ≡ seq_len - 1`。
  留一个可以与之打架的入口，就等于留一个必然有人填错的字段。

形状超界时**返回错误**，由上层换桶换 plan —— 绝不在热路径上扩容。扩容一次 = 重新分配 + 重新捕获图
= 这一帧的 30 ms 变成秒级。

零分配这件事必须被**断言**而不是被信任（`tests/zero_allocation.rs`）：计数分配器只统计本线程，
稳态 4096 步断言分配/扩容/释放全为 0，并带一个**正控制**用例证明钩子本身是活的 ——
否则"零分配"对"钩子根本没接上"同样成立，那是一个永远绿的、什么都不检查的门。

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

#### 5.1.1 前缀树的粒度：块级（`crates/anyinfer-kv`，E-02）

| 粒度 | 好处 | 代价 |
| --- | --- | --- |
| token 级（SGLang 风格） | 命中率最高，任意长度前缀都能共享 | 一个物理块被两个节点共享 → CoW 成为必需，每次分裂都要动块 |
| **块级（本基线）** | 一个块**只属于一个节点**，树结构即所有权结构 | 短于块边界的共享前缀不可复用（每序列最多浪费 `block_size - 1` 个 token 的缓存机会） |

选块级，是因为"块的所有权 == 树节点的所有权"这条不变量一次消掉一整类 bug：
没有块被两个节点共用，就不存在兄弟节点把块写脏、不存在分裂时的块复制、不存在两把锁的顺序问题。
代价是可量的（缓存命中率），不是猜的。

**副产物：这条决定让 E-03 的 CoW 变成多余。** 原计划 E-03 要做写时复制，理由是
"分叉后写不污染兄弟"。块粒度下这个需求换一种方式就满足了，且不需要任何复制动作
（详见 §5.1.3）。所以没有实现 CoW，而是把"永远不会写共享块"做成结构性检查 ——
实现一个用不上的 CoW 路径反而会让后来人以为共享写是允许的。

#### 5.1.2 两种引用计数，只有一种能用来判断可回收

| 计数 | 含义 | 能否判断可回收 |
| --- | --- | --- |
| 直接附加数 | 有多少序列把 KV 路径停在这个节点 | **不能** |
| 祖先链引用计数 | 有多少活跃路径**经过**这个节点的块 | 能 |

序列命中 `root → A → B` 时，注意力会读 A 的块和 B 的块。若只数直接附加，A 的计数恒为 0
（没人把路径停在中间），于是任何"计数为 0 即可回收"的判据都会把 A 回收掉，
而挂在 B 上的活跃请求仍在读它 —— 读到别人写过或已释放的 HBM。设备上不崩、不报错，只让结果依赖历史。

因此 `attach` 对**整条 root→node 路径**加引用。两种计数都保留并都暴露，就是为了让这个差别可断言：
`naive_shallow_counting_would_free_a_block_that_is_still_being_read` 把朴素判据的判定结果
和真实后果算在一起摆出来，使"祖先链计数不可省"成为一个可测结论而不是一句注释。

回收判据同时用了两条（叶节点 ∧ 引用计数为零）。在纯树结构下二者冗余，**保留两条是有意的**：
等 E-05 的块表快照、或将来引入 token 粒度缓存打破"块只属于一个节点"之后，
叶节点条件不再蕴含引用条件，届时只有引用计数还在兜底。

#### 5.1.3 序列的 KV 布局：共享前缀 + 独占尾巴（`crates/anyinfer-kv::sequence`，E-03）

```
序列的块表 = [ 共享前缀块（只读，来自前缀树） | 独占尾巴块（可写，本序列私有） ]
```

| E-03 原判据 | 块粒度下如何满足 |
| --- | --- |
| 共享前缀只算一次 | 前缀树命中即复用同一批块，不复制 |
| 分叉后写不污染兄弟 | 新 token 写进**本序列私有的块**，兄弟序列根本拿不到同一块 |
| 共享前缀不被写 | 共享段的 token 数恒为 `block_size` 的整数倍 ⇒ 写入点必然落在共享段之后 |

第三条是整个结论的支点：**块粒度让"共享段"与"写入点"在结构上不相交**。
若是 token 粒度共享，两个序列可能共享同一个未填满的块，各自的下一个 token 要写进同一块的
同一个 slot —— 那才必须 CoW。

因此 `append_token` 在 `len < shared_tokens` 时直接返回 `WouldWriteSharedBlock`。
这条守卫在正常路径上**永远**碰不到（所以它同时也是对前缀树"节点恒为整块"的一次持续校验），
并且有一条负向对照证实：摘掉守卫，对应用例立刻红。

经验证据而不是论证：`randomized_workload_never_writes_a_shared_block` 跑 3000 步随机
insert/fork/append/join/回收，**每一次写入都断言目标块引用计数为 1 且不在共享段内**。

引用计数的不变量由此变得很干净：

```
对任意在用块：arena.refcount(block) == 正在使用这个块的活跃序列数
```

于是 `Arena::release` 的"引用计数非零即拒绝"不只兜住读侧，也兜住写侧。

### 5.2 驱逐 / 抢占 / 换出

三级，缺一不可：

1. **驱逐**：只允许淘汰 `refcount == 0` 的**叶**节点，按逻辑 LRU。
2. **换出**：叶节点都被引用且 HBM 紧张 → KV 换到 pinned host 内存。
3. **抢占**：换出空间也不够 → 丢弃被抢占序列的 KV，重算 prefill。

**抢占率**定义（可测）：`Recompute Tokens / Total Decoded Tokens`，目标 < 5%。

#### 5.2.1 三级的代价差两个数量级，所以顺序是硬顺序（`crates/anyinfer-kv::reclaim`，E-04）

| 级别 | 代价 | 可逆 | 何时允许 |
| --- | --- | --- | --- |
| 驱逐 | 0（丢掉没人引用的缓存条目） | 只是"以后可能要" | 总是先做 |
| 换出 | 一次 D2H + 一次 H2D 的带宽 | 可逆，KV 还在 | 驱逐不够时 |
| 抢占 | **整段 prefill 重算** | 不可逆 | 换出空间也不够时 |

4k token 的 prefill 是毫秒级计算，换出 100 MB 在 2 TB/s 上是几十微秒 —— 差两个数量级。
所以"只要还有一个可驱逐的缓存条目，就绝不允许抢占"是硬约束，不是启发式。
`ReclaimPlan::respects_escalation` 把它变成可断言的性质。

#### 5.2.2 一个不显然的结论：换出要保护"贵的"，抢占要牺牲"便宜的"

换出本身不花计算，所以应该用它去**保住重算代价高的序列**；
真要抢占时，应该挑重算代价最低的。于是两级的挑选顺序**相反**：

- 换出：按 `tokens` **降序**（先保护最贵的）
- 抢占：按 `tokens` **升序**（先牺牲最便宜的）

如果两级用同一个顺序（比如都按 LRU），换出就会先把便宜的保护起来，
留下贵的等着被抢占 —— 恰好和"最小化重算总量"的目标相反。
平局按序列 id 裁决，保证同输入同计划（否则回收策略会悄悄破坏"同 seed 同轨迹"）。

#### 5.2.3 抢占率必须按窗口看，累计率会撒谎

一次 10% 的抖动放进 10 倍长的历史里就是 1%，"看起来达标"。
所以同时给累计率与**最差窗口率**，后者才是会让人半夜被叫起来的那个数。
分母为 0 时返回 `None` 而不是 `0.0` —— 分母为零的"完美达标"和真正的零抢占长得一模一样，
而前者只说明这一窗什么都没跑。

**比率的可比性有前提**：只有在**服务了同样多的工作**的两个 run 之间才可比。
容量小到负载跑不完时，比率被"饿死"主导而不是策略：
定档表里 14 块时开驱逐的比率（0.176）反而比关驱逐（0.081）好看，原因是两个 run 服务的请求数不同。

#### 5.2.4 容量定档（仿真给出的数，不是拍的）

工作负载：4 路并发 × (8 token 共享前缀 + 生成 12 token)，缓存 3 条系统提示词，40 个作业。

| 块数 | 服务 token | 抢占次数 | 重算 token | 抢占率 |
| --- | --- | --- | --- | --- |
| 12 | 45 | 6 | 9 | 0.200 |
| 16 | 540 | 32 | 72 | 0.133 |
| 18 | 488 | 1 | 8 | 0.016 |
| **19** | 480 | 0 | 0 | **0.000** |

所以"零抢占"的门槛被钉在 19 块 —— 这条数字进了测试，
准入/回收/块粒度任何一处变了，`capacity_that_reaches_zero_preemption_is_pinned` 会先红。

#### 5.2.5 仿真抓到的设计缺陷：纯 LRU 驱逐可以是净负的

不在原计划里，是 `simulation.rs` 跑出来的：

| 12 块同容量 | 服务 token | 重算 token |
| --- | --- | --- |
| 开驱逐 | 45 | 9 |
| 关驱逐 | **200** | 32 |

原因是缓存里那三块是**共享系统提示词**，而 LRU 只看"最久没被命中"，
于是它优先踢掉的恰好是价值最高的节点：腾出 2 块，却让后续十几个作业
失去 8 token 的免费前缀，每个都要多占 2 块。

**结论：驱逐不能按"最近使用"挑牺牲者，要按复用价值挑。** 列为 E-07。
当前 `evict_leaf_lru` 的行为被表征测试钉住，改策略时那个测试会先红。

**"Recompute Tokens" 只算本引擎生成过的 token**：序列被抢占时其 `len()` 里含着
从缓存白拿的共享前缀，那些重试时照样命中、一分计算都不用付。
把它们算进分子会让指标凭空翻几倍 —— 这是仿真第一版踩过的坑。

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

### 6.1.1 算子晋升裁决（`native/harness/accept.h`，B-04b）

一次算子改动只有三条门同时成立才允许晋升：

| 门 | 判据 | 来源 |
| --- | --- | --- |
| `correctness_gate` | L1 位级 / L2 容差 | `bit_exact.h` / `fp64_reference.h` |
| `microbench` | **不劣化**（不是"更快"） | `bench.h` 的配对符号检验 |
| `l4_gate` | 端到端 token + logprob | `tools/toy/compare.py` |

三条硬规则：

1. **三态而非布尔**。`GateState{Pass, Fail, NotRun}`。若只用 true/false，
   "这道门今天没跑"和"这道门通过了"就是同一个值 —— 于是第一个版本的验收工具
   会永远输出 `l4_gate: pass`，而没有人会注意到它从未运行过。
   `NotRun` 阻断晋升，退出码 2 与 `fail` 的 1 分开：处置方式不同
   （`fail` → 改候选；`incomplete` → 补证据，禁止基于残缺证据决策）。

2. **微基准的门槛是"不劣化"**。`indistinguishable` 判通过（证据不足不等于更差），
   `a_faster` 判失败。另加守护线：回归幅度超过 10% 时**即使未达显著**也判失败 ——
   测不出显著性的巨大回归说明测量本身不可信，不可信就不许晋升。

3. **归因必须可区分**。`invalid_reference` 说明坏的是参考实现、不是被测算子；
   `length_mismatch` 说明坏的是调用方。两者都阻断，但责任人不同，
   报告里必须分开写，否则有人会去优化一个本来就没问题的 kernel。

裁决权威只有 `accept.h` 一处（C++ CLI `anyinfer-accept`）。
跨语言入口 `tools/accept.py` 只做提取、不做聚合 —— 两份实现迟早会分叉。
每份判决都记录当时用的策略参数（`max_regression`），因为**没有规则记录的判决不可审计**。

### 6.2 logprob 保真（P0）

- 引擎必须输出 `(token, logprob[, top-k logprobs])`。
- 验收门（取代模糊的 KL 容差）：

```
max|Δlogprob| < 1e-3   ∧   clip_rate(PPO) < 1%
```

- `Δlogprob` 相对**训练端**计算，不是相对自己上一次。

**采样分布与导出的 logprob 是两个量，不许互相代替**（`crates/anyinfer-sampler`，D-04）：

| 量 | 定义 | 用途 |
| --- | --- | --- |
| 采样分布 | `softmax(logits / T)` 截断到 top-k 再截断到 top-p，在**候选集**上归一 | 决定出哪个 token |
| 导出的 logprob | `logit[chosen] - logsumexp(全词表)` | 训练端 / PPO 的 KL 与重要性比 |

用采样分布算 logprob，会把温度和 top-p 的裁剪混进训练信号，PPO 的重要性比立刻算错——
而且是"看起来只差一点点"的那种错，最难发现。所以导出的 logprob 恒等式
`values[chosen] - full_logsumexp` **与 temperature / top_p 无关**，这条有专门的测试盯着。

配套的诚实性约束：top-k 截断过多时，采样分布与真实分布已经不是一回事，
而这**不会报错**。`candidate_mass()` 报出候选集覆盖的概率质量；要么报数，要么别裁。

### 6.3 batch invariance（分级）

- **严格**：decode 的 dense 路径（attention / GEMM / 归约）在 batch=1 与 batch=N 下逐位一致。
  实现手段：固定 tile、固定累加顺序、**不用 split-K、不用 atomic**。
- **允许容差**：MoE 路由置换、prefix 共享复用。
- 理由见 D5：被共享的前缀 KV 是在某次特定 batch shape 下算出来的；共享 = 注入"别人的归约顺序"。要么规范化 prefill shape，要么接受容差——本基线选后者。

### 6.4 确定性

同 seed + 同 prompt + 同 weight_version ⇒ **同轨迹**（token 序列逐位相同）。这是 RL 调试的底线，也是 L1 位级自检的前提。

两条实现上的硬约束（`crates/anyinfer-sampler`）：

1. **决策路径上不出现 libm**。`exp` 用自己实现的 `detmath::exp_f32`（只用 IEEE 保证正确舍入的
   `+ - * /` 与 `floor`，加位操作）。libm 允许末位 ulp 的实现差异，一次采样的概率边界差 1 ulp
   就可能在高熵 token 上翻转选择 —— 换机器后"同 seed 同轨迹"失效，且 99.999% 的 token 一样，
   极难发现。诊断量（如 `candidate_mass`）允许用 libm，因为它不参与决策。
2. **随机数必须是 `(seed, position)` 的纯函数**（计数器式，不是流式）。流式 RNG 的结果依赖
   "之前抽了多少次"，而抽了多少次取决于批里有几行、有没有抢占重算、padding 占了几位 ——
   于是同一请求在不同 batch 组合下得到不同 token，§6.3 的分级不变性直接崩掉。

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

### 11.1 测量协议（`native/harness/bench.h`，B-04）

RSI 的每一步"变快了"都要能站住，所以测量协议本身是架构契约，不是实现细节。

| 约定 | 取值 | 理由 |
| --- | --- | --- |
| warmup | 5 | 让频率/缓存进稳态；0 次预热的首样本是垃圾数据 |
| samples | 7 | 配对样本数；7:0 全胜时精确符号检验 p = 0.015625 < 0.05 |
| iterations / sample | 8 | 分组计时，把单次取时钟的几十 ns 摊薄 |
| 计时单位 | 每迭代纳秒 = 分组总时长 / iterations | 直接取单次迭代会被时钟开销系统性抬高 |
| 交错方式 | **ABBA** | A、B 在时间轴上的平均位置相同，残余线性漂移不偏向任一方 |
| 裁决量 | **逐样本配对差值**，取中位数 | 独立比两个中位数抵不掉慢变量（温度、邻居租户） |
| 显著性 | **精确双尾符号检验**，α = 0.05 | 只有 7 个样本，百分位比较没有分辨力 |
| 报告 | 中位数 + min/max，**禁用 p99** | n=7 时 `ceil(0.99*7)=7`，p99 就是最大值 |

**硬约束**：`sign_test_p > α` 一律判 `indistinguishable`，无论中位数差多少。
微基准噪声足以稳定地"测出"1–3% 的假提升；放行等于让 RSI 去追噪声。
`paired_report_json` 里的 `p95_degenerate` 字段把"这个分位数没有分辨力"这件事写进机器可读的输出，
避免下游把 p95 当成分位数用。

时钟是可注入的（`Clock` = `std::function<uint64_t()>`），因此以上全部算术在无卡机器上
可被逐位断言 —— 这是本协议唯一能在 CI 里存在的形式。设备计时的 `Sync` 钩子必须提供，
否则读到的是下发时刻而不是完成时刻。

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
