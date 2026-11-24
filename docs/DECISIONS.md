# 决策记录（ADR）

规则：**只追加，不修改历史条目**。要推翻旧决策，追加一条新 ADR 并标注 `supersedes: ADR-xxx`。

格式：决策 / 背景 / 被否决方案 / 后果。

---

## ADR-001 定位收敛到本地 RSI，而非云端 serving

- **背景**：早期蓝图把"Host 下发延迟从毫秒压到微秒"当作核心卖点，目标是 V4 级 MoE 的云端 serving。
- **决策**：定位改为单机本地 RSI / RL rollout 闭环；主 KPI 改为 p99 TTFT、采样吞吐、前缀命中率。
- **被否决**：以 host 延迟为卖点做云端 serving —— 在 30 ms 步长下 host 优化收益 < 1%，且 vLLM 0.30 已自带 Rust 前端（`VLLM_USE_RUST_FRONTEND`），无差异化。
- **后果**：单卡 64 GB 包络，"秒级启动"才成立；V4 与多卡移出基线。

## ADR-002 不复用 vLLM 执行路径，只作差分测试对照组

- **背景**：原型是 PyTorch + vLLM 的 Rust wrapper，vendoring 了 vllm-ascend 的全部源码。
- **决策**：新引擎不 vendor、不依赖 vLLM/PyTorch 运行时；vLLM 仅作 oracle 对照。
- **被否决**：继续以 vLLM 为引擎、只优化算子 —— 无法满足"引擎自研"，且被上游版本钉死（`bootstrap.py` 硬校验 0.30.0）。
- **后果**：需要自建 tokenizer、sampler、scheduler、KV 管理；工作量大幅上升，但可控。

## ADR-003 FFI 粒度：每步一次，不是每 op 一次

- **背景**：早期草图在 Rust 侧逐 op 调 `launch_aclnn_*`。
- **决策**：`anyinfer_engine_step(engine, step*, status*)` 每步跨 FFI 一次；per-op 序列在 C++ 内展开。
- **被否决**：按 op 跨 FFI —— 32 层 × 十几个算子 = 每 forward 数千次跨界，"消除边界开销"的叙事自我拆台。
- **后果**：C++ 侧需要一个静态图/plan 结构；Rust 侧只持有不透明句柄。

## ADR-004 C ABI 只允许 POD，禁止 STL 类型跨界

- **背景**：曾建议用 `-D_GLIBCXX_USE_CXX11_ABI=0` 适配 Cloud EulerOS。
- **决策**：边界上只放 POD、裸指针、标量；函数签名照抄原型 `hc_post.h` 的风格。
- **被否决**：调 ABI 宏 —— `=0` 与 torch_npu/CANN 的 `=1` 冲突，在保留 libtorch 的阶段必然链接失败；且只要边界无 C++ 类型，该宏就无关。
- **后果**：所有跨边界信息必须在 C 结构体里显式表达，不能传 `std::vector`/`std::string`。

## ADR-005 KV 走设备侧 block table 间接寻址

- **背景**：CoW 分叉、驱逐、换出都会让同一序列的 KV 物理地址在两步之间改变，而图重放要求地址静态。
- **决策**：物理块地址由设备侧 block table 缓冲间接给出；每步只 H2D 一个小的索引数组到固定地址。
- **被否决**：host 侧拼裸指针 —— 与 CoW 语义不兼容，且使图捕获不可用。
- **后果**：block table 布局成为核心数据结构；图捕获可重放的前提是"只改内容、不改地址"。

## ADR-006 图捕获边界止于采样前，但延伸到 top-k + logsumexp

- **背景**：grammar FSM 依赖已生成 token，是数据相关分支，与静态图冲突。
- **决策**：捕获区覆盖 embed → layers → lm_head → mask 应用 → logsumexp → top-k；FSM 与采样留在 host。
- **被否决（一）**：捕获止于 logits —— 每步 D2H 16.4 MB（128k 词表 × batch 32 × fp32）并强制流同步，3–5% 开销且打断预取。
- **被否决（二）**：把 FSM 搬到设备侧 —— 工作量过大，且丧失 FSM 可迭代性。
- **后果**：FSM 的产物必须表达为 vocab 维**位掩码**（16 KB），走固定地址设备缓冲，图内只读不写；top-k 必须足够大（≥256）并提供全量 logits 的慢路径回退。

## ADR-007 MoE 固定容量：只 pad，绝不 drop

- **背景**：fixed capacity 是专家 GEMM shape 静态化的前提，而静态 shape 是图捕获的前提。
- **决策**：容量上界 `max_num_seqs × top_k`，溢出只 padding；prefill 走动态 shape 不受此约束。
- **被否决**：超容量丢弃 token —— 改变模型输出，属语义错误而非数值漂移，PPO 会将其误读为策略行为。
- **后果**：decode 的专家 GEMM 可被捕获；capacity factor 成为部署期参数。

## ADR-008 `weight_version` 同时是 prefix cache 的 epoch

- **背景**：Drain & Swap 保证不在一个 step 内混合权重，但 Radix Tree 里的旧 KV 仍是旧权重产出。
- **决策**：版本变更即整体失效前缀缓存，并把重新 prefill 系统提示词的耗时计入 reload 预算。
- **被否决**：保留旧前缀、只换权重 —— 旧权重 KV + 新权重计算 = 静默污染，且不报错。
- **后果**：`T_reload = W/BW + T_reprompt`；reload 频率应由 RL 训练节奏决定，而非追求更快。

## ADR-009 热重载放弃"隐藏 I/O"，只保证语义正确

- **背景**：早期设计声称"分层流式替换 + 1~2 层冗余 buffer 可隐藏 I/O"，但同文又给出 `T ≈ W/BW`。
- **决策**：接受 `T ≈ W/BW` 的硬暂停（本地 NVMe ~7 s，RDMA ~4 s），把工程投入放在版本切换的正确性上。
- **理由**：50 GB 权重 / 30 ms 步长 ⇒ 需要 ~2 TB/s 才能隐藏，物理上不存在；且 Drain 后引擎空闲，没有计算可重叠。
- **后果**：reload 排到自然屏障；真正的省时手段只有 delta/LoRA 更新与多机交替。

## ADR-010 batch invariance 分级，而非绝对

- **背景**：要求"batch=1 与 batch=32 绝对一致"，同时要求前缀 KV 只算一次并共享。
- **决策**：decode dense 路径严格不变（固定 tile / 固定累加顺序 / 无 split-K / 无 atomic）；MoE 路由与 prefix 共享允许容差。
- **理由**：被共享的前缀 KV 是在某次特定 batch shape 下算出的，共享即注入"别人的归约顺序"；两条要求在数学上冲突。
- **后果**：验收门写成 `max|Δlogprob| < 1e-3 ∧ clip_rate < 1%`，而不是"绝对一致"。

## ADR-011 自研算子优先 Ascend C 裸指针，ACLNN 次之

- **背景**：`aclTensor` 描述符在 `GetWorkspaceSize` 阶段就把 `device_ptr` 写进宿主内存描述符，图捕获会固化它。
- **决策**：主路径用 Ascend C kernel（`ACLRT_LAUNCH_KERNEL`，参数是裸 `GM_ADDR`）；必须复用 ACLNN 内置算子时，走 A'（指针重绑）。
- **依据**：`../anyinfer/operators/vllm-ascend/vllm_ascend/compilation/updatable_graph.py` 中的 `graph_task_update_begin/end` 即生产级重绑机制。
- **后果**：算子轨道优先自研，减少对 ACLNN 的依赖；A' 只在 attention 等少数路径保留。

## ADR-012 测试必须有哨兵与控制臂，否则结论不可归因

- **背景**：早期草案以"重放后仍是旧值"直接判定"指针被折叠"。
- **决策**：每次重放前写哨兵（`aclrtMemset(..., 0xEF, ...)`）；先跑不捕获的 Arm 0 与直接地址的 Arm 1；断言图节点数 ≥ 1。
- **理由**："旧值"同样出现在"kernel 未捕获""图未执行""H2D 未生效"三种情形，无哨兵不可区分。
- **后果**：裁决从二元变为可归因的七态表（见 DESIGN §10）。

## ADR-013 自举安全：位级自检 + 运行期回退

- **背景**：RSI 允许 agent 修改引擎自身；一次坏修改可能让底座永久宕机。
- **决策**：kernel 双版本；启动与运行期都做**位级**自检（不是 1e-3 容差）；回退可在 step 边界触发；测量/语料/构建/回退逻辑对 agent 不可写。
- **被否决**：用 `max|Δlogprob| > 1e-3` 作为回退阈值 —— 那是跨实现容差，用在同实现复算上会掩盖确定性 bug。
- **后果**：确定性采样成为硬需求（否则位级自检不成立）。
