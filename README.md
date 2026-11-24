# dev_anyinfer

Ascend 原生推理引擎的开发工作区。**纯 Rust 控制面 + Ascend C 自研算子的裸金属运行时**，目标是让引擎与算子都能被 RSI（递归自我改进）闭环持续优化，并为 RL rollout 提供可信的数值。

设计基线见 [`docs/DESIGN.md`](docs/DESIGN.md)，历史裁决见 [`docs/DECISIONS.md`](docs/DECISIONS.md)，
当前任务队列见 [`docs/TODO.md`](docs/TODO.md)。

> 前身是 `../anyinfer`（PyTorch + vLLM wrapper 原型）。本仓库**不 vendoring、不依赖** vLLM / PyTorch 运行时；原型仅作为差分测试对照与工具方法论来源。

---

## 当前状态

**这个仓库停在一个刻意的检查点上：所有能纯 host 验证的东西都已经做完并全绿，
剩下的全部指向同一个阻塞点 —— 需要一张真实的 910B。**

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M-A | 图捕获 + KV 间接寻址**判决** | 🚧 代码就绪，**待上机**（唯一的阻塞点） |
| M-B | 算子 harness（位级/FP64 门 + JSON 裁决 + 熔断） | ✅ host 部分完成（B-01..B-04b），B-05..B-07 待设备 |
| M-C | 玩具模型 + 逐层对拍导出 | ✅ 参考侧完成（C-01..C-04），C-05/C-06 待引擎输出 |
| 引擎骨架 v0.1 | ABI / plan / 采样 / KV | ✅ D-01/D-02/D-04 + E-01..E-04 完成；D-03/D-05 待设备 |

```
测试   111 项（Rust）+ 32 项（Python/C++）
CI     34 个步骤全绿，clippy 零告警，rustdoc -D warnings 通过
依赖   Rust 侧零第三方依赖；Python 侧仅标准库
```

### 已完成的部分实测什么

| 已完成 | 它保证了什么 |
| --- | --- |
| D-01 C ABI | `sizeof` 钉死在 24 / 144 字节，零隐式 padding；一份布局产物、Rust 与 C++ **各自独立**校验 |
| D-02 步级 plan | 11 个设备区 + 9 个 host 缓冲在 plan 期各分配一次，**稳态 4096 步零分配**（有正控制证明钩子真在数） |
| D-04 采样器 | 决策路径不用 libm；`logprob` 与 `temperature`/`top_p` 无关（6×4×200 组逐位验证）；计数器式 RNG 使 batch 不变性成为结构性结论 |
| E-01 块分配器 | 双释放、释放被引用块都变成 `Err`；O(1) 栈式自由队列 |
| E-02 前缀树 | 祖先链引用计数（只有它能判断可回收）；`insert` 原子；10 万步随机负载守恒 |
| E-03 序列布局 | 块粒度让 CoW 变成多余（ADR-014）；写侧所有权是不变量而非约定 |
| E-04 三级回收 | 硬顺序 + 换出/抢占反序挑人 + 双窗口抢占率；容量定档 19 块被钉住 |

---

## 目录布局

```
docs/
  DESIGN.md            架构契约（改设计前先改这里）
  DECISIONS.md         ADR，只追加
  TODO.md              任务队列（完成判据 + 依赖）
  TOOLCHAIN.md         本机工具链的实测版本与安装方式
native/graph_probe/    里程碑 A：图捕获探针
  graph_probe.h          C ABI（纯 POD）
  graph_probe_kernel.cpp Ascend C kernel
  graph_probe_launch.cpp 启动器 + 参数校验
  graph_compat.h         图 API 兼容层（唯一的不确定项）
  graph_probe_verdict.h  四臂 -> 裁决映射（host 可测）
  graph_probe_test.cpp   四臂驱动
native/abi/            C ABI 边界（D-01）
  anyinfer.h            AnyInferStep / AnyInferStatus（C11 + C++ 双可用，POD only）
  abi_layout.txt        布局产物：Rust 生成，C 与 Rust 各自独立校验
native/harness/        算子验收 harness（B-01..B-04b，纯 host）
  exact.h                有界二进制有理数 + 精确性判据（L1 的仲裁者）
  bit_exact.h            L1 位级比对
  fp64_reference.h       L2 FP64 参考比对
  report.h               共用 Gate 语义与 JSON 形状
  bench.h                微基准协议（ABBA 配对 + 符号检验 + 退化分位数标记）
  accept.h               算子晋升裁决（三态门，NotRun 阻断晋升）
crates/anyinfer-kv/       KV 内存（E-01..E-04，零依赖）
  src/arena.rs          O(1) 栈式自由队列 + 引用计数（双释放/释放被引用块都变成 Err）
  src/prefix.rs         块粒度前缀树 + 祖先链引用计数 + 子树/叶 LRU 回收
  src/sequence.rs       序列 KV 布局：共享前缀（只读）+ 独占尾巴（见 ADR-014）
  src/reclaim.rs        三级回收决策 + 抢占率记账（见 ADR-015）
  src/simulation.rs     E-01..E-04 端到端仿真（仅测试）
crates/anyinfer-plan/     步级 plan（D-02，零依赖）
  src/lib.rs            PlanConfig / RegionSource / Plan::new（唯一发生分配的地方）
  src/staging.rs        定长 staging + prepare_decode_step（先全校验再写）
  tests/zero_allocation.rs  计数分配器：稳态零分配 + 正控制
crates/anyinfer-sampler/  确定性采样器（D-04，零依赖）
  src/detmath.rs        自带 exp（决策路径不用 libm ⇒ 跨平台逐位一致）
  src/rng.rs            计数器式 RNG：uniform01 是 (seed, position) 的纯函数
  src/lib.rs            sample_row（top-k/top-p/greedy）+ logprob 契约
crates/anyinfer-abi/   与 anyinfer.h 一一对应的 Rust 定义（零依赖）
  src/lib.rs            POD 结构体 + 布局报告生成
  src/bin/abi_dump.rs   生成 abi_layout.txt
  tests/                布局与产物一致性（逐行指出漂移）
ci/
  host-only.sh           无 CANN/NPU/Rust 也能全绿的 CI 入口
tools/
  accept.py              一次算子验收的单命令入口（只提取，裁决交给 C++）
  accept/accept_main.cpp 裁决 CLI：全流程唯一的裁决权威
tools/toy/               玩具模型与 FP64 参考 oracle（纯标准库，无 numpy/torch）
  safetensors_io.py      容器读写（引擎要实现的就是这个格式）
  model.py               模型规格 + FP64 参考前向 + 逐层激活捕获
  export.py              产出权重 / 参考轨迹 / manifest（.part 后原子改名）
  compare.py             对拍执行器（与 C++ Gate 同名同归因顺序）
  selftest.py            对拍器自证：参考vs参考必过、扰动/NaN 必红
  check_lock.py          权重哈希锁校验
  perturb.py             制造负向样本
tests/
  golden/toy_corpus.json       语料 + 容差表
  golden/toy_manifest.lock.json 权重哈希锁（只锁权重，理由见 check_lock.py）
  graph_probe_logic_test.cpp  裁决逻辑自测（无需 NPU，CI 必跑）
  harness_exact_test.cpp      精确性判据 + 正/负向对照
  harness_compare_test.cpp    L1/L2 比对器的语义边界
  harness_bench_test.cpp      微基准协议（注入假时钟，全确定性）
  harness_accept_test.cpp     裁决策略（默认 NotRun、三态优先级、守护线）
  abi_layout_test.cpp         C 侧布局 + 零隐式 padding（对着 abi_layout.txt）
  toy_reference_test.py       参考实现/容器/对拍器（26 项）
```

---

## 快速开始

> 本机工具链的实测版本与安装方式见 [`docs/TOOLCHAIN.md`](docs/TOOLCHAIN.md)。
> 本机 `cmake` / `ninja` 装在 `~/.local/bin`，**默认不在 PATH 上**：
> `export PATH="$HOME/.local/bin:$PATH"`

### 1. host-only（任何机器，无需 CANN / NPU）

```bash
export PATH="$HOME/.local/bin:$PATH"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

只裸编译也是可以的（不依赖 cmake）：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -I native/graph_probe \
    tests/graph_probe_logic_test.cpp -o build/graph-probe-logic-test
./build/graph-probe-logic-test
```

覆盖「四臂结果 → 七种裁决」的完整映射。**这一层必须在没有设备的 CI 里永远通过**：
裁决写错比 kernel 写错更隐蔽，也更贵。

### 2. 设备侧（需 CANN + 一张卡）

```bash
cmake -B build -DANYINFER_WITH_CANN=ON -DANYINFER_SOC=ascend910b1
cmake --build build -j
./build/bin/anyinfer-graph-probe-test --output graph-probe.json
```

或 `ctest --test-dir build --output-on-failure -R graph-probe`。

**上机前先花 10 分钟确认两件事**（这也是 `docs/DESIGN.md` §12 的 Q1/Q2）：

```bash
grep -nE "StreamBeginCapture|StreamEndCapture|GraphInstantiate|GraphLaunch|GraphExec|GraphGetNodes" \
  "$ASCEND_HOME_PATH/include/acl/"*.h
grep -rn "aclmdlRI" "$ASCEND_HOME_PATH/include/" | head
```

结论决定 `native/graph_probe/graph_compat.h` 里 `ANYINFER_GRAPH_API` 填 `1` 还是 `2`。
**这是本里程碑唯一无法由代码自行确定的未知项**，所以它被刻意收在一个可替换的边界里。

---

## 上机第一天该跑什么（按顺序，前一步不绿就不要做下一步）

这一节存在的理由：**上机时间是稀缺资源**。下面每一步都写了"不通过意味着什么"，
因为失败时最贵的不是修，是不知道该修哪儿。

### 第 0 步：确认环境（10 分钟，不写代码）

```bash
npu-smi info                      # 卡在不在，型号对不对
echo "$ASCEND_HOME_PATH"          # 不能是空
ls "$ASCEND_HOME_PATH/include/acl/" | head
```

**不通过意味着**：环境问题，后面所有结论都不可信。不要往下走。

### 第 1 步：解决唯一的未知项（A-01）

```bash
grep -nE "StreamBeginCapture|StreamEndCapture|GraphInstantiate|GraphLaunch|GraphExec|GraphGetNodes" \
  "$ASCEND_HOME_PATH/include/acl/"*.h
grep -rn "aclmdlRI" "$ASCEND_HOME_PATH/include/" | head
```

按结果把 `native/graph_probe/graph_compat.h` 的 `ANYINFER_GRAPH_API` 填成 `1`（`aclrtGraph*`）
或 `2`（`aclmdlRI*`）。**这是本仓库唯一一处靠猜会全盘皆错的地方**，所以它被刻意关在一个
可替换的边界里 —— 填错只影响这一个头文件。

### 第 2 步：编译并跑 A-02 四臂探针

```bash
cmake -B build -DANYINFER_WITH_CANN=ON -DANYINFER_SOC=ascend910b1
cmake --build build -j
ctest --test-dir build --output-on-failure -R graph-probe
```

看 `graph-probe.json` 的 `verdict`：

- **`indirection_supported`** → 主路径成立，按 DESIGN §5 推进 Phase 2。**这是好消息。**
- `content_folded` → 真折叠。**不要改代码去绕**，改架构：转后备 A′/B/C，并在
  `docs/DECISIONS.md` 记 ADR —— 这条会推翻 ADR-005，是整个项目最贵的分支。
- 其余四种 → 都是"还没测到位"，不是架构结论。按 README 上一节的表逐项排查。
  尤其：**没有控制臂的失败是不可归因的失败**，别急着下结论。

### 第 3 步：只有第 2 步绿了，才做 A-06（CoW 两行探针）

A-06 是里程碑 A 真正的收尾：两个活跃序列共享前缀、分叉后各写一行、断言互不污染。
它验证的是 E-03 在 host 上证明过的那条不变量在设备上是否也成立。

### 上机前先在 host 上跑一遍

```bash
./ci/host-only.sh          # 34 步，应该全绿
```

如果这一步在设备机上红了，那是环境差异（编译器版本、glibc），
先在移植问题上花时间，别把环境问题误判成设备问题。

---

## 里程碑 A 的裁决怎么读

输出一行 JSON，`verdict` 即结论：

| verdict | 含义 | 下一步 |
| --- | --- | --- |
| `indirection_supported` | 换表内容能被重放看到 —— 主路径成立 | 按 DESIGN §5 推进 Phase 2 |
| `kernel_broken` | 不捕获直接下发就错 | 与图无关，修 kernel |
| `empty_graph` | 捕获到 0 个节点 | 该下发路径不可捕获 → 后备 A' |
| `capture_path_broken` | 控制臂（直接地址）失败 | 查流顺序、捕获区内是否混入 sync |
| `first_replay_stale` | 首次重放读错块 | 查 block table 的 H2D 排序 |
| `replay_not_executing` | 重放后输出仍是哨兵 | 查 instantiate/launch |
| **`content_folded`** | 换了表仍读到旧块 | **真折叠** → 转 A'/B/C |

退出码：`0` 仅当 `indirection_supported`。

### 为什么是四臂 + 哨兵

「重放后输出还是旧值」这一现象，**至少**对应三种不同故障：kernel 没被捕获、图没执行、H2D 没生效。
如果只写一个"两次重放比对"的测试，这三种会和"指针被折叠"混在一起，得出错误的架构结论。
所以：每次重放前用 `aclrtMemset(..., 0xEF, ...)` 写哨兵；先跑不捕获的 Arm 0 和直接地址的 Arm 1；
并断言节点数 ≥ 1。**没有控制臂的失败是不可归因的失败。**

---

## 与 `../anyinfer` 的关系

**继承**：位级校验方法论、纯 POD C ABI 范式、paired A/B 度量纪律、哈希链溯源、必需算子清单。

**切断**：vLLM 依赖、PyTorch 运行时、`reference/` 对比栈。

**只读引用**：`../anyinfer/operators/vllm-ascend/`（算法参考与差分对照）。
其中 `vllm_ascend/compilation/updatable_graph.py` 是里程碑 A 的关键依据 ——
它是生产代码，证明 CANN 提供图节点参数重绑（`graph_task_update_begin/end`），
即"捕获不折叠，但 ACLNN 算子需要显式重绑"。

---

## 工程纪律

0. **检查点纪律**：能在 host 上验证的东西没验完，就不写设备代码。没编译过的设备代码
   不是"进步"，是**未经验证的假设**，而它会在上机那天一次性全部爆掉。
1. **时间预算**：全量重建 ≤ 10 min，smoke ≤ 60 s。RSI 的迭代频率 = 1/(构建+测量)，超预算即视为架构缺陷。
2. **两把尺子不许混用**：位级相等只用于「同代码同输入复算」与元素级算子；`1e-3` 容差只用于跨实现 / 跨 batch 的端到端比对。
3. **补零只 pad 不 drop**：MoE 容量溢出若丢 token，是语义改变而非数值漂移。
4. **不可自修改面**：测量代码、golden 语料、构建与回退逻辑对 agent 只读且哈希锁定。
5. **lint 基线必须干净**：clippy 零告警是 CI 的硬步骤。长期带告警的仓库没法区分
   "新引入的"和"一直有的"，而 RSI 自检恰恰需要这条区分才能判断一次改动是否越界。

## 许可

Apache-2.0，见 `LICENSE`。代码来源、第三方依赖范围与版权状态见 `NOTICE`
（当前版权持有方是占位符，对外分发前必须替换）。
