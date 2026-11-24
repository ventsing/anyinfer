# dev_anyinfer

Ascend 原生推理引擎的开发工作区。**纯 Rust 控制面 + Ascend C 自研算子的裸金属运行时**，目标是让引擎与算子都能被 RSI（递归自我改进）闭环持续优化，并为 RL rollout 提供可信的数值。

设计基线见 [`docs/DESIGN.md`](docs/DESIGN.md)，历史裁决见 [`docs/DECISIONS.md`](docs/DECISIONS.md)，
当前任务队列见 [`docs/TODO.md`](docs/TODO.md)。

> 前身是 `../anyinfer`（PyTorch + vLLM wrapper 原型）。本仓库**不 vendoring、不依赖** vLLM / PyTorch 运行时；原型仅作为差分测试对照与工具方法论来源。

---

## 当前状态

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| **M-A** | 图捕获 + KV 间接寻址**判决** | 🚧 代码就绪，待上机验证（见下） |
| M-B | 算子 harness（位级/FP64 门 + JSON 裁决 + 熔断） | ⬜ |
| M-C | 玩具模型 + 逐层对拍导出 | ⬜ |
| v0.1 | 单卡端到端生成 + 每步一次 FFI | ⬜ |

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

1. **时间预算**：全量重建 ≤ 10 min，smoke ≤ 60 s。RSI 的迭代频率 = 1/(构建+测量)，超预算即视为架构缺陷。
2. **两把尺子不许混用**：位级相等只用于「同代码同输入复算」与元素级算子；`1e-3` 容差只用于跨实现 / 跨 batch 的端到端比对。
3. **补零只 pad 不 drop**：MoE 容量溢出若丢 token，是语义改变而非数值漂移。
4. **不可自修改面**：测量代码、golden 语料、构建与回退逻辑对 agent 只读且哈希锁定。
