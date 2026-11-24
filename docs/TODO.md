# 任务队列

> 最后更新：2026-11-24 ｜ 维护规则见文末。
>
> 分工：**里程碑与硬门**在 `DESIGN.md` §9；**这里只放可勾选的具体任务**。
> 两者冲突时以 `DESIGN.md` 为准，并修正本文件。

图例：✅ 完成 ｜ 🚧 进行中 ｜ ⬜ 待做 ｜ ⛔ 被阻塞 ｜ 🅿️ 挂起（已决定不做/延后）

---

## 当前焦点

```
A 轨道（等硬件）：A-01 上机预检 ──> A-02 编译 ──> A-03 四臂判决 ──┬─> [supported] ──> E-01…
                                                                └─> [其它] ──────> A-04

B 轨道（本机可推进）：B-01 ✅ ─> B-02 ✅ ─> B-03 ✅ ─> B-04 ✅ ─> B-04b ✅ ─> B-06 hc_post（需上卡）
C 轨道（本机可推进）：C-01 ✅ ─> C-02 ✅ ─> C-03 ✅ ─> C-04 ✅ ─> C-05 引擎侧接线（等 D-03）
```

**Track B 的 host 侧已全部完成**（B-01..B-04b）。剩余 B-05/B-06/B-07 都需要上卡。
Track D 与 E-01..E-03 的 host 部分已全部做完。剩余 D-03/D-05、E-04 的设备侧测量、E-05/E-06、B-05..B-07、A 轨道、C-05 全部指向同一阻塞点：需要真实 910B 编译机。

---

## Track A — 里程碑 A：图捕获 / 间接寻址判决

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| A-01 | 上机预检：确认 CANN 暴露哪套图 API | `grep` 结果写进 `DESIGN.md` §12 Q1，`ANYINFER_GRAPH_API` 已定值并编译通过 | — | 0.2 h |
| A-02 | 设备侧编译并跑通 Arm 0 | `anyinfer-graph-probe-test` 在 Arm 0 退出前不报 `kernel_broken`；同时确认 `DataCopy` count 语义（Q2） | A-01 | 2 h |
| A-03 | 跑完四臂，拿到 verdict | `graph-probe.json` 落盘，`verdict` 写入本文件「判决记录」 | A-02 | 0.5 h |
| A-04 | 若未通过，实现后备探针 A'（`graph_task_update` 等价物） | 用 ACLNN 算子（如 `aclnnFusedInferAttention`）演示指针重绑成功，并量出每步 update 开销 | A-03 判定为 `content_folded`/`empty_graph` | 3 d |
| A-05 | verdict 接入 CI | 有卡 runner 上 `ctest -R graph-probe` 通过；失败时给出 `remedy` 字段 | A-03 | 0.5 d |
| A-06 | 探针 v2：两序列 CoW 分叉读不同 block | 单图内两行分别指向不同物理块，且换表后各自正确；这是 Phase 2 的真实形态 | A-03 | 1 d |

> **A-06 是 M-A 的真正终点**。A-03 只证明「单行单块」连通，A-06 才证明「同一张图里多序列指向不同 block」。

---

## Track B — M-B：算子 harness（RSI 第一道门）

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| B-01 | ✅ 精确输入生成器（有界 dyadic） | `native/harness/exact.h`：`product_is_exact` / `sum_is_exact` / `dot_is_exact` + 确定性生成；正向对照（顺序无关）与**负向对照**（{2^24,1,1} 两种结合律给出不同位模式）均通过 | — | 完成 |
| B-02 | ✅ L1 位级比对器 + JSON 裁决 | `native/harness/bit_exact.h`：位模式语义（+0/-0、NaN 负载均判不同）、`NonFinite` 独立归类、JSON 可解析 | B-01 | 完成 |
| B-03 | ✅ L2 FP64 参考比对器 | `native/harness/fp64_reference.h`：`atol + rtol*abs(ref)` 判据、参考非法归因到 `InvalidReference`、零容差退化用于确定性自检 | B-02 | 完成 |
| B-04 | ✅ 微基准：warmup 5 / samples 7 / iterations 8 + 中位数 + **paired A/B** | `native/harness/bench.h`：时钟可注入（故协议在无卡机上可逐位断言）；ABBA 交错 + 配对差值抵掉漂移；**精确符号检验**代替小样本百分位；`p95_degenerate` 显式标记（n=7 时 p95=p99=max） | B-02 | 完成 |
| B-04b | ✅ 把三件套串成「一次算子验收」入口 | `accept.h`（三态门 + 裁决）+ `anyinfer-accept`（唯一权威 CLI）+ `tools/accept.py`（只提取不聚合）；单命令产出三段 JSON，退出码 0/1/2/64 语义分离；`NotRun` 阻断晋升 | B-03, B-04 | 完成 |
| B-05 | 熔断与回退：双版本 `.so` + 运行期切换 | 故意注入坏 kernel → 自检失败 → 自动切 stable 且服务不断 | B-02 | 3 d |
| B-06 | 首个目标算子：`hc_post` 参数化 + tile/schedule 搜索 | 三条门全绿（L1 位级 + 微基准不劣化 + 端到端无回归） | B-03, B-04 | 3 d |
| B-07 | 搜索驱动脚本（RSI 闭环最短验证） | 给定参数空间自动跑 N 组，按 KPI 排序并输出推荐 | B-06 | 2 d |

> **B-06 + B-07 是「RSI 闭环能不能闭合」的最小证据**。它不依赖 Track A，可独立推进。

---

## Track C — M-C：玩具模型与对拍弹药

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| C-01 | ✅ 权重导出（safetensors → 裸二进制 + 清单） | `tools/toy/safetensors_io.py`（纯标准库，写 .part 后原子改名）+ `model.py` 的 dyadic 生成器；实测权重逐位可被 fp32 精确表示，跨进程/跨次导出哈希一致 | — | 完成 |
| C-02 | ✅ FP64 参考导出（逐层激活 + logprob） | `model.py` 的 `reference_forward`：28 个命名张量（含 RoPE 前/后分开导出）；因果性、RoPE 保范数、logprob 归一化均被断言 | C-01 | 完成 |
| C-03 | ✅ 语料与容差表 | `tests/golden/toy_corpus.json`（4 条语料 + 前缀容差规则）+ `tests/golden/toy_manifest.lock.json`（只锁权重哈希，理由见 check_lock.py） | C-01 | 完成 |
| C-04 | ✅ 对拍执行器 | `tools/toy/compare.py`：与 C++ `Gate` 同名同归因顺序；`selftest.py` 自证"参考vs参考必过、扰动必红、NaN 必红" | C-02 | 完成 |
| C-05 | 引擎侧导出同名张量的接线 | 引擎能在 `layer{i}.norm1` 等同一批位置导出 F32 张量，直接喂 `compare.py` | D-03 | 2 d |
| C-06 | 容差表实测校准 | 用真实 fp32 引擎输出实测每层 max_rel，把 L2 容差从"猜"变为"量出来的" | C-05 | 1 d |

> 选 2–4 层、hidden 512 的玩具模型即可。目的是**定位能力**，不是性能。

---

## Track D — v0.1 引擎骨架

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| D-01 | ✅ Cargo workspace + FFI 层 | `native/abi/anyinfer.h`（C11/C++ 双可用）+ `crates/anyinfer-abi`；`AnyInferStep` / `AnyInferStatus` 定稿含 `failed_op_id`；sizeof 硬钉（24/144）；布局产物 `abi_layout.txt` 由两侧独立校验；C 侧断言零隐式 padding | — | 完成 |
| D-02 | ✅ plan 阶段一次性分配 | `crates/anyinfer-plan`：11 块设备区域 + 9 块 host 缓冲各申请一次；稳态 4096 步断言 alloc/realloc/free 全 0；带正控制（证明钩子没哑）与负向对照（插一次分配立刻红）；padding 两端隔离、块表无残留、半写不落地、slot 由块表推导 | D-01 | 完成 |
| D-03 | prefill 逐 op 序列 + 缓存 `aclOpExecutor` | 复用 executor 时 `aclTensor` 句柄一并缓存（否则 use-after-free）；重复下发位级一致 | D-01 | 3 d |
| D-04 | ✅ 采样器：确定性 + top-k/top-p + logprob 导出 | `crates/anyinfer-sampler`：决策路径无 libm（自带 `exp_f32`，相对误差 ~1e-8）+ 计数器式 RNG（`(seed,position)` 纯函数 ⇒ 批次无关）；导出的 logprob 与 T/top_p 无关（专项测试 4800 组交叉验证）；27 项测试 | D-01 | 完成 |
| D-05 | 端到端 L4 门 | token 序列与参考一致；logprob 最大绝对偏差 < 1e-3 | C-04, D-03, D-04 | 2 d |

---

## Track E — v0.2 内存

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| E-01 | ✅ Arena + Block 分配器 | `crates/anyinfer-kv::arena`：block size 部署期参数；O(1) 栈式自由队列；双释放 / 释放被引用块 / 保留块被释放 / 计数下溢 全部变成 `Err`（在设备上这些错误只会让结果依赖历史，不崩不报错） | — | 完成 |
| E-02 | ✅ 块粒度前缀树 + **祖先链**引用计数 | 后代持有整条 root→node 路径的引用；负向用例 `naive_shallow_counting_would_free_a_block_that_is_still_being_read` 把朴素判据的判定与真实后果摆在一起；2000 步随机负载每步校验块守恒、收尾后引用计数归零且一块不剩；两条负向对照证实测试有灵敏度 | E-01 | 完成 |
| E-03 | ✅ 序列 KV 布局（**CoW 被消解**） | `anyinfer-kv::sequence`：共享前缀（只读）+ 独占尾巴（私有）。原判据两条均满足且无需复制：共享前缀只算一次（复用同一批块）、分叉后写不污染兄弟（新 token 落在私有块）。`append_token` 在 `len < shared_tokens` 时拒绝；随机负载 3000 步对**每一次写入**断言目标块独占；负向对照摘掉守卫即红。见 ADR-014 | E-02 | 完成 |
| E-04 | 三级回收：驱逐 / 换出 / 抢占 | 抢占率 = `Recompute Tokens / Total Decoded Tokens` < 5% | E-03 | 3 d |
| E-05 | block table 每步 H2D 与图重放联通 | 与 A-06 合流：树变异后重放仍读对块 | A-06, E-03 | 2 d |
| E-06 | padding 双端隔离 | dummy 行不落 KV（写侧 `slot_mapping` 截断）+ 读侧 trim；附负向用例 | E-01 | 1 d |

---

## Track F — 基础设施与治理（本仓库）

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| F-00 | ✅ 本机工具链 + host-only 构建链 | cmake 4.4.5 / ninja 1.13.2 已装；`cmake -S . -B build -G Ninja` + `ctest` 全绿；两条负向用例（缺 SOC / 无 CANN）按预期报错。见 `docs/TOOLCHAIN.md` | — | 完成 |
| F-01 | ✅ CI：host-only 必跑 | `ci/host-only.sh` 12 项全绿：`bash -n` + 三个测试以 `-Werror -Wpedantic` 编译并运行 + cmake/ctest + 两条 CANN 守卫负向用例 | F-00 | 完成 |
| F-02 | `rust-toolchain.toml`（`rust-version = "1.85"` 已在 workspace 定） | 锁定版本；CI 用同一版本 | D-01 | 0.2 d |
| F-03 | 时间预算守护 | 脚本量测全量重建与 smoke，超 10 min / 60 s 直接失败 | B-06 | 0.5 d |
| F-04 | LICENSE / NOTICE 决策 | 明确 derive-from-Apache-2.0（保留 notices）还是 clean-room；落成文件 | — | 0.2 d |
| F-05 | 不可自修改面清单 | 列出测量代码/语料/构建/回退逻辑，加哈希锁定与写保护检查 | B-05 | 1 d |

---

## Track G — 对原型仓库的欠账（scope: `../anyinfer`）

> 这些不影响新引擎的正确性，但**影响我们对它的信任**。G-02 尤其关键：它会让算子轨道的排期基于错误数据。

| ID | 任务 | 完成判据 | 估时 |
| --- | --- | --- | --- |
| G-01 | 删除根目录 `qa`（0 B）与 `qa.tgz` | 工作树干净 | 0.1 h |
| G-02 | 修 `tools/operator_catalog.py:24` 深度敏感 glob | 204 → 229 命中；补 `310P` 列；断言 catalog 与 `inventory.tsv` 一致 | 0.5 d |
| G-03 | 加 `LICENSE` / `NOTICE` | 有明确授权文件（当前完全缺失） | 0.5 h |
| G-04 | 代码里校验 `SOURCE.json` 的 `archive_sha256` | 目前**从不校验**；加校验并要求失败即中止 | 0.5 d |
| G-05 | 去掉 `build.sh:119` 的 `mktemp` 随机后缀 | 产物名可复现 | 0.2 h |
| G-06 | `docs/*.pdf` 移出 git | 1.8 MB 孤儿二进制移出；同时查明 `user-manual.pdf` 内容实为 reference-manual 的问题 | 0.5 h |
| G-07 | 给 `publish_release.py` / `apply_overlay.py` / `operator_integrity.py` 加测试 | 三处失败路径（锁冲突、路径逃逸、哈希不符）有测试覆盖 | 1 d |

---

## 未决问题（阻塞部分任务）

| # | 问题 | 阻塞 | 验证方式 |
| --- | --- | --- | --- |
| Q1 | CANN 图 API 符号 | A-02 | `grep` 头文件（A-01） |
| Q2 | `DataCopy` 第二参数语义 | A-02 | 查头文件注释 |
| Q5 | top-k 回退触发率 | D-04 调参 | 玩具模型上统计 |
| Q6 | 图桶 padding 浪费比 | 图桶档位选择 | 同上 |

---

## 判决记录

| 日期 | 里程碑 | 判决 | 依据 |
| --- | --- | --- | --- |
| 2026-11-24 | E-03 | ✅ 通过 | 原计划"CoW 分叉"被 E-02 的块粒度决定消解：块粒度下共享段与写入点结构性不相交，复制从未被需要。改为满足判据本身（共享前缀只算一次 + 写不污染兄弟）+ 一条写入侧所有权守卫；3000 步随机负载逐次写入断言独占；摘掉守卫即红。记 ADR-014 |
| 2026-11-24 | E-01/E-02 | ✅ 通过 | 25 项测试；两条负向对照均触发（detach 漏一块 -> 引用计数 405≠0；祖先链退化成只计自身 -> 核心断言立刻红）；随机负载 2000 步块守恒 |
| 2026-11-24 | D-02 | ✅ 通过 | 22 项测试；插一次 `vec![]` 进热路径 -> 稳态用例报 allocate 4096（每步一次）并红；plan 期分配次数钉死为 9（加缓冲必须显式改这个数）；staging 长度与设备预留大小交叉校验（两条独立公式） |
| 2026-11-24 | D-04 | ✅ 通过 | 27 项测试；logprob 恒等式在 6×4×200 组 (T, top_p, position) 交叉下逐位成立；top-p 截断点在均匀分布上锐利（1 万次抽样外候选零次出现，对照臂证明可达）；经验分布与解析概率 4σ 内一致 |
| 2026-11-24 | D-01 | ✅ 通过 | 布局产物单一来源、两侧独立校验：C 侧改字段顺序 -> 精确报出两个字段的偏移；改产物 size -> Rust 报出行号、C 侧同时报出 trailing padding；缺 cargo 时显式 skip 而非静默跳过 |
| 2026-11-24 | B-04b | ✅ 通过 | 26 项 accept 断言 + CI 端到端 8 步；缺证据退 2、失败退 1、用法错误退 64（刻意与 2 分离）；参考坏掉时归因指向 oracle |
| 2026-11-24 | B-04 | ✅ 通过 | 注入假时钟后协议完全确定：分组除算、预热不计入、ABBA 顺序、7:0 -> p=0.015625 均被逐位断言；4:3 噪声与全平局均判 `indistinguishable`（负向对照） |
| 2026-11-24 | C-01…C-04 | ✅ 通过 | 26 项 unittest 全绿；export 两次哈希一致；自比对 pass、+0.5% 扰动与 NaN 注入均被精确归因（只报被改的张量） |
| 2026-11-24 | B-01/02/03 | ✅ 通过 | 三份 host-only 头文件 + 两个测试；精确性判据带正/负向对照；L1/L2 语义边界（±0、NaN 负载、rtol 尺度）逐条钉住 |
| 2026-11-24 | F-01 | ✅ 通过 | `ci/host-only.sh` 12/12 全绿（含 `-Werror -Wpedantic` 编译与两条 CANN 守卫负向用例） |
| 2026-11-24 | F-00 工具链 | ✅ 通过 | cmake 4.4.5 + ninja 1.13.2，host-only 配置/构建/`ctest` 全绿；缺 SOC 与无 CANN 两条负向用例按预期报错 |
| 2026-11-24 | A（代码就绪） | 待上机 | host-only 裁决逻辑 10/10 通过；device 目标未编译 |

---

## 维护规则

1. **完成即勾选，不要批量勾选**。`✅` 意味着「完成判据」那条真的成立，且有据可查（日志/JSON/测试输出）。
2. 新增任务必须带**完成判据**。写不出判据的任务说明还没想清楚，放到「未决问题」。
3. `⛔` 必须写明阻塞在哪个 ID 或哪个 Q。
4. 里程碑级变更（新增/删除阶段、改硬门）→ 先改 `DESIGN.md`，再改本文件。
5. 推翻既有决策 → 追加 `DECISIONS.md` 的 ADR，不在本文件里争论。
6. 每周复盘一次 `🅿️`：挂起项要么给日期，要么删掉。
