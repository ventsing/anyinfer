# 任务队列

> 最后更新：2026-11-24 ｜ 维护规则见文末。
>
> 分工：**里程碑与硬门**在 `DESIGN.md` §9；**这里只放可勾选的具体任务**。
> 两者冲突时以 `DESIGN.md` 为准，并修正本文件。

图例：✅ 完成 ｜ 🚧 进行中 ｜ ⬜ 待做 ｜ ⛔ 被阻塞 ｜ 🅿️ 挂起（已决定不做/延后）

---

## 当前焦点

```
A-01 上机预检 ──> A-02 编译 ──> A-03 四臂判决 ──┬─> [indirection_supported] ──> E-01…
    (0.2h)          (2h)          (0.5h)        └─> [其它 verdict] ──────────> A-04 后备探针

并行（不依赖 A）：
  C-01/C-02 玩具模型导出（纯 Python，无卡可做）
  F-01 CI 骨架
```

**下一动作**：A-01 —— 编译机上两条 `grep`，10 分钟出结论，决定 `graph_compat.h` 的唯一未知项。

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
| B-01 | 精确输入生成器（有界 dyadic） | `native/include/exact.h`：乘积与求和可精确表示；含负向用例证明「非精确输入会使位级相等不成立」 | — | 0.5 d |
| B-02 | L1 位级比对器 + JSON 裁决 | 复用 `graph_probe_verdict.h` 的裁决风格；输出 `pass`/`mismatch@index` | B-01 | 1 d |
| B-03 | L2 FP64 参考比对器（GEMM/attention 类） | 相对误差 + max_abs；不再声称位级 | B-02 | 1 d |
| B-04 | 微基准：warmup 5 / samples 7 / iterations 8 + 中位数 + **paired A/B** | 同卡同进程交错测得两版本差异；报告分位数而非均值 | B-02 | 2 d |
| B-05 | 熔断与回退：双版本 `.so` + 运行期切换 | 故意注入坏 kernel → 自检失败 → 自动切 stable 且服务不断 | B-02 | 3 d |
| B-06 | 首个目标算子：`hc_post` 参数化 + tile/schedule 搜索 | 三条门全绿（L1 位级 + 微基准不劣化 + 端到端无回归） | B-02, B-04 | 3 d |
| B-07 | 搜索驱动脚本（RSI 闭环最短验证） | 给定参数空间自动跑 N 组，按 KPI 排序并输出推荐 | B-06 | 2 d |

> **B-06 + B-07 是「RSI 闭环能不能闭合」的最小证据**。它不依赖 Track A，可独立推进。

---

## Track C — M-C：玩具模型与对拍弹药

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| C-01 | 权重导出（safetensors → 裸二进制 + 清单） | 清单含每张量 offset/shape/dtype + 整体 sha256；越界与重叠校验 | — | 1 d |
| C-02 | PyTorch 参考导出：`register_forward_hook` 固定 seed | 逐层激活 + 逐 token logprob 落盘；同 seed 重跑位级一致 | C-01 | 1 d |
| C-03 | 语料与容差表 | 语料 < 1 MB 进 git；每层激活的 L2 相对误差门写入 `DESIGN.md` §6.1 | C-02 | 0.5 d |
| C-04 | 对拍执行器 | 引擎输出 vs 参考输出 → 定位到**具体层 + 具体 token** | C-02 | 2 d |

> 选 2–4 层、hidden 512 的玩具模型即可。目的是**定位能力**，不是性能。

---

## Track D — v0.1 引擎骨架

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| D-01 | Cargo workspace + FFI 层 | `AnyInferStep` / `AnyInferStatus` 定稿（含 `failed_op_id`）；C 侧 `static_assert` 尺寸与对齐 | — | 2 d |
| D-02 | plan 阶段一次性分配 | 零分配断言（沿用原型 `tests/host_decode_allocations.rs` 手法） | D-01 | 1 d |
| D-03 | prefill 逐 op 序列 + 缓存 `aclOpExecutor` | 复用 executor 时 `aclTensor` 句柄一并缓存（否则 use-after-free）；重复下发位级一致 | D-01 | 3 d |
| D-04 | 采样器：确定性 + top-k/top-p + logprob 导出 | 同 seed 同轨迹；logprob 用 fp32 且归约顺序与训练端一致 | D-01 | 2 d |
| D-05 | 端到端 L4 门 | token 序列与参考一致；logprob 最大绝对偏差 < 1e-3 | C-04, D-03, D-04 | 2 d |

---

## Track E — v0.2 内存

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| E-01 | Arena + Block 分配器 | block size 为部署期参数（不硬编 16/32）；O(1) 队列分配 | A-03 | 2 d |
| E-02 | Radix tree + **祖先链**引用计数 | 后代持有全部祖先引用；构造「淘汰中间节点会毁活跃请求」的负向用例 | E-01 | 3 d |
| E-03 | CoW 分叉 | 分叉后写不污染兄弟；共享前缀只算一次 | E-02 | 2 d |
| E-04 | 三级回收：驱逐 / 换出 / 抢占 | 抢占率 = `Recompute Tokens / Total Decoded Tokens` < 5% | E-03 | 3 d |
| E-05 | block table 每步 H2D 与图重放联通 | 与 A-06 合流：树变异后重放仍读对块 | A-06, E-03 | 2 d |
| E-06 | padding 双端隔离 | dummy 行不落 KV（写侧 `slot_mapping` 截断）+ 读侧 trim；附负向用例 | E-01 | 1 d |

---

## Track F — 基础设施与治理（本仓库）

| ID | 任务 | 完成判据 | 依赖 | 估时 |
| --- | --- | --- | --- | --- |
| F-00 | ✅ 本机工具链 + host-only 构建链 | cmake 4.4.5 / ninja 1.13.2 已装；`cmake -S . -B build -G Ninja` + `ctest` 全绿；两条负向用例（缺 SOC / 无 CANN）按预期报错。见 `docs/TOOLCHAIN.md` | — | 完成 |
| F-01 | CI：host-only 必跑 | `g++ graph_probe_logic_test` + `bash -n` + `cmake -B build`（无 CANN 路径）全绿 | F-00 | 0.5 d |
| F-02 | `rust-toolchain.toml` + `rust-version` | 锁定版本；CI 用同一版本 | D-01 | 0.2 d |
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
