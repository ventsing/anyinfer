# 本机工具链

> 记录**实际验证过**的版本与安装方式，保证任何人能复现同一套构建环境。
> 换版本要重跑一次「验证记录」里的命令并把结果更新到本文件。

## 已验证版本

| 工具 | 版本 | 来源 | 路径 |
| --- | --- | --- | --- |
| gcc / g++ | 15.2.0 (Ubuntu 26.04) | 系统包 | `/usr/bin/g++` |
| make | GNU Make (系统) | 系统包 | `/usr/bin/make` |
| **cmake** | **4.4.5** | pip 轮子（阿里云镜像） | `~/.local/bin/cmake` |
| **ninja** | **1.13.2** | pip 轮子（阿里云镜像） | `~/.local/bin/ninja` |
| **rustc / cargo** | **1.100.0** | rustup（TUNA 镜像），profile = minimal | `~/.cargo/bin/` |
| python3 | 3.14 | 系统包 | `/usr/bin/python3` |

`~/.local/bin` 与 `~/.cargo/bin` **默认都不在 PATH 上**，每个 shell 需要：

```bash
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
```

或写进 `~/.bashrc`（本项目不自动修改用户 dotfile）。

## 本机约束（决定了安装方式）

| 约束 | 实测 | 影响 |
| --- | --- | --- |
| 非 root，`sudo` 需交互认证 | `id -u` = 1000；`sudo -n` 失败 | 不能用 apt 装包 |
| 无 `pip` / `ensurepip` | `python3 -m ensurepip` → 无此模块 | 需自举 pip |
| `github.com` 不可达 | `http=000` | 不能用 GitHub Release 的二进制包 |
| `pypi.org` 直连 | ~23 KB/s | 不可用 |
| `rsproxy.cn` 不可达 | `http=000` | Rust 走 TUNA |

**可用镜像（实测速度）**

| 源 | 速度 |
| --- | --- |
| `mirrors.aliyun.com/pypi/simple/` | ~1.6 MB/s |
| `mirrors.ustc.edu.cn/crates.io` | ~510 KB/s |
| `pypi.tuna.tsinghua.edu.cn/simple/` | ~0.9 MB/s |
| `mirrors.tuna.tsinghua.edu.cn/rustup` | ~380 KB/s |
| `static.crates.io` 直连 | ~20 KB/s |
| `pypi.org` 直连 | ~23 KB/s |

不可达：`github.com`、`rsproxy.cn`、`index.crates.io` 直连（均 `http=000` 或超时）。

## cmake / ninja 的安装方法（无 root、无 pip 时）

**Step 1：自举 pip**（wheel 本身就是 zip，可直接用 `PYTHONPATH` 加载，无需安装）

```bash
cd /tmp
curl -sSLO https://mirrors.aliyun.com/pypi/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/pip-25.3-py3-none-any.whl
PYTHONPATH=/tmp/pip-25.3-py3-none-any.whl python3 -m pip --version
# 该 wheel 的 sha256 应为
# 9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd
```

**Step 2：装到用户目录**（Ubuntu 26.04 有 PEP 668 外部管理标记，故需 `--break-system-packages`）

```bash
PYTHONPATH=/tmp/pip-25.3-py3-none-any.whl python3 -m pip install \
  --user --break-system-packages \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  cmake ninja
```

## Rust（Track D/F-02 才需要，但已装好）

`static.rust-lang.org` 只有 ~46 KB/s，必须走 TUNA 镜像：

```bash
export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
export RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup
cd /tmp && curl -sSLO https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup/dist/x86_64-unknown-linux-gnu/rustup-init
chmod +x rustup-init
./rustup-init -y --no-modify-path --profile minimal --default-toolchain stable
```

**crates.io 必须换镜像**，否则拉依赖会超时。已写入 `~/.cargo/config.toml`：

```toml
[source.crates-io]
replace-with = "ustc"

[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
```

> 注意：TUNA 的 `crates.io-index` 里 `config.json` 的 `dl` 仍指向 `static.crates.io`（~20 KB/s），
> 而 USTC 的 `dl` 指向其本地端点（~510 KB/s）。**所以选 USTC 而不是 TUNA。**

## 验证记录

本机随时可重跑，作为环境自检：

```bash
export PATH="$HOME/.local/bin:$PATH"
cd dev_anyinfer

# 1) 版本
cmake --version | head -1     # cmake version 4.4.5
ninja --version               # 1.13.2

# 2) host-only 构建（无需 CANN / NPU）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3) 跑裁决逻辑测试
ctest --test-dir build --output-on-failure
#  1/1 Test #1: graph-probe-logic .............. Passed
#  100% tests passed, 0 tests failed out of 1

# 4) 负向：缺 SOC_VERSION 应报错
cmake -S . -B /tmp/b1 -G Ninja -DANYINFER_WITH_CANN=ON 2>&1 | grep -F "requires -DANYINFER_SOC"

# 5) 负向：无 CANN 应报错并给出修复提示
cmake -S . -B /tmp/b2 -G Ninja -DANYINFER_WITH_CANN=ON -DANYINFER_SOC=ascend910b1 2>&1 | grep -F "CANN not found"
```

**2026-11-24 实测结果**：1)–5) 全部符合预期。

### Rust 验证记录

原型仓库 `../anyinfer` 的测试套件在本轮之前**从未被执行过**（缺 Rust 工具链）。现已跑通：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd ../anyinfer && cargo test --locked --release
```

```
lib                   15 passed   （scheduler 并发 7 / stream 3 / model 4 / oracle 1）
src/main.rs            8 passed   （head_runner / opbench / scheduler_bench / host_observation）
host_decode_allocations 1 passed   （稳态 decode 不调用 Rust 分配器）
────────────────────────────────
合计                  24 passed, 0 failed
```

工作树保持干净（`target/` 已在原型的 `.gitignore` 中）。

## 本机**不能**验证的

| 项 | 原因 |
| --- | --- |
| 任何 NPU 相关内容 | 无 CANN、无 NPU、无 `npu-smi` |
| Ascend C kernel 编译 | 需 CANN 的 `ascendc.cmake` |
| `graph_probe_test.cpp` 链接 | 需 `acl/acl.h` 与 `libascendcl.so` |
| 原型仓库的 C++ / 设备侧测试 | 同上（其 Rust 测试已可跑，见上） |

---

## Rust 版本钉定：为什么**没有** `rust-toolchain.toml`（F-02）

原本 F-02 的判据是"加 `rust-toolchain.toml` 锁定版本"。实测后否决了它。

### 实测到的两个问题

1. **rustup 把版本号当作独立槽位**。本机 stable 恰好就是 1.100.0，但写上
   `channel = "1.100.0"` 之后 rustup 仍要重新下载一整套工具链（约 200 MB）。
   本机到 static.rust-lang.org 的实测速率 **~6 KB/s**（10 分钟拉了 3.7 MB），
   拉完要几小时；而**只要该文件存在，每次 cargo 调用都会先同步** ——
   在拉完之前整个仓库不可构建。一个让仓库不可构建的"改进"不是改进。
2. **它给出的是虚假的确信**。`rust-toolchain.toml` 保证"用了哪个编译器"，
   不保证"结果可复算"。真正需要的是**可归因**：一份测量结果必须能说明
   它是哪个编译器产的。锁版本只做到了前者的弱化版。

### 采用的替代

| | 做法 |
| --- | --- |
| 可见性 | `tools/check_toolchain.py` 打印 `rustc` / `recorded_with` / `minimum` / `target`，CI 里只报告不阻断 |
| 强制 | `--strict` 在版本与测量基线不符时 exit 1，留给**测量**路径调用 |
| 机读 | `--json` 供测量报告吸收版本戳 |
| 最低版本 | `Cargo.toml` 的 `rust-version = "1.85"`（edition 2024 的硬要求），由工具检查 |

### 这件事的诚实边界

**离线环境下没法把版本钉定做实。** 能做的是让不匹配可见，
并在测量路径上拒绝产出不可比较的数字。等真有两台机器要比位级结果时，
应该做的是让每份测量 JSON 带上编译器戳，而不是指望两边编译器相同。
