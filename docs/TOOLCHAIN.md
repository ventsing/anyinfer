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
| python3 | 3.14 | 系统包 | `/usr/bin/python3` |

`~/.local/bin` **默认不在 PATH 上**，每个 shell 需要：

```bash
export PATH="$HOME/.local/bin:$PATH"
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
| `pypi.tuna.tsinghua.edu.cn/simple/` | ~0.9 MB/s |
| `mirrors.tuna.tsinghua.edu.cn/rustup` | ~0.38 MB/s |
| `pypi.org` 直连 | ~23 KB/s |

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

## Rust（可选，Track D/F-02 才需要）

`static.rust-lang.org` 只有 ~46 KB/s，必须走 TUNA 镜像：

```bash
export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
export RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup
cd /tmp && curl -sSLO https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup/dist/x86_64-unknown-linux-gnu/rustup-init
chmod +x rustup-init
./rustup-init -y --no-modify-path --profile minimal --default-toolchain stable
```

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

## 本机**不能**验证的

| 项 | 原因 |
| --- | --- |
| 任何 NPU 相关内容 | 无 CANN、无 NPU、无 `npu-smi` |
| Ascend C kernel 编译 | 需 CANN 的 `ascendc.cmake` |
| `graph_probe_test.cpp` 链接 | 需 `acl/acl.h` 与 `libascendcl.so` |
| 原型仓库 `cargo test` | 需 Rust（见上文安装段） |
