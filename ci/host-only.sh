#!/usr/bin/env bash
# F-01：host-only CI 的单一入口。
#
# 设计意图：**没有 CANN、没有 NPU、没有 Rust 的机器上也能完整跑通**。
# 任何一项失败即整体失败；用于在每次改动后 60 秒内给出"能不能合"的答案。
#
# 用法：
#   ./ci/host-only.sh            完整检查
#   ./ci/host-only.sh --quick    跳过 cmake 构建（只做一次干净编译 + 逻辑测试）
set -euo pipefail

cd "$(dirname "$0")/.."

QUICK=0
case "${1:-}" in
--quick) QUICK=1 ;;
"") ;;
*)
    echo "usage: $0 [--quick]" >&2
    exit 2
    ;;
esac

# 本机 cmake/ninja 装在 ~/.local/bin，默认不在 PATH 上
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

BUILD_DIR="build/ci"
pass=0
fail=0

step() {
    local name="$1"
    shift
    if "$@" >/tmp/anyinfer-ci-step.log 2>&1; then
        printf '  ok   %s\n' "$name"
        pass=$((pass + 1))
    else
        printf '  FAIL %s\n' "$name"
        sed 's/^/       /' /tmp/anyinfer-ci-step.log | tail -20
        fail=$((fail + 1))
    fi
}

echo "== 1. 源码卫生 =="
step "shell 语法 (bash -n)" bash -n ci/host-only.sh
if compgen -G "*.sh" >/dev/null; then
    step "顶层脚本语法 (bash -n)" bash -c 'for f in *.sh; do bash -n "$f"; done'
fi
mkdir -p "$BUILD_DIR"

echo "== 2. C++ 编译（严格告警）=="
CXX=${CXX:-g++}
CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror)

step "graph_probe 裁决逻辑" \
    "$CXX" "${CXXFLAGS[@]}" -I native/graph_probe \
    tests/graph_probe_logic_test.cpp -o "$BUILD_DIR/graph-probe-logic-test"
step "harness exact" \
    "$CXX" "${CXXFLAGS[@]}" -I native/harness \
    tests/harness_exact_test.cpp -o "$BUILD_DIR/harness-exact-test"
step "harness compare" \
    "$CXX" "${CXXFLAGS[@]}" -I native/harness \
    tests/harness_compare_test.cpp -o "$BUILD_DIR/harness-compare-test"
step "harness bench" \
    "$CXX" "${CXXFLAGS[@]}" -I native/harness \
    tests/harness_bench_test.cpp -o "$BUILD_DIR/harness-bench-test"

echo "== 3. 运行测试 =="
step "graph_probe 裁决逻辑" "$BUILD_DIR/graph-probe-logic-test"
step "harness exact（含正/负向对照）" "$BUILD_DIR/harness-exact-test"
step "harness compare" "$BUILD_DIR/harness-compare-test"
step "harness bench（注入假时钟）" "$BUILD_DIR/harness-bench-test"

echo "== 4. Python 参考 oracle（C 轨道）=="
step "unittest (toy_reference_test)" python3 -m unittest discover -s tests -p 'toy_*_test.py'
step "export 玩具模型 + 参考轨迹" python3 tools/toy/export.py --out "$BUILD_DIR/toy"
step "权重哈希锁" python3 tools/toy/check_lock.py --manifest "$BUILD_DIR/toy/manifest.json"
# 对拍器自证：参考vs参考必须过，扰动/注入NaN必须红
step "对拍器自证有效" python3 tools/toy/selftest.py --dir "$BUILD_DIR/toy"

if [ "$QUICK" -eq 0 ]; then
    echo "== 5. cmake host-only 构建 =="
    step "configure (WITH_CANN=OFF)" cmake -S . -B "$BUILD_DIR/cmake" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release
    step "build" cmake --build "$BUILD_DIR/cmake"
    step "ctest" ctest --test-dir "$BUILD_DIR/cmake" --output-on-failure

    echo "== 6. 负向用例：CANN 守卫必须报错 =="
    # 这两个用例"失败"才是正确行为，故用 ! 取反
    step "缺 SOC 必须 FATAL" bash -c \
        '! cmake -S . -B /tmp/anyinfer-ci-nosoc -G Ninja -DANYINFER_WITH_CANN=ON >/dev/null 2>&1'
    step "无 CANN 必须 FATAL" bash -c \
        '! cmake -S . -B /tmp/anyinfer-ci-nocann -G Ninja -DANYINFER_WITH_CANN=ON \
             -DANYINFER_SOC=ascend910b1 -DASCEND_CANN_PACKAGE_PATH=/nonexistent >/dev/null 2>&1'
    rm -rf /tmp/anyinfer-ci-nosoc /tmp/anyinfer-ci-nocann
fi

echo
printf '通过 %d 项，失败 %d 项\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
echo "host-only CI：全部通过"
