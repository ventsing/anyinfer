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
step "harness accept" \
    "$CXX" "${CXXFLAGS[@]}" -I native/harness \
    tests/harness_accept_test.cpp -o "$BUILD_DIR/harness-accept-test"

echo "== 3. 运行测试 =="
step "graph_probe 裁决逻辑" "$BUILD_DIR/graph-probe-logic-test"
step "harness exact（含正/负向对照）" "$BUILD_DIR/harness-exact-test"
step "harness compare" "$BUILD_DIR/harness-compare-test"
step "harness bench（注入假时钟）" "$BUILD_DIR/harness-bench-test"
step "harness accept（NotRun 必须阻断晋升）" "$BUILD_DIR/harness-accept-test"

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

    echo "== 6. 一次算子验收端到端（B-04b）=="
    ACCEPT="$BUILD_DIR/cmake/anyinfer-accept"
    TOY="$BUILD_DIR/toy"
    REF="$TOY/reference_simple.safetensors"

    step "裁决器已构建" test -x "$ACCEPT"
    step "产出 L4 报告" python3 tools/toy/compare.py \
        --reference "$REF" --candidate "$REF" --json "$TOY/l4.json" --quiet

    # 缺证据必须退 2（incomplete），不能退 0 也不能退 1
    step "缺证据 -> incomplete(2)" \
        bash -c 'python3 tools/accept.py --op hc_post --l4-report "$1" --quiet; [ $? -eq 2 ]' \
        _ "$TOY/l4.json"

    # 合成三段报告（真实生产者需要设备，这里只验裁决通路）
    step "合成三段报告" python3 -c "
import json, pathlib
d = pathlib.Path('$TOY')
d.joinpath('correctness.json').write_text(json.dumps({'gate': 'pass'}))
d.joinpath('bench.json').write_text(json.dumps(
    {'verdict': 'b_faster', 'speedup': 1.08, 'sign_test_p': 0.015625}))
d.joinpath('bench_guard.json').write_text(json.dumps(
    {'verdict': 'indistinguishable', 'speedup': 0.70, 'sign_test_p': 0.5}))
"

    # 拆成两个平铺的 step：嵌套引号在这里只会制造不可维护的转义
    step "三段齐备 -> pass(0)" bash -c '
        python3 tools/accept.py --op hc_post --correctness-report "$1" \
            --bench-report "$2" --l4-report "$3" --out "$4" --quiet
    ' _ "$TOY/correctness.json" "$TOY/bench.json" "$TOY/l4.json" "$TOY/accept.json"

    step "验收 JSON 形状正确" python3 -c '
import json, sys
report = json.load(open(sys.argv[1]))
required = {"correctness_gate", "microbench", "l4_gate", "verdict", "promotable", "blocking"}
missing = required - set(report)
if missing:
    print("missing keys: " + str(sorted(missing)))
    sys.exit(1)
if report["verdict"] != "pass" or report["promotable"] != "true":
    print("unexpected verdict: " + str(report["verdict"]))
    sys.exit(1)
' "$TOY/accept.json"

    # 未显著但 30% 回归：守护线必须拦截
    step "大幅回归未显著 -> fail(1)" bash -c '
        python3 tools/accept.py --op hc_post --correctness-report "$1" \
            --bench-report "$2" --l4-report "$3" --quiet
        [ $? -eq 1 ]
    ' _ "$TOY/correctness.json" "$TOY/bench_guard.json" "$TOY/l4.json"

    # 参数打错必须是 64，不能是 2（否则会被读成"证据不全"）
    step "参数打错 -> 用法错误(64)" bash -c '
        python3 tools/accept.py --op x --not-an-option 2>/dev/null
        [ $? -eq 64 ]
    '

    echo "== 7. 负向用例：CANN 守卫必须报错 =="
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
