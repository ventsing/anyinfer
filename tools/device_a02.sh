#!/usr/bin/env bash
# A-02：设备侧编译并跑通 Arm 0 —— 也就是 README「上机第一天」的第 2 步。
#
# 为什么要有这个脚本：README 里那三条命令（cmake / build / ctest）看不出**先后依赖**
# 和**失败归属**。上机时间稀缺，出错时最贵的不是修，是不知道该修哪儿。所以这里把
# 前置检查、编译、运行、判决解读串成一条，每一步失败都给出"这意味着什么"。
#
# 用法（在设备机上，仓库根目录）：
#     ./tools/device_a02.sh                     # 自动探测 SoC，全流程
#     ./tools/device_a02.sh --soc ascend910_9391 --graph-api 1
#     ./tools/device_a02.sh --jobs 32
#     ./tools/device_a02.sh --dry-run           # 只打印将要执行的命令（无卡机器也能跑）
#     ./tools/device_a02.sh --interpret graph-probe.json   # 只解读已有结果
#
# 退出码：0 = Arm 0 通过且 verdict 已产出；1 = 前置检查失败；2 = 编译失败；
#         3 = 探针运行失败；4 = verdict 是失败态（这是**有效结果**，不是脚本出错）

set -u

SOC=""
GRAPH_API=""
JOBS="$( (nproc 2>/dev/null || echo 8) )"
OUT="graph-probe.json"
DRY_RUN=0
INTERPRET=""
BUILD_DIR="build"

while [ $# -gt 0 ]; do
    case "$1" in
        --soc)        SOC="${2:-}"; shift 2 ;;
        --graph-api)  GRAPH_API="${2:-}"; shift 2 ;;
        --jobs|-j)    JOBS="${2:-}"; shift 2 ;;
        --out)        OUT="${2:-}"; shift 2 ;;
        --build-dir)  BUILD_DIR="${2:-}"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --interpret)  INTERPRET="${2:-}"; shift 2 ;;
        -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $1（--help 看用法）" >&2; exit 64 ;;
    esac
done

section() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
ok()      { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()     { printf '  \033[31m✗\033[0m %s\n' "$1"; }
warn()    { printf '  \033[33m!\033[0m %s\n' "$1"; }
note()    { printf '    %s\n' "$1"; }

# ---------------------------------------------------------------- verdict 解读
# 单独成函数，因为它是这个脚本里唯一能在无卡机器上被真正测到的部分（--interpret）。
interpret() {
    local file="$1"
    if [ ! -f "$file" ]; then
        bad "找不到 $file"
        return 1
    fi

    local verdict remedy
    if command -v python3 >/dev/null 2>&1; then
        verdict=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('verdict',''))" "$file" 2>/dev/null)
        remedy=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('remedy',''))" "$file" 2>/dev/null)
    else
        verdict=$(grep -oE '"verdict"[[:space:]]*:[[:space:]]*"[^"]*"' "$file" | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
        remedy=$(grep -oE '"remedy"[[:space:]]*:[[:space:]]*"[^"]*"' "$file" | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
    fi

    if [ -z "$verdict" ]; then
        bad "$file 里没有 verdict 字段 —— 探针可能中途崩了，看它的 stderr"
        return 3
    fi

    printf '  verdict = %s\n' "$verdict"
    [ -n "$remedy" ] && printf '  remedy  = %s\n' "$remedy"
    echo

    case "$verdict" in
        indirection_supported)
            ok "Arm 0 通过，且图内间接寻址成立。**这是好消息。**"
            note "下一步：A-03 跑完整四臂确认，然后按 DESIGN §5 推进 Phase 2。"
            return 0 ;;
        kernel_broken)
            bad "Arm 0 就没起来 —— **这不是架构结论，是 kernel 或编译问题。**"
            note "先怀疑：DataCopy 的 count 语义（元素 vs 字节，见 --interpret 前的 Q2 采集）、"
            note "        kernel 编译产物与 SoC 不匹配、aclrtMalloc 的地址对齐。"
            note "修好之前不要往下走：拿一个坏 kernel 当证据会得出错误结论。"
            return 4 ;;
        empty_graph)
            bad "图捕获到了 0 个节点。"
            note "这**通常不是**「这张图不可捕获」，而是捕获区间没包住 kernel 下发。"
            note "先查：capture 是否在正确的 stream 上、kernel 下发是否被 host 同步提前返回。"
            note "确认无效后再转 A-04。"
            return 4 ;;
        capture_path_broken)
            bad "捕获路径本身断了。"
            note "查：stream 顺序、捕获区间里是否混入了同步调用（同步会终止捕获）。"
            return 4 ;;
        content_folded)
            bad "**真折叠** —— 内容在捕获期被烘进图里，改 buffer 无效。"
            note "这就是那个会推翻主路径的结论。**不要改代码去绕**，改架构："
            note "转后备梯队 A'（显式 graph_task_update）/ B / C，并在 DECISIONS.md 记 ADR 推翻 ADR-005。"
            return 4 ;;
        replay_not_executing)
            bad "replay 没有真正执行。"
            note "查：graph instantiate / launch 的返回值与执行流顺序。"
            return 4 ;;
        first_replay_stale)
            bad "首次 replay 读到旧值 —— block table 的 H2D 与 replay 顺序有问题。"
            note "查：H2D 是否在 replay 之前完成（同一 stream 上的序）。"
            note "注意这条与 CANN 侧已知的 topk→gather 首次 replay 越界缺陷可能同源，"
            note "相关案例：Ascend/pytorch issue #4629。"
            return 4 ;;
        *)
            bad "未知 verdict: $verdict —— 探针与这个脚本版本不匹配？"
            return 3 ;;
    esac
}

if [ -n "$INTERPRET" ]; then
    section "解读已有结果"
    interpret "$INTERPRET"
    exit $?
fi

# ---------------------------------------------------------------- 1. 前置检查
section "1. 前置检查"

FATAL=0

if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -d "${ASCEND_HOME_PATH}" ]; then
    ok "ASCEND_HOME_PATH = $ASCEND_HOME_PATH"
else
    bad "ASCEND_HOME_PATH 未设置或不存在"
    note "source 一下环境：source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    FATAL=1
fi

if command -v npu-smi >/dev/null 2>&1; then
    ok "npu-smi 可用"
else
    bad "npu-smi 不在 PATH 上"
    note "没有它就无法自动探测 SoC；可以用 --soc 手工指定"
fi

# SoC 自动探测：优先 npu-smi，其次 ASCEND_HOME_PATH 下的 version.info
if [ -z "$SOC" ]; then
    if command -v npu-smi >/dev/null 2>&1; then
        RAW=$(npu-smi info 2>/dev/null | head -20)
        if printf '%s' "$RAW" | grep -qE '910_[0-9]{4}'; then
            SOC=$(printf '%s' "$RAW" | grep -oE '910_[0-9]{4}' | head -1 | sed 's/^/ascend/')
        elif printf '%s' "$RAW" | grep -qiE '910c'; then
            bad "npu-smi 只报出 910C 而没有四位后缀"
            note "编译期 SoC 名需要具体型号（ascend910_9391 之类），不是 '910C'。"
            note "从设备信息或驱动版本信息里找；找不到就用 --soc 指定。"
            FATAL=1
        elif printf '%s' "$RAW" | grep -qE '910[bB][0-9]?'; then
            SOC=$(printf '%s' "$RAW" | grep -oE '910[bB][0-9]?[a-z]?' | head -1 | tr 'A-Z' 'a-z' | sed 's/^/ascend/')
        fi
    fi
fi

case "$SOC" in
    *910_9*) ok "SoC = $SOC（910C / A3 家族）" ;;
    *910b*)  ok "SoC = $SOC（910B / A2 家族）" ;;
    "")      bad "无法确定 SoC —— 这是 A-02 的必要参数"; note "用 --soc 指定，或先跑 ./tools/device_probe.sh"; FATAL=1 ;;
    *)       ok "SoC = $SOC（未识别的家族，继续，但 block_size 需要另行确认）" ;;
esac

# 图 API：这里最容易自欺。
#
# graph_compat.h 里写的是 `#ifndef ANYINFER_GRAPH_API` + 默认 1，所以**它永远有值**。
# 也就是说「取到了值」根本不能证明 A-01 做过 —— 不查这一点，脚本会对一个纯猜测
# 打出绿色的 ✓，而人看到 ✓ 就会跳过 A-01。这是最典型的假绿，必须区分：
#   默认值还在 + 用户没显式指定  → 警告（不是通过）
#   用户显式 --graph-api N       → 通过，且真的传进编译
#   头文件里的默认分支已被改掉    → 通过（有人手工定了）
COMPAT="native/graph_probe/graph_compat.h"
CMAKE_EXTRA=()
if [ -f "$COMPAT" ]; then
    DEFAULT_API=$(grep -oE 'ANYINFER_GRAPH_API[[:space:]]+[0-9]' "$COMPAT" | awk '{print $2}' | head -1)
    HAS_GUARD=0
    grep -qE '#ifndef[[:space:]]+ANYINFER_GRAPH_API' "$COMPAT" && HAS_GUARD=1

    if [ -n "$GRAPH_API" ]; then
        case "$GRAPH_API" in
            1) ok "ANYINFER_GRAPH_API = 1（由 --graph-api 显式指定）" ;;
            2) ok "ANYINFER_GRAPH_API = 2（由 --graph-api 显式指定）" ;;
            *) bad "--graph-api 只能是 1 或 2，收到 '$GRAPH_API'"; FATAL=1 ;;
        esac
        CMAKE_EXTRA+=("-DCMAKE_CXX_FLAGS=-DANYINFER_GRAPH_API=$GRAPH_API")
    elif [ "$HAS_GUARD" -eq 1 ] && [ -n "$DEFAULT_API" ]; then
        warn "ANYINFER_GRAPH_API 仍是头文件默认值 $DEFAULT_API —— **这不是 A-01 的结论**"
        note "任何 SoC 上它都能编译通过，所以它不会报错，只会悄悄用错的那套 API。"
        note "确认过就显式传：--graph-api 1（aclrtGraph*）或 --graph-api 2（aclmdlRI*）"
        note "还没有结论？先跑 ./tools/device_probe.sh"
        note "（本脚本按 --graph-api 传入 -DCMAKE_CXX_FLAGS；正式选项化记在 TODO F-06）"
    elif [ -n "$DEFAULT_API" ]; then
        ok "ANYINFER_GRAPH_API = $DEFAULT_API（头文件里的默认分支已改过，视为已定）"
    else
        bad "在 $COMPAT 里找不到 ANYINFER_GRAPH_API"
        FATAL=1
    fi
fi

if [ "$FATAL" -ne 0 ]; then
    section "前置检查未通过"
    bad "不要往下走。上面每条都写了怎么补。"
    exit 1
fi

# ---------------------------------------------------------------- 2. 编译
section "2. 配置与编译"

CMAKE_ARGS=(-B "$BUILD_DIR" -DANYINFER_WITH_CANN=ON "-DANYINFER_SOC=$SOC" "${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"}")
printf '  cmake %s\n' "${CMAKE_ARGS[*]}"
printf '  cmake --build %s -j %s\n' "$BUILD_DIR" "$JOBS"

if [ "$DRY_RUN" -eq 1 ]; then
    note "（--dry-run：不实际执行）"
    printf '\n  随后会运行：\n    ./%s/bin/anyinfer-graph-probe-test --output %s\n' "$BUILD_DIR" "$OUT"
    exit 0
fi

if ! cmake "${CMAKE_ARGS[@]}"; then
    section "cmake 配置失败"
    bad "编译配置没过。常见原因："
    note "ASCEND_HOME_PATH 指向的 toolkit 与 SoC 不匹配、缺 ascendcl 库、CANN 版本太旧"
    exit 2
fi

if ! cmake --build "$BUILD_DIR" -j "$JOBS"; then
    section "编译失败"
    bad "编译没过。这时还没有任何架构结论 —— 先修编译。"
    exit 2
fi
ok "编译完成"

# ---------------------------------------------------------------- 3. 运行
section "3. 运行探针（四臂）"

PROBE="./$BUILD_DIR/bin/anyinfer-graph-probe-test"
if [ ! -x "$PROBE" ]; then
    bad "找不到 $PROBE"
    note "确认 CMakeLists 里 WITH_CANN 下确实生成了这个目标"
    exit 2
fi

printf '  %s --output %s\n\n' "$PROBE" "$OUT"

# Arm 0 在设备上要跑 kernel，给足时间；但不要无限等——挂住时要能拿到现场。
if command -v timeout >/dev/null 2>&1; then
    timeout 600 "$PROBE" --output "$OUT"
    RC=$?
    if [ "$RC" -eq 124 ]; then
        section "探针超时（600 s）"
        bad "Arm 0 挂住了。这不是 verdict 失败，是运行期死锁或设备异常。"
        note "查：aclrtSynchronizeStream 没有返回、kernel 死循环、设备被别的进程占着（npu-smi info）"
        exit 3
    fi
else
    "$PROBE" --output "$OUT"; RC=$?
fi

if [ "$RC" -ne 0 ]; then
    section "探针以非零退出"
    bad "exit=$RC。verdict 可能没落盘。"
    note "如果是在 Arm 0 之前就退的，A-02 的判据是「Arm 0 退出前不报 kernel_broken」，"
    note "所以先看它是崩在 arm0 里还是 arm0 之前。"
    [ -f "$OUT" ] || exit 3
fi

# ---------------------------------------------------------------- 4. 解读
section "4. 判决"
interpret "$OUT"
exit $?
