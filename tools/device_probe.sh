#!/usr/bin/env bash
# 上机第 0 天的机器身份采集（在 910B/910C 设备机上跑，不写任何引擎代码）。
#
# 为什么单独做成脚本：这一天的产出不是代码，是**把设计包络从假设变成实测**。
# 手敲十条命令，一定会漏；漏掉的那条会在三天后以"性能数字对不上"的形式回来。
#
# 用法（在设备机上）：
#     ./tools/device_probe.sh                    # 人读摘要
#     ./tools/device_probe.sh --json out.json     # 同时写机读 JSON
#
# 设计原则：**任何一步失败都不中断**。反正是采集，缺哪项就报 unknown ——
# 一个因为 `npu-smi` 版本不同而整体中止的探测脚本比没有更糟。

set -u

JSON_OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --json) JSON_OUT="${2:-device-probe.json}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 64 ;;
    esac
done

# 采集结果的键值表。值里的换行会被压成一行，避免 JSON 里出现裸换行。
KEYS=()
VALS=()

record() {  # record <key> <value>
    KEYS+=("$1")
    VALS+=("$(printf '%s' "$2" | tr '\n' '|' | sed 's/|/ \\/ /g; s/  */ /g' | cut -c1-500)")
}

try() {  # try <key> <cmd...>  —— 命令不存在或失败都记成 unavailable
    local key="$1"; shift
    if ! command -v "$1" >/dev/null 2>&1; then
        record "$key" "unavailable: $1 不在 PATH 上"
        return 0
    fi
    local out
    if out=$("$@" 2>&1); then
        record "$key" "${out:-<空输出>}"
    else
        record "$key" "failed(rc=$?): $(printf '%s' "$out" | head -5)"
    fi
}

section() { printf '\n=== %s ===\n' "$1"; }
show() { printf '  %-26s %s\n' "$1" "$2"; }

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
record probe_started_utc "$started"
record hostname "$(hostname 2>/dev/null || echo unknown)"
record kernel "$(uname -r 2>/dev/null || echo unknown)"
record arch "$(uname -m 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------- 1. 卡与驱动
section "1. 卡 / 驱动 / 固件"

NPU_SMI_INFO=""
if command -v npu-smi >/dev/null 2>&1; then
    NPU_SMI_INFO=$(npu-smi info 2>&1 || true)
    record npu_smi_info "$NPU_SMI_INFO"
else
    record npu_smi_info "unavailable: npu-smi 不在 PATH 上"
fi

# SoC 型号。这是**决定 block_size 的那一项**（DESIGN §5.1：A2/V4.1 用 128，其余 32）。
# vllm-ascend 的映射表在 operators/vllm-ascend/vllm_ascend/device/hardware.py：
#   910b* -> A2 ; 910c / ascend910_93xx -> A3 ; 310p* -> 310P ; ascend950 -> A5
SOC=""
if [ -n "$NPU_SMI_INFO" ]; then
    SOC=$(printf '%s' "$NPU_SMI_INFO" | grep -oE '910[A-Za-z]|910_[0-9]{4}' | sort -u | tr '\n' ',' | sed 's/,$//')
fi
record soc_seen_in_npu_smi "${SOC:-unknown}"

try npu_smi_version npu-smi --version
try driver_version cat /usr/local/Ascend/driver/version.info
try firmware_version cat /usr/local/Ascend/firmware/version.info

section "2. HBM"
try hbm_info_npu_smi npu-smi info -t memory
# aclrtGetMemInfo 需要写代码；先用最浅的方式：从 npu-smi 文本里抓数字，抓不到就留 unknown
HBM=$(printf '%s' "$NPU_SMI_INFO" | grep -oE '[0-9]+ */ *[0-9]+' | head -1)
record hbm_numeric_seen "${HBM:-unknown}"

# ---------------------------------------------------------------- 3. CANN
section "3. CANN 工具链"

record ascend_home_path "${ASCEND_HOME_PATH:-<未设置>}"

CANN_VERSION=""
for f in "${ASCEND_HOME_PATH:-/nonexistent}/version.cfg" \
         /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
         /usr/local/Ascend/latest/version.cfg; do
    if [ -f "$f" ]; then
        CANN_VERSION=$(cat "$f" 2>/dev/null | head -20)
        record cann_version_cfg "$f"
        break
    fi
done
record cann_version "${CANN_VERSION:-unknown: 未找到 version.cfg}"
try cann_version_info cat "${ASCEND_HOME_PATH:-/nonexistent}/compiler/version.info"
try atc_version atc --version

# ---------------------------------------------------------------- 4. A-01
# 唯一的架构未知项：CANN 暴露哪一套图 API。
# 结论决定 native/graph_probe/graph_compat.h 里 ANYINFER_GRAPH_API 填 1 还是 2。
section "4. A-01：图 API 符号（决定 ANYINFER_GRAPH_API）"

ACL_INC="${ASCEND_HOME_PATH:-/nonexistent}/include/acl"
GRAPH_API_HITS=""
if [ -d "$ACL_INC" ]; then
    GRAPH_API_HITS=$(grep -nE "StreamBeginCapture|StreamEndCapture|GraphInstantiate|GraphLaunch|GraphExec|GraphGetNodes" "$ACL_INC"/*.h 2>/dev/null | head -40)
    record acl_graph_symbols "$GRAPH_API_HITS"
else
    record acl_graph_symbols "unavailable: $ACL_INC 不存在"
fi

MDLRI_HITS=""
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -d "$ASCEND_HOME_PATH/include" ]; then
    MDLRI_HITS=$(grep -rn "aclmdlRI" "$ASCEND_HOME_PATH/include/" 2>/dev/null | head -20)
    record acl_mdlri_symbols "$MDLRI_HITS"
else
    record acl_mdlri_symbols "unavailable: ASCEND_HOME_PATH 未设置"
fi

# 判定建议：aclrtGraph* 齐全 → 1；只有 aclmdlRI* → 2；两者都有 → 优先 1（更底层）
GRAPH_API_GUESS="unknown（两组符号都没抓到，先查 ASCEND_HOME_PATH）"
if [ -n "$GRAPH_API_HITS" ]; then
    GRAPH_API_GUESS="1  (aclrtGraph* 可用)"
    [ -n "$MDLRI_HITS" ] && GRAPH_API_GUESS="1  (aclrtGraph* 与 aclmdlRI* 都在，优先 1)"
elif [ -n "$MDLRI_HITS" ]; then
    GRAPH_API_GUESS="2  (只有 aclmdlRI*)"
fi
record anyinfer_graph_api_suggestion "$GRAPH_API_GUESS"

# ---------------------------------------------------------------- 5. Q2
# DataCopy 第二参数的语义（元素数 vs 字节数）。写错不会崩，只会静默拷错长度。
section "5. Q2：DataCopy 语义"

ASCENDC_INC="${ASCEND_HOME_PATH:-/nonexistent}/include/ascendc"
DATACOPY_HITS=""
for d in "$ASCENDC_INC" "${ASCEND_HOME_PATH:-/nonexistent}/include"; do
    [ -d "$d" ] || continue
    DATACOPY_HITS=$(grep -rn -A4 "void DataCopy" "$d" 2>/dev/null | head -40)
    [ -n "$DATACOPY_HITS" ] && break
done
record datacopy_signature "$DATACOPY_HITS"

# ---------------------------------------------------------------- 6. 编译期 SoC
# 910C 的编译期 SoC_NAME 是 ascend910_93xx 系列（不是 ascend910bN）。
section "6. 编译期 SoC 版本建议"

case "${SOC:-}" in
    *910_9*|*910c*|*910C*) SOC_SUGGEST="ascend910_93xx（具体四位后缀从 npu-smi 读，见 soc_seen_in_npu_smi）" ;;
    *910b*|*910B*)         SOC_SUGGEST="ascend910bN（从 npu-smi 读，例如 ascend910b1）" ;;
    *310p*|*310P*)         SOC_SUGGEST="ascend310pN —— 注意这不是训练/推理主力卡，block_size 与算子路径都不同" ;;
    *)                     SOC_SUGGEST="unknown：npu-smi 没给出可识别的型号" ;;
esac
record anyinfer_soc_suggestion "$SOC_SUGGEST"

# ---------------------------------------------------------------- 输出
section "摘要"
for i in "${!KEYS[@]}"; do
    show "${KEYS[$i]}" "$(printf '%s' "${VALS[$i]}" | cut -c1-140)"
done

printf '\n关键结论\n'
show "ANYINFER_GRAPH_API" "$GRAPH_API_GUESS"
show "ANYINFER_SOC" "$SOC_SUGGEST"
printf '  %-26s %s\n' "下一步" "把这两项填进上面的命令，再跑第 1 步（A-02）"

if [ -n "$JSON_OUT" ]; then
    {
        printf '{\n'
        printf '  "probe_started_utc": "%s",\n' "$started"
        printf '  "facts": {\n'
        n=${#KEYS[@]}
        for i in "${!KEYS[@]}"; do
            k=${KEYS[$i]}
            v=$(printf '%s' "${VALS[$i]}" | tr -d '\\' | sed 's/"/\\"/g')
            sep=","; [ "$((i + 1))" -eq "$n" ] && sep=""
            printf '    "%s": "%s"%s\n' "$k" "$v" "$sep"
        done
        printf '  }\n}\n'
    } > "$JSON_OUT"
    printf '\n机读结果已写入 %s\n' "$JSON_OUT"
fi
