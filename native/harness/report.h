#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

// 所有验收门的共用报告类型。
// 为什么单独抽出来：L1（位级）、L2（FP64 参考）、L3（路由一致性）用的是同一套
// Gate 语义与 JSON 形状，RSI 的自动裁决脚本只应解析一种格式。

namespace anyinfer::harness {

enum class Gate {
    Pass,
    LengthMismatch,    // 缓冲区/长度不匹配（调用方 bug，不是算子 bug）
    NonFinite,         // 期望有限值而实际是 Inf/NaN —— 通常是 kernel 写飞了
    BitMismatch,       // L1：位模式不同
    ToleranceExceeded, // L2：超出 atol + rtol*|ref|
    InvalidReference,  // 参考值自身非法（非有限）—— 参考实现出问题，不是被测件
};

inline const char *gate_name(Gate gate)
{
    switch (gate) {
    case Gate::Pass:
        return "pass";
    case Gate::LengthMismatch:
        return "length_mismatch";
    case Gate::NonFinite:
        return "non_finite";
    case Gate::BitMismatch:
        return "bit_mismatch";
    case Gate::ToleranceExceeded:
        return "tolerance_exceeded";
    case Gate::InvalidReference:
        return "invalid_reference";
    }
    return "unknown";
}

inline bool gate_pass(Gate gate) { return gate == Gate::Pass; }

/* 极简 JSON 片段拼装。不引入第三方依赖，也不追求通用性：
 * key 必须是简单标识符（不含引号/反斜杠），值是字符串/数值/整数。 */
inline void json_kv(std::string &out, const char *key, const std::string &value)
{
    out += '"';
    out += key;
    out += "\":\"";
    out += value;
    out += "\",";
}

inline void json_kv(std::string &out, const char *key, const char *value)
{
    json_kv(out, key, std::string(value));
}

inline void json_kv(std::string &out, const char *key, double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    out += '"';
    out += key;
    out += "\":";
    out += buffer;
    out += ',';
}

inline void json_kv(std::string &out, const char *key, uint64_t value)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%llu",
                  static_cast<unsigned long long>(value));
    out += '"';
    out += key;
    out += "\":";
    out += buffer;
    out += ',';
}

// 去掉尾逗号、补上开头花括号并闭合对象。
inline std::string json_close(std::string body)
{
    if (!body.empty() && body.back() == ',') {
        body.pop_back();
    }
    return "{" + body + "}";
}

} // namespace anyinfer::harness
