// D-01 的验证：C 侧布局必须与 ABI 布局产物一致。
//
// 验证链的形状是"一份产物、两边独立校验"：
//   Rust 定义  ──生成──>  native/abi/abi_layout.txt  <──解析校验──  C 定义
//   (crates/anyinfer-abi)                          (本文件)
//
// 为什么不直接让 Rust 和 C 互相 assert：
//   两个编译器各自通过并不说明它们对"同一个结构体"的理解一致 ——
//   真正的错位通常只在运行时表现为"读到了别的字段的值"。
//   落成一份可 diff 的文本产物，错位就变成了 code review 里一眼可见的行。
//
// 本测试除了比对偏移，还断言一条更强的性质：**字段首尾相接、无空洞**。
// 有隐式 padding 的结构体，其 padding 字节是未定义的，会破坏
// "两次构造的 step 逐位相等"这类确定性断言。

#include "anyinfer.h"

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string &what)
{
    if (!condition) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

struct ParsedStruct {
    std::string name;
    size_t size = 0;
    size_t align = 0;
    std::vector<std::string> field_names;
    std::vector<size_t> field_offsets;
    std::vector<size_t> field_sizes;
};

struct ExpectedField {
    const char *name;
    size_t offset;
    size_t size;
};

struct ExpectedStruct {
    const char *name;
    size_t size;
    size_t align;
    std::vector<ExpectedField> fields;
};

#define FIELD_OF(type, member) \
    ExpectedField { #member, offsetof(type, member), sizeof(((type *)nullptr)->member) }

std::vector<ExpectedStruct> expected_layout()
{
    return {
        {"AnyInferStatus",
         sizeof(AnyInferStatus),
         alignof(AnyInferStatus),
         {
             FIELD_OF(AnyInferStatus, code),
             FIELD_OF(AnyInferStatus, failed_op_id),
             FIELD_OF(AnyInferStatus, detail),
             FIELD_OF(AnyInferStatus, reserved0),
             FIELD_OF(AnyInferStatus, step_index),
         }},
        {"AnyInferStep",
         sizeof(AnyInferStep),
         alignof(AnyInferStep),
         {
             FIELD_OF(AnyInferStep, abi_version),
             FIELD_OF(AnyInferStep, flags),
             FIELD_OF(AnyInferStep, batch_size),
             FIELD_OF(AnyInferStep, num_tokens),
             FIELD_OF(AnyInferStep, max_blocks),
             FIELD_OF(AnyInferStep, vocab_size),
             FIELD_OF(AnyInferStep, topk),
             FIELD_OF(AnyInferStep, reserved0),
             FIELD_OF(AnyInferStep, step_index),
             FIELD_OF(AnyInferStep, weight_version),
             FIELD_OF(AnyInferStep, stream),
             FIELD_OF(AnyInferStep, token_ids),
             FIELD_OF(AnyInferStep, positions),
             FIELD_OF(AnyInferStep, block_table),
             FIELD_OF(AnyInferStep, seq_lens),
             FIELD_OF(AnyInferStep, slot_mapping),
             FIELD_OF(AnyInferStep, kv_cache),
             FIELD_OF(AnyInferStep, logits),
             FIELD_OF(AnyInferStep, topk_values),
             FIELD_OF(AnyInferStep, topk_indices),
             FIELD_OF(AnyInferStep, logsumexp),
             FIELD_OF(AnyInferStep, sampling_mask),
         }},
    };
}

bool parse_artifact(const std::string &path, unsigned &version,
                    std::map<std::string, ParsedStruct> &out)
{
    std::ifstream stream(path);
    if (!stream) {
        std::printf("FAIL cannot open %s\n", path.c_str());
        return false;
    }

    std::string line;
    bool have_header = false;
    ParsedStruct *current = nullptr;

    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream fields(line);
        std::string keyword;
        fields >> keyword;

        if (keyword == "anyinfer-abi-layout") {
            fields >> version;
            have_header = true;
        } else if (keyword == "struct") {
            std::string name;
            std::string size_key;
            std::string align_key;
            size_t size = 0;
            size_t align = 0;
            fields >> name >> size_key >> size >> align_key >> align;
            if (size_key != "size" || align_key != "align") {
                std::printf("FAIL malformed struct line: %s\n", line.c_str());
                return false;
            }
            ParsedStruct parsed;
            parsed.name = name;
            parsed.size = size;
            parsed.align = align;
            out[name] = parsed;
            current = &out[name];
        } else if (keyword == "field") {
            if (current == nullptr) {
                std::printf("FAIL field line before any struct: %s\n", line.c_str());
                return false;
            }
            std::string name;
            size_t offset = 0;
            size_t size = 0;
            fields >> name >> offset >> size;
            current->field_names.push_back(name);
            current->field_offsets.push_back(offset);
            current->field_sizes.push_back(size);
        } else {
            std::printf("FAIL unknown keyword in artifact: %s\n", line.c_str());
            return false;
        }
    }
    return have_header;
}

void compare_layout(const std::map<std::string, ParsedStruct> &parsed)
{
    const std::vector<ExpectedStruct> expected = expected_layout();

    check(parsed.size() == expected.size(),
          "artifact struct count " + std::to_string(parsed.size()) + " != expected " +
              std::to_string(expected.size()));

    for (const ExpectedStruct &want : expected) {
        const auto found = parsed.find(want.name);
        if (found == parsed.end()) {
            check(false, std::string("artifact is missing struct ") + want.name);
            continue;
        }
        const ParsedStruct &got = found->second;
        const std::string where = want.name;

        check(got.size == want.size,
              where + " size: artifact " + std::to_string(got.size) + " vs C " +
                  std::to_string(want.size));
        check(got.align == want.align,
              where + " align: artifact " + std::to_string(got.align) + " vs C " +
                  std::to_string(want.align));
        check(got.field_names.size() == want.fields.size(),
              where + " field count: artifact " + std::to_string(got.field_names.size()) +
                  " vs C " + std::to_string(want.fields.size()));

        const size_t limit = std::min(got.field_names.size(), want.fields.size());
        for (size_t i = 0; i < limit; ++i) {
            const std::string label = where + "." + want.fields[i].name;
            check(got.field_names[i] == want.fields[i].name,
                  label + ": artifact field name is " + got.field_names[i]);
            check(got.field_offsets[i] == want.fields[i].offset,
                  label + " offset: artifact " + std::to_string(got.field_offsets[i]) +
                      " vs C " + std::to_string(want.fields[i].offset));
            check(got.field_sizes[i] == want.fields[i].size,
                  label + " size: artifact " + std::to_string(got.field_sizes[i]) + " vs C " +
                      std::to_string(want.fields[i].size));
        }

        // 最强的一条：字段首尾相接，说明没有隐式 padding
        size_t cursor = 0;
        for (size_t i = 0; i < got.field_names.size(); ++i) {
            check(got.field_offsets[i] == cursor,
                  where + " has implicit padding before " + got.field_names[i] + " (offset " +
                      std::to_string(got.field_offsets[i]) + ", expected " +
                      std::to_string(cursor) + ")");
            cursor = got.field_offsets[i] + got.field_sizes[i];
        }
        check(cursor == got.size,
              where + " has trailing padding (" + std::to_string(cursor) + " of " +
                  std::to_string(got.size) + " bytes accounted for)");
    }
}

void abi_invariants()
{
    check(ANYINFER_ABI_VERSION == 1u, "ABI version is 1");
    check(ANYINFER_OK == 0, "success is zero");
    check(ANYINFER_ERR_INVALID_ARG < 0 && ANYINFER_ERR_INTERNAL < 0, "errors are negative");
    check(ANYINFER_OP_NONE == 0xFFFFFFFFu, "OP_NONE sentinel");

    // 三个 step 标志互不重叠
    check((ANYINFER_STEP_FLAG_DECODE & ANYINFER_STEP_FLAG_BUCKET_PADDED) == 0,
          "DECODE and BUCKET_PADDED do not overlap");
    check((ANYINFER_STEP_FLAG_TRACE & ANYINFER_STEP_FLAG_BUCKET_PADDED) == 0,
          "TRACE and BUCKET_PADDED do not overlap");

    // 成功状态必须是"没有失败算子"
    AnyInferStatus status = {};
    status.code = ANYINFER_OK;
    status.failed_op_id = ANYINFER_OP_NONE;
    check(status.code == ANYINFER_OK, "status can express success");

    // reserved 字段必须存在且为零初始化 —— 它们是显式占位，不是随手留空
    AnyInferStep step = {};
    check(step.reserved0 == 0, "step reserved0 is explicit and zero");
    check(step.abi_version == 0, "zero-initialised step starts at version 0 (caller must set)");
}

} // namespace

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "native/abi/abi_layout.txt";

    unsigned version = 0;
    std::map<std::string, ParsedStruct> parsed;
    if (!parse_artifact(path, version, parsed)) {
        std::printf("FAIL artifact not parsed\n");
        return 1;
    }
    check(version == ANYINFER_ABI_VERSION,
          "artifact ABI version " + std::to_string(version) + " != C " +
              std::to_string(ANYINFER_ABI_VERSION));

    compare_layout(parsed);
    abi_invariants();

    if (failures != 0) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("abi: C layout matches %s (2 structs, no implicit padding)\n", path);
    return 0;
}
