// B-04b：算子验收裁决 CLI —— 全流程中**唯一**的裁决权威。
//
// 定位：它不测量任何东西，只把三段门的结论聚合成一个可审计的判决，
// 并给出退出码。设备驱动可以直接 include accept.h 内联调用；
// 跨语言的入口（Python 的 tools/accept.py）则调用这个可执行文件，
// 从而保证两条路径不可能得出不同结论。
//
// 退出码：0 可晋升 / 1 失败 / 2 证据不全（三段门没有全部跑过）

#include "accept.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

void usage(const char *program)
{
    std::fprintf(
        stderr,
        "usage: %s --op NAME\n"
        "          --correctness <pass|fail|not_run> | --correctness-gate <gate>\n"
        "          --microbench <b_faster|a_faster|indistinguishable|not_run>\n"
        "          --l4 <pass|fail|not_run> | --l4-gate <gate>\n"
        "          [--correctness-reason TEXT] [--l4-reason TEXT]\n"
        "          [--bench-speedup F] [--bench-sign-test-p F] [--max-regression F]\n"
        "          [--json PATH] [--quiet]\n"
        "\n"
        "gate names: pass length_mismatch non_finite bit_mismatch "
        "tolerance_exceeded invalid_reference\n",
        program);
}

bool parse_report_gate(const std::string &text, anyinfer::harness::Gate &out)
{
    using anyinfer::harness::Gate;
    if (text == "pass") {
        out = Gate::Pass;
    } else if (text == "length_mismatch") {
        out = Gate::LengthMismatch;
    } else if (text == "non_finite") {
        out = Gate::NonFinite;
    } else if (text == "bit_mismatch") {
        out = Gate::BitMismatch;
    } else if (text == "tolerance_exceeded") {
        out = Gate::ToleranceExceeded;
    } else if (text == "invalid_reference") {
        out = Gate::InvalidReference;
    } else {
        return false;
    }
    return true;
}

bool parse_double(const char *text, double &out)
{
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    out = value;
    return true;
}

int fail_missing(const char *what)
{
    std::fprintf(stderr, "error: --%s is required\n", what);
    return 64; // EX_USAGE
}

} // namespace

int main(int argc, char **argv)
{
    using namespace anyinfer;
    using namespace anyinfer::accept;

    AcceptInput input;
    bool have_correctness = false;
    bool have_microbench = false;
    bool have_l4 = false;
    std::string json_path;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--op") {
            const char *value = next("--op");
            if (value == nullptr) {
                return 64;
            }
            input.op = value;
        } else if (arg == "--correctness" || arg == "--correctness-gate") {
            const char *value = next(arg.c_str());
            if (value == nullptr) {
                return 64;
            }
            if (arg == "--correctness") {
                GateState state;
                if (!parse_gate_state(value, state)) {
                    std::fprintf(stderr, "error: bad --correctness '%s'\n", value);
                    return 64;
                }
                input.correctness.state = state;
            } else {
                harness::Gate gate;
                if (!parse_report_gate(value, gate)) {
                    std::fprintf(stderr, "error: bad --correctness-gate '%s'\n", value);
                    return 64;
                }
                input.correctness = from_gate(gate, "correctness");
            }
            have_correctness = true;
        } else if (arg == "--correctness-reason") {
            const char *value = next("--correctness-reason");
            if (value == nullptr) {
                return 64;
            }
            input.correctness.reason = value;
        } else if (arg == "--microbench") {
            const char *value = next("--microbench");
            if (value == nullptr) {
                return 64;
            }
            input.bench_verdict = value;
            have_microbench = true;
        } else if (arg == "--bench-speedup") {
            const char *value = next("--bench-speedup");
            if (value == nullptr || !parse_double(value, input.bench_speedup)) {
                std::fprintf(stderr, "error: bad --bench-speedup\n");
                return 64;
            }
        } else if (arg == "--bench-sign-test-p") {
            const char *value = next("--bench-sign-test-p");
            if (value == nullptr || !parse_double(value, input.bench_sign_test_p)) {
                std::fprintf(stderr, "error: bad --bench-sign-test-p\n");
                return 64;
            }
        } else if (arg == "--max-regression") {
            const char *value = next("--max-regression");
            if (value == nullptr || !parse_double(value, input.bench_policy.max_regression)) {
                std::fprintf(stderr, "error: bad --max-regression\n");
                return 64;
            }
        } else if (arg == "--l4" || arg == "--l4-gate") {
            const char *value = next(arg.c_str());
            if (value == nullptr) {
                return 64;
            }
            if (arg == "--l4") {
                GateState state;
                if (!parse_gate_state(value, state)) {
                    std::fprintf(stderr, "error: bad --l4 '%s'\n", value);
                    return 64;
                }
                input.l4.state = state;
            } else {
                harness::Gate gate;
                if (!parse_report_gate(value, gate)) {
                    std::fprintf(stderr, "error: bad --l4-gate '%s'\n", value);
                    return 64;
                }
                input.l4 = from_gate(gate, "l4");
            }
            have_l4 = true;
        } else if (arg == "--l4-reason") {
            const char *value = next("--l4-reason");
            if (value == nullptr) {
                return 64;
            }
            input.l4.reason = value;
        } else if (arg == "--json") {
            const char *value = next("--json");
            if (value == nullptr) {
                return 64;
            }
            json_path = value;
        } else if (arg == "--quiet") {
            quiet = true;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            usage(argv[0]);
            return 64;
        }
    }

    // 三段门都必须显式给出：默认值一律是 not_run，但不能靠"忘了传"来表达
    if (!have_correctness) {
        return fail_missing("correctness");
    }
    if (!have_microbench) {
        return fail_missing("microbench");
    }
    if (!have_l4) {
        return fail_missing("l4");
    }

    if (have_microbench) {
        input.microbench = from_bench(input.bench_verdict, input.bench_speedup,
                                      input.bench_sign_test_p, input.bench_policy);
    }

    const AcceptanceReport report = evaluate(input);
    const std::string json = acceptance_json(input, report);

    if (!json_path.empty()) {
        const std::string tmp = json_path + ".part";
        std::FILE *handle = std::fopen(tmp.c_str(), "wb");
        if (handle == nullptr) {
            std::fprintf(stderr, "error: cannot write %s\n", tmp.c_str());
            return 74; // EX_IOERR
        }
        std::fwrite(json.data(), 1, json.size(), handle);
        std::fputc('\n', handle);
        std::fclose(handle);
        if (std::rename(tmp.c_str(), json_path.c_str()) != 0) {
            std::fprintf(stderr, "error: cannot rename %s\n", tmp.c_str());
            return 74;
        }
    }

    if (!quiet) {
        std::printf("%s\n", json.c_str());
    }

    return exit_code(report);
}
