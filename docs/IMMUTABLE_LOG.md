# 不可自修改面变更日志

只增不改。每次 `tools/immutable.py seal` 都会在这里留一条记录，
所以任何对判据 / 参考实现 / 语料的改动都能在 git diff 里被看见。

## 2026-11-24 16:30:53

理由：F-05 首次封印：判据、参考 oracle、语料、裁决权威、构建逻辑，以及执行者自己

- 新增 `CMakeLists.txt`：661b06954c1bd063
- 新增 `ci/host-only.sh`：3ad2a3059cc346d4
- 新增 `docs/DECISIONS.md`：cbd1c616e1a783d0
- 新增 `native/abi/abi_layout.txt`：79fd3318518b72c0
- 新增 `native/harness/accept.h`：03e5a90cb861f664
- 新增 `native/harness/bench.h`：ed0398e0fc6ade35
- 新增 `native/harness/bit_exact.h`：7b6f002712b9dbe3
- 新增 `native/harness/exact.h`：222fbea772597bc8
- 新增 `native/harness/fp64_reference.h`：ee3674fd7bb1c8c2
- 新增 `native/harness/report.h`：7b3beeb70cb53534
- 新增 `tests/abi_layout_test.cpp`：af10b64618a44166
- 新增 `tests/golden/toy_corpus.json`：8f7b051f43b1c828
- 新增 `tests/golden/toy_manifest.lock.json`：21eb6413dbb5b143
- 新增 `tests/graph_probe_logic_test.cpp`：4faff92140b6438d
- 新增 `tests/harness_accept_test.cpp`：292ffb89a44bcc01
- 新增 `tests/harness_bench_test.cpp`：ccb381f24d87a223
- 新增 `tests/harness_compare_test.cpp`：ed617ea36e63eaf2
- 新增 `tests/harness_exact_test.cpp`：846622988b4fc817
- 新增 `tests/toy_reference_test.py`：bbef6251270b504f
- 新增 `tools/accept/accept_main.cpp`：f6f8d26916b8cadf
- 新增 `tools/immutable.py`：b94c502b0a314837
- 新增 `tools/toy/check_lock.py`：91c7633875363bd2
- 新增 `tools/toy/compare.py`：6d980503f33aff9c
- 新增 `tools/toy/export.py`：856231867ae1f275
- 新增 `tools/toy/model.py`：eb83a8322b3d9000
- 新增 `tools/toy/perturb.py`：860d8c676f219d3c
- 新增 `tools/toy/safetensors_io.py`：1540c9d2013672a1
- 新增 `tools/toy/selftest.py`：b09cbf5b9161c56e

## 2026-11-24 16:30:53

理由：幂等性自检

（无变化，仅重新确认）

## 2026-11-24 16:32:16

理由：F-05：给 CI 增加不可自修改面检查与负向对照（这次封印正是检查器抓到本改动后按流程执行的）

- 修改 `ci/host-only.sh`：3ad2a3059cc346d4 → 3b059e236d2a753b

## 2026-11-24 16:33:42

理由：F-02：CI 增加工具链可见性步骤（决策见 ADR-016）

- 修改 `ci/host-only.sh`：3b059e236d2a753b → d0febbeda677d225

## 2026-11-24 16:33:56

理由：ADR-016 + TODO 记录 F-02/F-04/F-05 判决

- 修改 `docs/DECISIONS.md`：cbd1c616e1a783d0 → 711c3c12b6f631de

## 2026-11-24 16:34:43

理由：修正 CI 调用：step_optional 需要把解释器同时作为工具名与命令首词

- 修改 `ci/host-only.sh`：d0febbeda677d225 → c264edc31fb56b91

## 2026-11-24 16:47:29

理由：CI 增加 cargo fmt --check 与 tools/ 新文件说明

- 修改 `ci/host-only.sh`：c264edc31fb56b91 → 863a6e0ca7a3df3b

## 2026-11-24 16:48:24

理由：更新 F-01 的 CI 步骤数与整理记录

（无变化，仅重新确认）
