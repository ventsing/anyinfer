cmake_minimum_required(VERSION 3.16)
project(anyinfer_dev LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CTest)

# 关键设计：**没有 CANN 也能构建和测试**。
# 原型 ../anyinfer 在缺 SOC_VERSION 时直接 FATAL_ERROR，导致纯 host 的 CI 跑不起来。
option(ANYINFER_WITH_CANN "Build device-side targets (requires CANN toolkit)" OFF)
set(ANYINFER_SOC "" CACHE STRING "SoC version, e.g. ascend910b1 (required when WITH_CANN)")

# ---------------------------------------------------------------- host-only
# 裁决逻辑不依赖任何设备，必须在任何机器上可构建、可测试。
add_library(anyinfer_graph_probe_verdict INTERFACE)
target_include_directories(anyinfer_graph_probe_verdict INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/native/graph_probe")

if(BUILD_TESTING)
  add_executable(anyinfer-graph-probe-logic-test tests/graph_probe_logic_test.cpp)
  target_link_libraries(anyinfer-graph-probe-logic-test PRIVATE anyinfer_graph_probe_verdict)
  add_test(NAME graph-probe-logic COMMAND anyinfer-graph-probe-logic-test)
endif()

# ---------------------------------------------------------------- 算子验收 harness
# B-01/B-02/B-03：精确输入判据 + L1 位级比对 + L2 FP64 参考比对。
# 全部是纯 host 头文件，不含任何设备依赖。
add_library(anyinfer_harness INTERFACE)
target_include_directories(anyinfer_harness INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/native/harness")

if(BUILD_TESTING)
  add_executable(anyinfer-harness-exact-test tests/harness_exact_test.cpp)
  target_link_libraries(anyinfer-harness-exact-test PRIVATE anyinfer_harness)
  add_test(NAME harness-exact COMMAND anyinfer-harness-exact-test)

  add_executable(anyinfer-harness-compare-test tests/harness_compare_test.cpp)
  target_link_libraries(anyinfer-harness-compare-test PRIVATE anyinfer_harness)
  add_test(NAME harness-compare COMMAND anyinfer-harness-compare-test)
endif()

# ---------------------------------------------------------------- device
if(NOT ANYINFER_WITH_CANN)
  message(STATUS "ANYINFER_WITH_CANN=OFF: device targets skipped (host-only build)")
  return()
endif()

if(NOT ANYINFER_SOC)
  message(FATAL_ERROR "ANYINFER_WITH_CANN=ON requires -DANYINFER_SOC=ascend910bN")
endif()

# CANN 包路径：优先环境变量，其次常见安装位置。
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()

find_library(ASCENDCL ascendcl PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64" NO_DEFAULT_PATH)
find_path(ACL_INCLUDE_DIR acl/acl.h PATHS "${ASCEND_CANN_PACKAGE_PATH}/include" NO_DEFAULT_PATH)
find_path(ASCENDC_INCLUDE_DIR ascendc.cmake
  PATHS
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/ascendc_kernel_cmake"
)
if(NOT ASCENDCL OR NOT ACL_INCLUDE_DIR OR NOT ASCENDC_INCLUDE_DIR)
  message(FATAL_ERROR
    "CANN not found under ${ASCEND_CANN_PACKAGE_PATH}. "
    "Set ASCEND_HOME_PATH or -DASCEND_CANN_PACKAGE_PATH=<path>.")
endif()

include("${ASCENDC_INCLUDE_DIR}/ascendc.cmake")
include_directories("${ACL_INCLUDE_DIR}")

# 探针 kernel：与 ../anyinfer 的 hc_post 同构（ascendc_library + 纯 POD 启动器）。
ascendc_library(anyinfer_graph_probe_kernel SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/native/graph_probe/graph_probe_kernel.cpp")

add_library(anyinfer_graph_probe SHARED
  native/graph_probe/graph_probe_launch.cpp)
target_link_libraries(anyinfer_graph_probe PRIVATE anyinfer_graph_probe_kernel "${ASCENDCL}")
target_include_directories(anyinfer_graph_probe PRIVATE
  "${ACL_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/native/graph_probe")
set_target_properties(anyinfer_graph_probe PROPERTIES INSTALL_RPATH "$ORIGIN")

add_executable(anyinfer-graph-probe-test native/graph_probe/graph_probe_test.cpp)
target_link_libraries(anyinfer-graph-probe-test PRIVATE anyinfer_graph_probe "${ASCENDCL}")
target_include_directories(anyinfer-graph-probe-test PRIVATE
  "${ACL_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/native/graph_probe")
set_target_properties(anyinfer-graph-probe-test PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

if(BUILD_TESTING)
  add_test(NAME graph-probe COMMAND anyinfer-graph-probe-test
           --output "${CMAKE_CURRENT_BINARY_DIR}/graph-probe.json")
  # 无设备 / 无驱动时由 CTest 跳过（探针本身以 acl_error 退出 1，故仅对"无设备"约定 77；
  # 如需严格区分，请在 CI 里先跑 npu-smi info 探测）。
  set_tests_properties(graph-probe PROPERTIES SKIP_RETURN_CODE 77)
endif()

install(TARGETS anyinfer_graph_probe_kernel anyinfer_graph_probe anyinfer-graph-probe-test
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(FILES native/graph_probe/graph_probe.h DESTINATION include)
