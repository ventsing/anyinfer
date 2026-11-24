#include "graph_probe.h"

#include "acl/acl.h"
#include "aclrtlaunch_anyinfer_graph_probe_kernel.h"

extern "C" int32_t anyinfer_graph_probe_launch(void *stream, uint32_t blocks,
                                               uint32_t rows, uint32_t elems,
                                               uint32_t use_table, void *table,
                                               void *arena, void *out)
{
    // 参数全部前置校验：在 Ascend 上非法参数常常不崩，只静默产错数。
    if (stream == nullptr || blocks == 0 || rows == 0 || elems == 0 || elems % 8 != 0
        || arena == nullptr || out == nullptr || (use_table != 0 && table == nullptr)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACLRT_LAUNCH_KERNEL(anyinfer_graph_probe_kernel)(
        blocks, stream, static_cast<uint8_t *>(table), static_cast<uint8_t *>(arena),
        static_cast<uint8_t *>(out), rows, elems, use_table);
}
