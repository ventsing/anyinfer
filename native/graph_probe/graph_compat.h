#pragma once

/*
 * CANN 的图捕获 API 名称随版本变化。**这是本里程碑唯一不能由代码确定的部分。**
 *
 * 先在编译机上确认（10 分钟）：
 *   grep -nE "StreamBeginCapture|StreamEndCapture|GraphInstantiate|GraphLaunch|GraphExec|GraphGetNodes" \
 *     "$ASCEND_HOME_PATH/include/acl/"*.h
 *   grep -rn "aclmdlRI" "$ASCEND_HOME_PATH/include/" | head
 *
 * 然后选择 ANYINFER_GRAPH_API：
 *   1 = aclrtStream* / aclrtGraph*（较新，本文件默认）
 *   2 = aclmdlRI*（较旧）
 *
 * 两个分支各写了最常见的写法。若你的头文件签名不同，**只需改本文件**，
 * 其余代码不碰 —— 这就是把不确定性收进一个可替换边界的目的。
 */

#ifndef ANYINFER_GRAPH_API
#define ANYINFER_GRAPH_API 1
#endif

#include <acl/acl.h>

namespace graph_probe {

#if ANYINFER_GRAPH_API == 1

using Graph = aclrtGraph;
using GraphExec = aclrtGraphExec;

inline aclError capture_begin(aclrtStream stream)
{
    return aclrtStreamBeginCapture(stream, ACL_STREAM_CAPTURE_MODE_GLOBAL);
}

inline aclError capture_end(aclrtStream stream, Graph *graph)
{
    return aclrtStreamEndCapture(stream, graph);
}

/* 注：部分版本是 aclrtGraphInstantiate(graph, &exec, nullptr, nullptr)，
 * 部分是两参数形式。若编译报参数个数不符，改这里。 */
inline aclError instantiate(Graph graph, GraphExec *exec)
{
    return aclrtGraphInstantiate(graph, exec);
}

inline aclError launch(GraphExec exec, aclrtStream stream)
{
    return aclrtGraphLaunch(exec, stream);
}

inline aclError destroy(Graph graph) { return aclrtGraphDestroy(graph); }
inline aclError destroy_exec(GraphExec exec) { return aclrtGraphExecDestroy(exec); }

/* 返回节点数；-1 表示该 API 在本次构建中不可用（不视为失败）。 */
inline int node_count(Graph graph)
{
    size_t count = 0;
    return aclrtGraphGetNodes(graph, nullptr, &count) == ACL_SUCCESS
               ? static_cast<int>(count)
               : -1;
}

#else

using Graph = aclmdlRI;
using GraphExec = aclmdlRI;

inline aclError capture_begin(aclrtStream stream)
{
    return aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
}

inline aclError capture_end(aclrtStream stream, Graph *graph)
{
    return aclmdlRICaptureEnd(stream, graph);
}

/* RI 模型没有独立的 instantiate 阶段：捕获结束即可用。 */
inline aclError instantiate(Graph, GraphExec *) { return ACL_SUCCESS; }

inline aclError launch(GraphExec exec, aclrtStream stream)
{
    return aclmdlRIExecuteAsync(exec, stream);
}

inline aclError destroy(Graph graph) { return aclmdlRIDestroy(graph); }
inline aclError destroy_exec(GraphExec) { return ACL_SUCCESS; }

inline int node_count(Graph) { return -1; }

#endif

} // namespace graph_probe
