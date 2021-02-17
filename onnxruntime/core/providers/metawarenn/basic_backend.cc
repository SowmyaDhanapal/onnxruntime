#include "core/graph/graph.h"
#include "basic_backend.h"

#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace metawarenn_ep {

BasicBackend::BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto, metawarenn::MWNNGraph& mwnn_graph) {
  //To Create a MetaWareNN supported graph along with graph optimization.
  CreateCNNNetwork(model_proto, mwnn_graph);
  }
}  // namespace metawarenn_ep
}  // namespace onnxruntime
