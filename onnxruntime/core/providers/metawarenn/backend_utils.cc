#include "core/graph/graph.h"
#include "backend_utils.h"

namespace onnxruntime {

namespace metawarenn_ep {

void CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto, metawarenn::MWNNGraph& mwnn_graph)
{
  std::cout << "\n In CreateCNNNetwork!!";
  std::istringstream model_stream{model_proto.SerializeAsString()};
  metawarenn::import_onnx_model(model_stream, mwnn_graph);
}
}  // namespace metawarenn_ep
}  // namespace onnxruntime
