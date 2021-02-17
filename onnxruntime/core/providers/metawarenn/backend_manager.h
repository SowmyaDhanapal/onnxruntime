#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "basic_backend.h"
#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_utils.h"
#include "metawarenn.h"

namespace onnxruntime {
namespace metawarenn_ep {

class BackendManager {
 public:
  BackendManager(const onnxruntime::Node* fused_node, const logging::Logger& logger);
  void Compute(Ort::CustomOpApi api, OrtKernelContext* context, metawarenn::MWNNGraph& mwnn_graph);
  metawarenn::MWNNGraph mwnn_graph_;

 private:
  ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(
      const onnxruntime::Node* fused_node, const logging::Logger& logger) const;
  std::shared_ptr<BasicBackend> metawarenn_backend_;
};

}  // namespace metawarenn_ep
}  // namespace onnxruntime
