#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "basic_backend.h"

namespace onnxruntime {
namespace metawarenn {

class BackendManager {
 public:
  BackendManager(const onnxruntime::Node* fused_node, const logging::Logger& logger);

 private:
  ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(
      const onnxruntime::Node* fused_node, const logging::Logger& logger) const;
  ONNX_NAMESPACE::ModelProto model_proto_;
  std::shared_ptr<BasicBackend> metawarenn_backend_;
};

}  // namespace metawarenn
}  // namespace onnxruntime
