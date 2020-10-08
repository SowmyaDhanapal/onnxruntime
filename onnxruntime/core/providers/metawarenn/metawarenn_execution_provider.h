#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"
#include "backend_manager.h"

namespace onnxruntime {

class MetaWareNNExecutionProvider : public IExecutionProvider {
public:
  MetaWareNNExecutionProvider();
  ~MetaWareNNExecutionProvider();
  
  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;
  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
};

}  // namespace onnxruntime
