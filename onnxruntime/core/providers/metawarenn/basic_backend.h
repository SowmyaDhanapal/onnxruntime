#include "backend_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace metawarenn_ep {

class BasicBackend {
  public:
    BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto, metawarenn::MWNNGraph& mwnn_graph_);

  private:
    std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network_;
    InferenceEngine::ExecutableNetwork exe_network;
    InferenceEngine::InferRequest infer_req;
    mutable std::mutex compute_lock_;
};
}  // namespace metawarenn_ep
}  // namespace onnxruntime
