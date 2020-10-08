#include "core/graph/graph.h"
#include "basic_backend.h"

namespace onnxruntime {
namespace metawarenn {

BasicBackend::BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto) {
    std::cout << "\n   ---> In Basic Backend Empty Constructor\n" << model_proto.producer_name();
    //To Create a MetaWareNN supported graph along with graph optimization.
    //CreateCNNNetwork(model_proto);
    //Load Model into the Plugin
    //LoadNetwork()
    //Create Inference request pointer
    //CreateInferRequestPtr()
}
}  // namespace metawarenn
}  // namespace onnxruntime
