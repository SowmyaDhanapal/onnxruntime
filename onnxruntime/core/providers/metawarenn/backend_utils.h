#include "metawarenn.h"

namespace onnxruntime {
namespace metawarenn_ep {

std::shared_ptr<InferenceEngine::CNNNetwork>
CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto);

void SetIODefs(/*const ONNX_NAMESPACE::ModelProto& model_proto,
               std::shared_ptr<InferenceEngine::CNNNetwork> network*/);

InferenceEngine::Precision ConvertPrecisionONNXToOpenVINO(const ONNX_NAMESPACE::TypeProto& onnx_type);

}  // namespace metawarenn_ep
}  // namespace onnxruntime
