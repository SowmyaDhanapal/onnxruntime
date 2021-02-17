#include "metawarenn.h"

namespace onnxruntime {
namespace metawarenn_ep {

void CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto, metawarenn::MWNNGraph& mwnn_graph);

void SetIODefs(/*const ONNX_NAMESPACE::ModelProto& model_proto,
               std::shared_ptr<InferenceEngine::CNNNetwork> network*/);

InferenceEngine::Precision ConvertPrecisionONNXToOpenVINO(const ONNX_NAMESPACE::TypeProto& onnx_type);

}  // namespace metawarenn_ep
}  // namespace onnxruntime
