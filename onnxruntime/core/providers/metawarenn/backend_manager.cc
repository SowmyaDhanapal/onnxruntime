#include "backend_manager.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace metawarenn {

BackendManager::BackendManager(const onnxruntime::Node* fused_node, const logging::Logger& logger) {
  std::cout << "\n  --> In MetaWareNN BackendManager()  \n";
  model_proto_ = GetModelProtoFromFusedNode(fused_node, logger);
  metawarenn_backend_ = std::make_shared<BasicBackend>(model_proto_);
}

ONNX_NAMESPACE::ModelProto
BackendManager::GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node,
                                           const logging::Logger& logger) const {
  std::cout << "\n   ---> Generating Model Proto!!! \n";
  const auto* node_function = fused_node->GetFunctionBody();
  const std::string& name = fused_node->Name();
  ORT_ENFORCE(node_function != nullptr, "Could not extract function body for node: ", name);

  const onnxruntime::Graph& node_subgraph = node_function->Body();
  onnxruntime::Model model(node_subgraph.Name(), true, ModelMetaData{}, onnxruntime::ToPathString(""),
                           IOnnxRuntimeOpSchemaRegistryList{}, node_subgraph.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *(model_proto.mutable_graph()) = node_subgraph.ToGraphProto();
  return model_proto;
}

void BackendManager::Compute(Ort::CustomOpApi api, OrtKernelContext* context) {
  //Inference Part
  metawarenn_backend_->Infer(api, context);
}

}  // namespace metawarenn
}  // namespace onnxruntime
