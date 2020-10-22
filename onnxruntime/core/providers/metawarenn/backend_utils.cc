#include "core/graph/graph.h"
#include "backend_utils.h"

namespace onnxruntime {
namespace metawarenn {

std::shared_ptr<InferenceEngine::CNNNetwork>
CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto) {
  
  std::cout << "\n In CreateCNNNetwork!!";

  std::istringstream model_stream{model_proto.SerializeAsString()};
  std::shared_ptr<graph::Function> graph_func;

  try {
    graph_func = graph::onnx::import_onnx_model(/*model_stream*/);
    //LOGS_DEFAULT(INFO) << "ONNX Import Done";
  } catch (const std::exception& exp) {
    //ORT_THROW(log_tag + "[MetaWareNN-EP] Exception while importing model to nGraph Func: " + std::string(exp.what()));
  } catch (...) {
    //ORT_THROW(log_tag + "[MetaWareNN-EP] Unknown exception while importing model to nGraph Func");
  }

  return std::make_shared<InferenceEngine::CNNNetwork>(/*graph_func*/);

}

//TODO: Update for MetaWare datatype
InferenceEngine::Precision ConvertPrecisionONNXToOpenVINO(const ONNX_NAMESPACE::TypeProto& onnx_type) {
  ONNX_NAMESPACE::DataType type_string = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(onnx_type);
  if (*type_string == "float" || *type_string == "tensor(float)") {
    return InferenceEngine::Precision::FP32;
  }
  //just to get the default constructor object for return type
  return InferenceEngine::Precision();
}

void SetIODefs(/*const ONNX_NAMESPACE::ModelProto& model_proto,
               std::shared_ptr<InferenceEngine::CNNNetwork> network*/) {
  //GetInput Information from network
  //GetOutput Information from network
  //Iterate over both the inputs and outputs and get the precision type using
  //ConvertPrecisionONNXToOpenVINO() and 
  //Set_the_EV_precision() to the network for its ip & op
}
}  // namespace metawarenn
}  // namespace onnxruntime
