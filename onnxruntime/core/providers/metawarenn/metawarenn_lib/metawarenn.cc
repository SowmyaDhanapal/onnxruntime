#include "metawarenn.h"

namespace graph {

namespace onnx {

  std::shared_ptr<graph::Function> import_onnx_model(/*std::istream& stream*/) {
    std::cout << "\n import_onnx_model";
    return nullptr;
  }

} //namespace onnx

} //namespace graph

namespace InferenceEngine {

  CNNNetwork::CNNNetwork(/*const std::shared_ptr<graph::Function>& network*/) {
    std::cout << "\n CNNNetwork";
  }

  Precision::Precision() {}
  Precision::Precision(const Precision::ePrecision value) {
    std::cout << "\n Precision type " << value;
  }

  InferRequest::InferRequest() {}

  ExecutableNetwork::ExecutableNetwork() {}
  ExecutableNetwork ExecutableNetwork::LoadNetwork(/*const CNNNetwork& network*/) {
  std::cout << "\n LoadNetwork";
  ExecutableNetwork exe_nw;
  return exe_nw;
  }
  
  InferRequest ExecutableNetwork::CreateInferRequestPtr() {
  std::cout << "\n CreateInferRequestPtr";
  InferRequest infer_req_;
  return infer_req_;
  }

  } //namespace InferenceEngine
