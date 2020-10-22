#include "metawarenn.h"
#ifdef ONNX_ML
#include "onnx/onnx-ml.pb.h"
#else
#include "onnx/onnx.pb.h"
#endif

using namespace ONNX_NAMESPACE;

namespace graph {

namespace onnx {

  std::shared_ptr<graph::Function> import_onnx_model(std::istream& stream) {
    std::cout << "\n import_onnx_model";
    ModelProto model_proto;

    if (!model_proto.ParseFromIstream(&stream)) {
      std::cout << "\n Error in parsing model stream model!!!";
    }
    else {
      std::cout << "\n Sucessfully parsed the model stream buffer!!!";
    }

    GraphProto* graph_proto = model_proto.mutable_graph();

    for (auto& node : *graph_proto->mutable_node()) {
      node.set_domain("MetaWareNN");
      //std::cout << "\nnode.op_type() : " << node.op_type() << " -- " << node.domain();
    }

    std::cout << "\n\n ----------------Graph proto Initializers-----------------\n\n";
    for (const auto& initializer_tensor : graph_proto->initializer()) {
      std::cout << "\n Name : " << initializer_tensor.name();
    }

    std::cout << "\n\n -------------------Graph proto Input---------------------\n\n";
    for (const auto& input : graph_proto->input()) {
    std::cout << "\n Name : " << input.name();
    }

    std::cout << "\n\n -------------------Graph proto Output---------------------\n\n";
    for (const auto& output : graph_proto->output()) {
    std::cout << "\n Name : " << output.name();
    }

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
