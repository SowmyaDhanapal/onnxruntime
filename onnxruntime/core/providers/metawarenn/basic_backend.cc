#include "core/graph/graph.h"
#include "basic_backend.h"

#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace metawarenn_ep {

BasicBackend::BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto) {
  std::cout << "\n   ---> In Basic Backend Empty Constructor\n" << model_proto.producer_name();
  //To Create a MetaWareNN supported graph along with graph optimization.
  ie_cnn_network_ = CreateCNNNetwork(model_proto);
  //Set EP's Data Type for Model Inputs & Outputs
  SetIODefs(/*model_proto, ie_cnn_network_*/);
  //Load Model into the Plugin
  exe_network = exe_network.LoadNetwork(/*ie_cnn_network_*/);
  //Create Inference request pointer
  infer_req = exe_network.CreateInferRequestPtr();
  }

void BasicBackend::Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // Currently allows only one Infer execution at a time

  std::cout << "\n   ---> In Basic Backend Infer Function()\n";
  std::cout << reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->InputCount();
  std::cout << ort.KernelContext_GetInputCount(context);
  std::lock_guard<std::mutex> lock(compute_lock_);

  //TODO the below functions will be impl in backend_utils.cc
  //which internally calls EV EP using ie_cnn_network_.
  //size_t batch_size = 1;
  // Get Input and Output tensors
  //auto input_tensors = GetInputTensors(ort, context, ie_cnn_network_, subgraph_context_.input_indexes);
  //auto output_tensors = GetOutputTensors(ort, context, batch_size, infer_request_, ie_cnn_network_, subgraph_context_.output_names);

  //StartAsyncInference(ort, input_tensors, infer_request_, ie_cnn_network_);
  //CompleteAsyncInference(ort, output_tensors, infer_request_, ie_cnn_network_);
}

}  // namespace metawarenn_ep
}  // namespace onnxruntime
