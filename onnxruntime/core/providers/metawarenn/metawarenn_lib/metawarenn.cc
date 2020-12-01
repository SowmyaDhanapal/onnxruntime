#include "metawarenn.h"

namespace metawarenn {


std::shared_ptr<Function> import_onnx_model(std::istream& stream) {
  std::cout << "\n import_onnx_model";
  ModelProto model_proto;
  if (!model_proto.ParseFromIstream(&stream))
    std::cout << "\n Error in parsing model stream model!!!";
  else
    std::cout << "\n Sucessfully parsed the model stream buffer!!!";

  GraphProto graph_proto = model_proto.graph();

  MWNNModel mwnn_model(model_proto);
  MWNNGraph mwnn_graph(graph_proto, mwnn_model);

  std::cout << "\n ---------------------------Graph----------------------------- \n";
  std::cout << "\n Graph Name : " << mwnn_graph.get_name();
  std::cout << "\n Graph Input Name : " << mwnn_graph.get_graph_ip_name();
  std::cout << "\n Graph Output Name : " << mwnn_graph.get_graph_op_name();

  //Sort Node names
  {
    std::vector<std::string> sorted_names;
    for (auto g_n : mwnn_graph.get_graph_nodes()) {
      std::cout << "\n Node name : " << g_n.get_name();
      for (auto n_ip : g_n.get_inputs()) {
        if(n_ip == mwnn_graph.get_graph_ip_name()) {
          sorted_names.push_back(n_ip);
          }
        else if(mwnn_graph.mwnn_initializer_names.count(n_ip)) {
          sorted_names.push_back(n_ip);
          }
        else if(std::count(sorted_names.begin(), sorted_names.end(), n_ip)) {
          continue;
          }
        else {
          std::cout << "\nERROR : Input Not available ";
          std::cout << "\nInput Name : " << n_ip;
          exit(1);
        }
      }
      for (auto n_op : g_n.get_outputs()) {
        sorted_names.push_back(n_op);
      }
    }
     for (auto itr = sorted_names.begin(); itr != sorted_names.end(); ++itr)
    {
        std::cout << "\n" << *itr;
    }
  }

  //Sort Nodes
  {
  std::vector<op::Node> sorted_nodes;
  std::vector<std::string> output_names;
    for (auto g_n : mwnn_graph.get_graph_nodes()) {
      std::cout << "\n Node name : " << g_n.get_name();
      for (auto n_ip : g_n.get_inputs()) {
        if(std::count(output_names.begin(), output_names.end(), n_ip)) {
          continue;
        }
        else if(mwnn_graph.mwnn_graph_nodes.count(n_ip)) {
          sorted_nodes.push_back(mwnn_graph.mwnn_graph_nodes[n_ip]);
        }
        else {
          std::cout << "\nERROR : Input Not available ";
          std::cout << "\nInput Name : " << n_ip;
          exit(1);
        }
      }
      sorted_nodes.push_back(mwnn_graph.mwnn_graph_nodes[g_n.get_name()]);
      for (auto n_op : g_n.get_outputs()) {
        output_names.push_back(n_op);
      }
    }
    for (auto itr = sorted_nodes.begin(); itr != sorted_nodes.end(); ++itr) {
        std::cout << "\n" << itr->name;
    }
  }
  std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
  for (auto g_ip : mwnn_graph.get_graph_inputs()) {
    std::cout << "\n Input Name : " << g_ip.get_name();
    std::cout << "\n Data Type : " << g_ip.get_type();
    std::cout << "\n Input Dims : ";
    for (auto dim : g_ip.get_dims())
      std::cout << dim << ",";
  }
  std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
  for (auto g_op : mwnn_graph.get_graph_outputs()) {
    std::cout << "\n Output Name : " << g_op.get_name();
    std::cout << "\n Data Type : " << g_op.get_type();
    std::cout << "\n Output Dims : ";
    for (auto dim : g_op.get_dims())
      std::cout << dim << ",";
  }
  std::cout << "\n -----------------------Graph Nodes-------------------------- \n";
  for (auto g_n : mwnn_graph.get_graph_nodes()) {
    std::cout << "\n ================================================================ \n";
    std::cout << "\n Node Name : " << g_n.get_name();
    std::cout << "\n Op Type : " << g_n.get_op_type();
    for (auto n_ip : g_n.get_inputs())
      std::cout << "\n Input : " << n_ip;
    for (auto n_op : g_n.get_outputs())
      std::cout << "\n Output : " << n_op;
    std::cout << "\n ---------------------------------------------------------------- ";
    for (auto attribute : g_n.get_attributes()) {
      std::cout << "\n Attribute Name : " << attribute.get_name();
      std::cout << "\n Attribute Data Type : " << attribute.get_type();
      std::cout << "\n Attribute Data : ";
      if(attribute.get_type() == 3 || attribute.get_type() == 8) {
        for (auto str_data : attribute.get_string_data())
          std::cout << str_data << ",";
      }
      else {
      for (auto data : attribute.get_data())
        std::cout << data << ",";
      }
    }
  }
  std::cout << "\n -----------------------Graph Tensors-------------------------- \n";
  for (auto g_t : mwnn_graph.get_graph_initializers()) {
    std::cout << "\n Name : " << g_t.get_name();
    std::cout << "\n Type : " << g_t.get_type();
    std::cout << "\n Dims : ";
    for (auto dim : g_t.get_dims())
      std::cout << dim << ",";
    //Uncomment to Print the Tensor Values
    /*std::cout << "\n Tensor : [";
    for (auto t_val : g_t.get_tensor())
      std::cout << t_val << ",";
    std::cout << "]";*/
  }
  return nullptr;
}

} //namespace metawarenn

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
