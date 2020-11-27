#include "metawarenn_graph.h"

namespace metawarenn {

MWNNGraph::MWNNGraph(GraphProto& onnx_graph_proto, MWNNModel& model) {
  graph_proto = onnx_graph_proto;
  mwnn_model = model;
  name = graph_proto.name();

  for (auto tensor_proto : graph_proto.initializer()) {
    MWNNTensor mwnn_tensor(tensor_proto);
    mwnn_initializer_tensors.emplace_back(mwnn_tensor);
    mwnn_initializer_names.insert(mwnn_tensor.get_name());
    auto const_node = mwnn_tensor.get_constant_node();
    mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(*const_node);
  }
  for (auto node_proto : graph_proto.node()) {
    MWNNNode mwnn_node(node_proto);
    mwnn_nodes.emplace_back(mwnn_node);
    auto node = mwnn_node.get_node();
    mwnn_graph_nodes[mwnn_node.get_name()] = std::move(*node);
  }
  for (auto ip_value_info_proto : graph_proto.input()) {
    MWNNValueInfo mwnn_input(ip_value_info_proto);
    mwnn_inputs.emplace_back(mwnn_input);
    if(mwnn_initializer_names.count(mwnn_input.get_name()))
      continue;
    else
      ip_name = mwnn_input.get_name();
  }
  for (auto op_value_info_proto : graph_proto.output()) {
    MWNNValueInfo mwnn_output(op_value_info_proto);
    mwnn_outputs.emplace_back(mwnn_output);
    op_name = mwnn_output.get_name();
  }
}
} //namespace metawarenn
