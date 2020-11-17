#include "metawarenn_graph.h"

namespace metawarenn {

MWNNGraph::MWNNGraph(GraphProto& onnx_graph_proto, MWNNModel& model) {
    graph_proto = onnx_graph_proto;
    mwnn_model = model;
    std::cout << "\n In MetawareNN Graph Constructor";

    for (auto node_proto : graph_proto.node()) {
      MWNNNode mwnn_node(node_proto);
      mwnn_nodes.emplace_back(mwnn_node);
    }
    for (auto ip_value_info_proto : graph_proto.input()) {
      MWNNValueInfo mwnn_input(ip_value_info_proto);
      mwnn_inputs.emplace_back(mwnn_input);
    }
    for (auto op_value_info_proto : graph_proto.output()) {
      MWNNValueInfo mwnn_output(op_value_info_proto);
      mwnn_outputs.emplace_back(mwnn_output);
    }
}
MWNNGraph::~MWNNGraph() {
    //std::cout << "\n In MetawareNN Graph Destructor";
}

} //namespace metawarenn
