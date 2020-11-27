#ifndef METAWARENN_GRAPH_H_
#define METAWARENN_GRAPH_H_

#include "metawarenn_model.h"
#include "metawarenn_node.h"
#include "metawarenn_value_info.h"
#include "metawarenn_tensor.h"
#include "op/node.h"

namespace metawarenn {

class MWNNGraph {
  public:
    MWNNGraph(GraphProto& onnx_graph_proto, MWNNModel& model);
    std::string get_name() { return name; }
    std::string get_graph_ip_name() { return ip_name; }
    std::string get_graph_op_name() { return op_name; }
    std::vector<MWNNTensor> get_graph_initializers() { return mwnn_initializer_tensors; }
    std::vector<MWNNNode> get_graph_nodes() { return mwnn_nodes; }
    std::vector<MWNNValueInfo> get_graph_inputs() { return mwnn_inputs; }
    std::vector<MWNNValueInfo> get_graph_outputs() { return mwnn_outputs; }
    std::set<std::string> mwnn_initializer_names;
  private:
    GraphProto graph_proto;
    MWNNModel mwnn_model;
    std::string name;
    std::string ip_name;
    std::string op_name;
    std::vector<MWNNTensor> mwnn_initializer_tensors;
    std::vector<MWNNNode> mwnn_nodes;
    std::vector<MWNNValueInfo> mwnn_inputs;
    std::vector<MWNNValueInfo> mwnn_outputs;
    std::map<std::string, op::Node> mwnn_graph_nodes;
};

} //namespace metawarenn

#endif //METAWARENN_GRAPH_H_
