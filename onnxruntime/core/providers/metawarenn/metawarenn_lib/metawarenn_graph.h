#ifndef METAWARENN_GRAPH_H_
#define METAWARENN_GRAPH_H_

#include "metawarenn_model.h"
#include "metawarenn_node.h"
#include "metawarenn_value_info.h"
#include "metawarenn_tensor.h"

namespace metawarenn {

class MWNNGraph {
  public:
    MWNNGraph(GraphProto& onnx_graph_proto, MWNNModel& model);
    ~MWNNGraph();
  private:
    GraphProto graph_proto;
    MWNNModel mwnn_model;
    std::vector<MWNNTensor> mwnn_initializer_tensors;
    std::vector<MWNNNode> mwnn_nodes;
    std::vector<MWNNValueInfo> mwnn_inputs;
    std::vector<MWNNValueInfo> mwnn_outputs;
};

} //namespace metawarenn

#endif //METAWARENN_GRAPH_H_
