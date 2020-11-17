#ifndef METAWARENN_GRAPH_H_
#define METAWARENN_GRAPH_H_

#include "metawarenn_model.h"
#include "metawarenn_node.h"

namespace metawarenn {

class MWNNGraph {
  public:
    MWNNGraph(GraphProto& onnx_graph_proto, MWNNModel& model);
    ~MWNNGraph();
  private:
    GraphProto graph_proto;
    MWNNModel mwnn_model;
    std::vector<MWNNNode> mwnn_nodes;
};

} //namespace metawarenn

#endif //METAWARENN_GRAPH_H_
