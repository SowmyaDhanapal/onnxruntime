#ifndef METAWARENN_NODE_H_
#define METAWARENN_NODE_H_

#include "metawarenn_model.h"
#include "metawarenn_attribute.h"

namespace metawarenn {

class MWNNNode {
  public:
    MWNNNode() = default;
    MWNNNode(NodeProto& onnx_node_proto);
    ~MWNNNode();
  private:
    NodeProto node_proto;
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<MWNNAttribute> mwnn_attributes;
};

} //namespace metawarenn

#endif //METAWARENN_NODE_H_
