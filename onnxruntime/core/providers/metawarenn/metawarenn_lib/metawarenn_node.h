#ifndef METAWARENN_NODE_H_
#define METAWARENN_NODE_H_

#include "metawarenn_model.h"
#include "metawarenn_attribute.h"
#include "op/conv.h"
#include "op/relu.h"
#include "op/add.h"
#include "op/avg_pool.h"
#include "op/reshape.h"

namespace metawarenn {

class MWNNNode {
  public:
    MWNNNode(NodeProto& onnx_node_proto);
    std::string get_name() { return name; }
    std::string get_op_type() { return op_type; }
    std::vector<std::string> get_inputs() { return inputs; }
    std::vector<std::string> get_outputs() { return outputs; }
    std::vector<MWNNAttribute> get_attributes() { return mwnn_attributes; }
    std::shared_ptr<op::Node> get_node() {
      if(op_type == "Conv")
        return std::make_shared<op::Conv>(name);
      else if(op_type == "Relu")
        return std::make_shared<op::Relu>(name);
      else if(op_type == "Add")
        return std::make_shared<op::Add>(name);
      else if(op_type == "GlobalAveragePool")
        return std::make_shared<op::AvgPool>(name);
      else if(op_type == "Reshape")
        return std::make_shared<op::Reshape>(name);
      else
        return NULL;
    }
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
