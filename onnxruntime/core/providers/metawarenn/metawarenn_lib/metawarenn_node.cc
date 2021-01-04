#include "metawarenn_node.h"

namespace metawarenn {

MWNNNode::MWNNNode(NodeProto& onnx_node_proto) {
  node_proto = onnx_node_proto;
  name = node_proto.name();
  op_type = node_proto.op_type();
  for (auto input : node_proto.input()) {
    inputs.emplace_back(input);
  }
  for (auto output : node_proto.output()) {
    outputs.emplace_back(output);
  }
  for (auto attribute_proto : node_proto.attribute()) {
    MWNNAttribute mwnn_attribute(attribute_proto);
    mwnn_attributes.emplace_back(mwnn_attribute);
    if(mwnn_attribute.get_name() == "group")
    {
      op_type = (int)mwnn_attribute.get_data()[0] == 1 ? "Conv" : "DepthwiseConv";
    }
  }
}
} //namespace metawarenn
