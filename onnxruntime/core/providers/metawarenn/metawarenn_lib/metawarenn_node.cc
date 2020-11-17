#include "metawarenn_node.h"

namespace metawarenn {

MWNNNode::MWNNNode(NodeProto& onnx_node_proto) {
    std::cout << "\n In MetawareNN Node Constructor";
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
      }
    std::cout << "\n--------------------------------------------------------------\n";
    std::cout << "\nNode Op Type : " << op_type;
    std::cout << "\nNode Name : " << name;
    for (auto input : inputs)
        std::cout << "\nNode InPut : " << input;
    for (auto output : outputs)
        std::cout << "\nNode OutPut : " << output;
    std::cout << "\n--------------------------------------------------------------\n";
}

MWNNNode::~MWNNNode() {
    //std::cout << "\nIn MetawareNN Node Destructor";
}

} //namespace metawarenn
