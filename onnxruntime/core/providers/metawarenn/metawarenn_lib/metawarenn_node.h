#ifndef METAWARENN_NODE_H_
#define METAWARENN_NODE_H_

#include "metawarenn_model.h"
#include "metawarenn_attribute.h"
#include "op/conv.h"
#include "op/depthwise_conv.h"
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

    std::vector<float> get_attribute_value(std::string name) {
      auto it = std::find_if(
      std::begin(mwnn_attributes), std::end(mwnn_attributes), [&](MWNNAttribute& attribute) {
          return attribute.get_name() == name;
      });
      if (it == std::end(mwnn_attributes)) {
          std::cout << "\n ERROR : End of Attributes!!! - Couldn't find " << name;
      }
      return it->get_data();
    }
    std::vector<std::string> get_attribute_string(std::string name) {
      auto it = std::find_if(
      std::begin(mwnn_attributes), std::end(mwnn_attributes), [&](MWNNAttribute& attribute) {
          return attribute.get_name() == name;
      });
      if (it == std::end(mwnn_attributes)) {
          std::cout << "\n ERROR : End of Attributes!!! - Couldn't find " << name;
      }
      return it->get_string_data();
    }
    std::shared_ptr<op::Node> get_node() {
      if(op_type == "Conv") {
        return std::make_shared<op::Conv>(name, inputs, outputs,
                                          get_attribute_value("dilations"),
                                          get_attribute_value("strides"),
                                          get_attribute_value("kernel_shape"),
                                          get_attribute_value("pads"),
                                          get_attribute_string("auto_pad"));
      }
      else if(op_type == "DepthwiseConv") {
        return std::make_shared<op::DepthwiseConv>(name, inputs, outputs,
                                          get_attribute_value("dilations"),
                                          get_attribute_value("strides"),
                                          get_attribute_value("kernel_shape"),
                                          get_attribute_value("pads"),
                                          get_attribute_string("auto_pad"));
      }
      else if(op_type == "Relu") {
        return std::make_shared<op::Relu>(name, inputs, outputs);
      }
      else if(op_type == "Add") {
        return std::make_shared<op::Add>(name, inputs, outputs);
      }
      else if(op_type == "GlobalAveragePool") {
        return std::make_shared<op::AvgPool>(name, inputs, outputs);
      }
      else if(op_type == "Reshape") {
        return std::make_shared<op::Reshape>(name, inputs, outputs);
        }
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
