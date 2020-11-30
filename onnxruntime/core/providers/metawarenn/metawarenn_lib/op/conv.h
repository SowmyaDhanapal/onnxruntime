#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Conv : public Node {
  public:
    Conv(std::string name, std::vector<std::string> inputs,
         std::vector<std::string> outputs,
         std::vector<float> dilations,
         std::vector<float> group,
         std::vector<float> strides,
         std::vector<float> kernel_size,
         std::vector<float> pads,
         std::vector<std::string> auto_pad);
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<float> dilations;
    std::vector<float> group;
    std::vector<float> strides;
    std::vector<float> kernel_size;
    std::vector<float> pads;
    std::vector<std::string> auto_pad;
};

} //namespace op

} //namespace metawarenn
