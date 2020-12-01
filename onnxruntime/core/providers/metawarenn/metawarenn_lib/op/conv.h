#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Conv : public Node {
  public:
    Conv(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs,
         std::vector<float> n_dilations,
         std::vector<float> n_group,
         std::vector<float> n_strides,
         std::vector<float> n_kernel_size,
         std::vector<float> n_pads,
         std::vector<std::string> n_auto_pad);
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
