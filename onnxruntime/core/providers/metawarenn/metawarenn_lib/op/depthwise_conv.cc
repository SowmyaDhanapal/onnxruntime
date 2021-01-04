#include "depthwise_conv.h"

namespace metawarenn {

namespace op {

DepthwiseConv::DepthwiseConv(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs,
           std::vector<float> n_dilations,
           std::vector<float> n_strides,
           std::vector<float> n_kernel_size,
           std::vector<float> n_pads,
           std::vector<std::string> n_auto_pad) : Node(n_name, "DepthwiseConv") {
  inputs = n_inputs;
  outputs = n_outputs;
  dilations = n_dilations;
  strides = n_strides;
  kernel_size = n_kernel_size;
  pads = n_pads;
  auto_pad = n_auto_pad;
  }
} //namespace op

} //namespace metawarenn
