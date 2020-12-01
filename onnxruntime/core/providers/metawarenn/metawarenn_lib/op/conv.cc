#include "conv.h"

namespace metawarenn {

namespace op {

Conv::Conv(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs,
           std::vector<float> n_dilations,
           std::vector<float> n_group,
           std::vector<float> n_strides,
           std::vector<float> n_kernel_size,
           std::vector<float> n_pads,
           std::vector<std::string> n_auto_pad) : Node(n_name, "Conv") {
  inputs = n_inputs;
  outputs = n_outputs;
  dilations = n_dilations;
  group = n_group;
  strides = n_strides;
  kernel_size = n_kernel_size;
  pads = n_pads;
  auto_pad = n_auto_pad;
  }
} //namespace op

} //namespace metawarenn
