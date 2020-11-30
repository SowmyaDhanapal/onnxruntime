#include "conv.h"

namespace metawarenn {

namespace op {

Conv::Conv(std::string name, std::vector<std::string> inputs,
           std::vector<std::string> outputs,
           std::vector<float> dilations,
           std::vector<float> group,
           std::vector<float> strides,
           std::vector<float> kernel_size,
           std::vector<float> pads,
           std::vector<std::string> auto_pad) : Node(name, "Conv") {
  inputs = inputs;
  outputs = outputs;
  dilations = dilations;
  group = group;
  strides = strides;
  kernel_size = kernel_size;
  pads = pads;
  auto_pad = auto_pad;
  }
} //namespace op

} //namespace metawarenn
