#include "relu.h"

namespace metawarenn {

namespace op {

Relu::Relu(std::string name, std::vector<std::string> inputs,
           std::vector<std::string> outputs) : Node(name, "Relu") {
  inputs = inputs;
  outputs = outputs;
  }
} //namespace op

} //namespace metawarenn
