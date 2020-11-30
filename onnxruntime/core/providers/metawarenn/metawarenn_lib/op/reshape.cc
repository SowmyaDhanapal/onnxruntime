#include "reshape.h"

namespace metawarenn {

namespace op {

Reshape::Reshape(std::string name, std::vector<std::string> inputs,
                 std::vector<std::string> outputs) : Node(name, "Reshape") {
  inputs = inputs;
  outputs = outputs;
  }
} //namespace op

} //namespace metawarenn
