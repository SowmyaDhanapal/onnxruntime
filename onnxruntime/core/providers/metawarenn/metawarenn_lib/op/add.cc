#include "add.h"

namespace metawarenn {

namespace op {

Add::Add(std::string name, std::vector<std::string> inputs,
         std::vector<std::string> outputs) : Node(name, "Add") {
  inputs = inputs;
  outputs = outputs;
  }
} //namespace op

} //namespace metawarenn
