#include "avg_pool.h"

namespace metawarenn {

namespace op {

AvgPool::AvgPool(std::string name, std::vector<std::string> inputs,
                 std::vector<std::string> outputs) : Node(name, "AvgPool") {
  inputs = inputs;
  outputs = outputs;
  }
} //namespace op

} //namespace metawarenn
