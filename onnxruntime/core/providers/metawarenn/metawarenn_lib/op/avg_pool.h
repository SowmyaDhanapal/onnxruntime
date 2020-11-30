#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class AvgPool : public Node {
  public:
    AvgPool(std::string name, std::vector<std::string> inputs,
            std::vector<std::string> outputs);
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

} //namespace op

} //namespace metawarenn
