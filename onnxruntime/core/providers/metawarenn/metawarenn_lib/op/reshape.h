#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Reshape : public Node {
  public:
    Reshape(std::string name, std::vector<std::string> inputs,
            std::vector<std::string> outputs);
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

} //namespace op

} //namespace metawarenn
