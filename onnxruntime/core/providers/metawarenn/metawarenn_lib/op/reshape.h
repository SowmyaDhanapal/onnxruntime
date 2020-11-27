#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Reshape : public Node {
  public:
    Reshape(std::string name);
};

} //namespace op

} //namespace metawarenn
