#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class AvgPool : public Node {
  public:
    AvgPool(std::string name);
};

} //namespace op

} //namespace metawarenn
