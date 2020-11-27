#pragma once

#include "node.h"
#include "../metawarenn_element.h"

namespace metawarenn {

namespace op {

class Constant : public Node {
  public:
    Constant(std::string name, std::vector<int64_t> shape, std::vector<float> data, ElementType::element_type data_type);
  private:
    std::vector<int64_t> shape;
    std::vector<float> data;
    ElementType::element_type data_type;
};

} //namespace op

} //namespace metawarenn
