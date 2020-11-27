#include "constant.h"

namespace metawarenn {

namespace op {

Constant::Constant(std::string name, std::vector<int64_t> shape, std::vector<float> data, ElementType::element_type data_type)
         : Node(name, "constant") {
  shape = shape;
  data = data;
  data_type = data_type;
}
} //namespace op

} //namespace metawarenn
