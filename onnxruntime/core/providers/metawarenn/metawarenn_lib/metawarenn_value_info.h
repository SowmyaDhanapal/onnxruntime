#ifndef METAWARENN_VALUE_INFO_H_
#define METAWARENN_VALUE_INFO_H_

#include "metawarenn_model.h"
#include "metawarenn_element.h"
#include "op/input_data.h"

namespace metawarenn {

class MWNNValueInfo {
  public:
    MWNNValueInfo(ValueInfoProto& onnx_value_info_proto);
    std::string get_name() { return name; }
    int get_type() { return onnx_type; }
    std::vector<int64_t> get_dims() { return dims; }
    std::shared_ptr<op::Node> get_node() {
      return std::make_shared<op::InputData>(name, dims, t_type);
    }
  private:
    ValueInfoProto value_info_proto;
    std::string name;
    int onnx_type;
    ElementType::element_type t_type;
    std::vector<int64_t> dims;
};

} //namespace metawarenn

#endif //METAWARENN_VALUE_INFO_H_
