#ifndef METAWARENN_VALUE_INFO_H_
#define METAWARENN_VALUE_INFO_H_

#include "metawarenn_model.h"
#include "metawarenn_element.h"

namespace metawarenn {

class MWNNValueInfo {
  public:
    MWNNValueInfo() = default;
    MWNNValueInfo(ValueInfoProto& onnx_value_info_proto);
    ~MWNNValueInfo();
  private:
    ValueInfoProto value_info_proto;
    std::string name;
    int onnx_type;
    ElementType::element_type t_type;
    std::vector<int64_t> dims;

};

} //namespace metawarenn

#endif //METAWARENN_VALUE_INFO_H_
