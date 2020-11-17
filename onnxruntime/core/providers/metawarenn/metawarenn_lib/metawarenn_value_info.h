#ifndef METAWARENN_VALUE_INFO_H_
#define METAWARENN_VALUE_INFO_H_

#include "metawarenn_model.h"

namespace metawarenn {

class MWNNValueInfo {
  public:
    MWNNValueInfo() = default;
    MWNNValueInfo(ValueInfoProto& onnx_value_info_proto);
    ~MWNNValueInfo();
  public:
    enum class element_type
      {
          boolean_,
          double_,
          float16_,
          float_,
          int8_,
          int16_,
          int32_,
          int64_,
          uint8_,
          uint16_,
          uint32_,
          uint64_,
          dynamic_
      };
  private:
    ValueInfoProto value_info_proto;
    std::string name;
    int onnx_type;
    element_type t_type;
    std::vector<int64_t> dims;

};

} //namespace metawarenn

#endif //METAWARENN_VALUE_INFO_H_
