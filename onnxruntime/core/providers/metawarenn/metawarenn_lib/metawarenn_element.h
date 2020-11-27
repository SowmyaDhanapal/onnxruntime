#ifndef METAWARENN_ELEMENT_H_
#define METAWARENN_ELEMENT_H_

#ifdef ONNX_ML
#include "onnx/onnx-ml.pb.h"
#else
#include "onnx/onnx.pb.h"
#endif

using namespace ONNX_NAMESPACE;

namespace metawarenn {

class ElementType {
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

    static element_type get_mwnn_type(int onnx_type) {
        switch (onnx_type) {
            case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
                return element_type::boolean_;
            case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
                return element_type::double_;
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
                return element_type::float16_;
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
                return element_type::float_;
            case ONNX_NAMESPACE::TensorProto_DataType_INT8:
                return element_type::int8_;
            case ONNX_NAMESPACE::TensorProto_DataType_INT16:
                return element_type::int16_;
            case ONNX_NAMESPACE::TensorProto_DataType_INT32:
                return element_type::int32_;
            case ONNX_NAMESPACE::TensorProto_DataType_INT64:
                return element_type::int64_;
            case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
                return element_type::uint8_;
            case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
                return element_type::uint16_;
            case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
                return element_type::uint32_;
            case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
                return element_type::uint64_;
            case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
                return element_type::dynamic_;
            default:
                return element_type::dynamic_;
        }
    }
};

} //namespace metawarenn

#endif //METAWARENN_ELEMENT_H_
