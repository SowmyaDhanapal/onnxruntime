#include "metawarenn_value_info.h"

namespace metawarenn {

MWNNValueInfo::MWNNValueInfo(ValueInfoProto& onnx_value_info_proto) {
    std::cout << "\n In MetawareNN Value Info Constructor";
    value_info_proto = onnx_value_info_proto;
    name = value_info_proto.name();
    std::cout << "\n Name : " << name;
    if(value_info_proto.type().tensor_type().has_elem_type()) {
        onnx_type = value_info_proto.type().tensor_type().elem_type();
        switch (onnx_type) {
            case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
                t_type = element_type::boolean_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
                t_type = element_type::double_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
                t_type = element_type::float16_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
                t_type = element_type::float_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_INT8:
                t_type = element_type::int8_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_INT16:
                t_type = element_type::int16_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_INT32:
                t_type = element_type::int32_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_INT64:
                t_type = element_type::int64_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
                t_type = element_type::uint8_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
                t_type = element_type::uint16_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
                t_type = element_type::uint32_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
                t_type = element_type::uint64_;
                break;
            case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
                t_type = element_type::dynamic_;
                break;
            default:
                t_type = element_type::dynamic_;
                break;
        }
        //std::cout << "\n Types : ONNX : " << onnx_type << "MWNN : " << static_cast<int>(t_type);
        std::cout << "\n Shape : " ;
        for (const auto& onnx_dim : value_info_proto.type().tensor_type().shape().dim()) {
            if (onnx_dim.has_dim_value()) {
                std::cout << "\t" << onnx_dim.dim_value();
                dims.emplace_back(onnx_dim.dim_value());
            }
            else {
                dims.emplace_back(0);
                std::cout << "\t dynamic";
            }
        }
    }
}

MWNNValueInfo::~MWNNValueInfo() {
    //std::cout << "\nIn MetawareNN Value Info Destructor";
}

} //namespace metawarenn
