#include "metawarenn_value_info.h"

namespace metawarenn {

MWNNValueInfo::MWNNValueInfo(ValueInfoProto& onnx_value_info_proto) {
    std::cout << "\n In MetawareNN Value Info Constructor";
    value_info_proto = onnx_value_info_proto;
    name = value_info_proto.name();
    std::cout << "\n Name : " << name;
    if(value_info_proto.type().tensor_type().has_elem_type()) {
        onnx_type = value_info_proto.type().tensor_type().elem_type();
        t_type = ElementType::get_mwnn_type(onnx_type);
        std::cout << "\n Element Types : ONNX : " << onnx_type << "MWNN : " << static_cast<int>(t_type);
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
