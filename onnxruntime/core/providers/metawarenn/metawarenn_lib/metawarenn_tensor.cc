#include "metawarenn_tensor.h"

namespace metawarenn {

MWNNTensor::MWNNTensor(TensorProto& onnx_tensor_proto) {
  tensor_proto = onnx_tensor_proto;
  name = tensor_proto.name();
  onnx_type = tensor_proto.data_type();
  t_type = ElementType::get_mwnn_type(onnx_type);
  for(auto dim : tensor_proto.dims()) {
    dims.emplace_back(dim);
  }
  set_tensor();
}

void MWNNTensor::set_tensor() {
  switch (onnx_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      tensor = get_data<float>(tensor_proto.float_data());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      tensor = get_data<float>(tensor_proto.int64_data());
      break;
    default:
      break;
  }
}
} //namespace metawarenn
