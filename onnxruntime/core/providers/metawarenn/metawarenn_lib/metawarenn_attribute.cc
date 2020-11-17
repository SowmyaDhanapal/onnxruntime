#include "metawarenn_attribute.h"

namespace metawarenn {

MWNNAttribute::MWNNAttribute(AttributeProto& onnx_attribute_proto) {
    attribute_proto = onnx_attribute_proto;
    name = attribute_proto.name();
    type = attribute_proto.type();
    std::cout << "\n In MetawareNN Attribute Constructor";
    std::cout << "\n Name : " << get_name() << ", Type : " << static_cast<int>(get_type());
    switch (get_type()) {
    case Type::float_point:
      float_val = get_float();
      std::cout << "\n AttributeProto_AttributeType_FLOAT : " << float_val;
      break;
    case Type::integer:
      integer_val = get_int();
      std::cout << "\n AttributeProto_AttributeType_INT : " << integer_val;
      break;
    case Type::string:
      string_val = get_string();
      std::cout << "\n AttributeProto_AttributeType_STRING : " << string_val;
      break;
    case Type::tensor:
      std::cout << "\n AttributeProto_AttributeType_TENSOR : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case Type::graph:
      std::cout << "\n AttributeProto_AttributeType_GRAPH : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case Type::float_point_array:
      std::cout << "\n AttributeProto_AttributeType_FLOATS : ";
      float_array = get_float_array();
      for (auto& it : float_array)
         std::cout << "\t " << it;
      break;
    case Type::integer_array:
      std::cout << "\n AttributeProto_AttributeType_INTS : ";
      integer_array = get_integer_array();
      for (auto& it : integer_array)
           std::cout << "\t " << it;
      break;
    case Type::string_array:
      std::cout << "\n AttributeProto_AttributeType_STRINGS : ";
      string_array = get_string_array();
      for (auto& it : string_array)
         std::cout << "\t " << it;
      break;
    case Type::tensor_array:
      std::cout << "\n AttributeProto_AttributeType_TENSORS : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case Type::graph_array:
      std::cout << "\n AttributeProto_AttributeType_GRAPHS : Exiting Code Due to Nosupport";
      exit(1);
      break;
    default:
      std::cout << "\n In Default switch case";
      break;
    }
}

MWNNAttribute::~MWNNAttribute() {
    //std::cout << "\nIn MetawareNN Attribute Destructor";
}

} //namespace metawarenn
