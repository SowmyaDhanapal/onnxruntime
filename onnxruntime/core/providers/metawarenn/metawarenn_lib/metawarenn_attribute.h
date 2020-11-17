#ifndef METAWARENN_ATTRIBUTE_H_
#define METAWARENN_ATTRIBUTE_H_

#include "metawarenn_model.h"

namespace metawarenn {

class MWNNAttribute {
  public:
    MWNNAttribute() = default;
    MWNNAttribute(AttributeProto& onnx_attribute_proto);
    ~MWNNAttribute();
  private:
    AttributeProto attribute_proto;
    std::string name;
    int type;
    float float_val;
    int64_t integer_val;
    std::string string_val;
    std::vector<float> float_array;
    std::vector<int64_t> integer_array;
    std::vector<std::string> string_array;
  public:
    enum class Type
    {
        undefined = AttributeProto_AttributeType_UNDEFINED,
        float_point = AttributeProto_AttributeType_FLOAT,
        integer = AttributeProto_AttributeType_INT,
        string = AttributeProto_AttributeType_STRING,
        tensor = AttributeProto_AttributeType_TENSOR,
        graph = AttributeProto_AttributeType_GRAPH,
        float_point_array = AttributeProto_AttributeType_FLOATS,
        integer_array = AttributeProto_AttributeType_INTS,
        string_array = AttributeProto_AttributeType_STRINGS,
        tensor_array = AttributeProto_AttributeType_TENSORS,
        graph_array = AttributeProto_AttributeType_GRAPHS
    };
    std::string get_name() const { return name; }
    Type get_type() const { return static_cast<Type>(type); }
    float get_float() const { return attribute_proto.f(); }
    int64_t get_int() const { return attribute_proto.i(); }
    const std::string& get_string() const { return attribute_proto.s(); }

    std::vector<float> get_float_array() const {
        return {std::begin(attribute_proto.floats()),
                std::end(attribute_proto.floats())};
    }
    std::vector<int64_t> get_integer_array() {
        return {std::begin(attribute_proto.ints()),
                std::end(attribute_proto.ints())};
    }
    std::vector<std::string> get_string_array() const {
        return {std::begin(attribute_proto.strings()),
                std::end(attribute_proto.strings())};
    }
};

} //namespace metawarenn

#endif //METAWARENN_ATTRIBUTE_H_
