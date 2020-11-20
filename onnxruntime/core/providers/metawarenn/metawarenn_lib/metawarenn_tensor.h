#ifndef METAWARENN_TENSOR_H_
#define METAWARENN_TENSOR_H_

#include "metawarenn_model.h"
#include "metawarenn_element.h"

namespace metawarenn {

 template <typename T, typename Container>
inline std::vector<T> get_data(const Container& container)
{
  return std::vector<T>(std::begin(container), std::end(container));
}

class MWNNTensor {
  public:
    MWNNTensor(TensorProto& onnx_tensor_proto);
    void set_tensor();
    std::string get_name() { return name; }
    int get_type() { return onnx_type; }
    std::vector<int64_t> get_dims() { return dims; }
    std::vector<float> get_tensor() { return tensor; }
  private:
    TensorProto tensor_proto;
    std::string name;
    int onnx_type;
    ElementType::element_type t_type;
    std::vector<int64_t> dims;
    std::vector<float> tensor;
};

} //namespace metawarenn

#endif //METAWARENN_TENSOR_H_
