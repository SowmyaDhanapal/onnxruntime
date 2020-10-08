#ifndef ONNXRUNTIME_METAWARENN_IMPLEMENTATION_H_
#define ONNXRUNTIME_METAWARENN_IMPLEMENTATION_H_

#include "core/providers/metawarenn/metawarenn_lib/NeuralNetworksWrapper.h"

namespace onnxruntime {
namespace metawarenn {

class Model {
  friend class ModelBuilder;

 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;  

  const std::vector<std::string>& GetInputs() const;
  const std::vector<std::string>& GetOutputs() const; 

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

 private:
  const MetaWareNN* metawarenn_{nullptr};

  MetaWareNNModel* model_{nullptr};
  MetaWareNNCompilation* compilation_{nullptr};
  Model();

};

} // namespace metawarenn
} // namespace onnxruntime

#endif //ONNXRUNTIME_METAWARENN_IMPLEMENTATION_H_
