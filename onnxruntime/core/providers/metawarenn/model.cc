#include <core/common/logging/logging.h>

#include "model.h"
#include "core/providers/metawarenn/metawarenn_lib/metawarenn_implementation.h"

namespace onnxruntime {
namespace metawarenn {
    

Model::Model() : metawarenn_(MetaWareNNImplementation()) {}

Model::~Model() {}

const std::vector<std::string>& Model::GetInputs() const {
  return input_names_;
}

const std::vector<std::string>& Model::GetOutputs() const {
  return output_names_;
}

} // namespace metawarenn
} // namespace onnxruntime
