#ifndef METAWARENN_MODEL_H_
#define METAWARENN_MODEL_H_

//ONNXRuntime
#ifdef ONNX_ML
#include "onnx/onnx-ml.pb.h"
#else
#include "onnx/onnx.pb.h"
#endif

//TFLite
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <map>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"

using namespace ONNX_NAMESPACE;

namespace metawarenn {

class MWNNModel {
  public:
    MWNNModel() = default;
    MWNNModel(ModelProto& onnx_model_proto);
  private:
    ModelProto model_proto;
};

} //namespace metawarenn

#endif //METAWARENN_MODEL_H_
