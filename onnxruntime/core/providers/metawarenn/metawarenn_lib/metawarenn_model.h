#ifndef METAWARENN_MODEL_H_
#define METAWARENN_MODEL_H_

#ifdef ONNX_ML
#include "onnx/onnx-ml.pb.h"
#else
#include "onnx/onnx.pb.h"
#endif

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
