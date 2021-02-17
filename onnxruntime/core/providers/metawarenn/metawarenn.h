#ifndef METAWARENN_H_
#define METAWARENN_H_

#include <memory>
#include <iostream>

#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/optimizer/metawarenn_optimizer.h"
#include "metawarenn_lib/optimizer/remove_reshape.h"
#include "metawarenn_lib/metawarenn_utils.h"

#define CHW_TO_HWC 1
namespace metawarenn {

  void import_onnx_model(std::istream& stream, metawarenn::MWNNGraph& mwnn_graph);

} //namespace metawarenn

namespace InferenceEngine {

  class CNNNetwork {
  public:
    CNNNetwork();
  };

//Update for MetaWare specific types
class Precision {
public:
    /** Enum to specify of different  */
    enum ePrecision : uint8_t {
        FP32 = 0,         /**< 32bit floating point value */
    };

    Precision();
    Precision(const Precision::ePrecision value);
  };

class InferRequest {
public:
  InferRequest();
};

class ExecutableNetwork {
public:
  ExecutableNetwork();
  ExecutableNetwork LoadNetwork(/*const CNNNetwork& network*/);
  InferRequest CreateInferRequestPtr();
};
} //namespace InferenceEngine

#endif //METAWARENN_H_
