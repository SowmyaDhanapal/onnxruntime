#ifndef METAWARENN_H_
#define METAWARENN_H_

#include <memory>
#include <iostream>

#include "function.h"
#include "metawarenn_graph.h"
#include "optimizer/pass_manager.h"
#include "optimizer/metawarenn_optimizer.h"
#include "optimizer/dummy_pass_1.h"
#include "optimizer/dummy_pass_2.h"

namespace metawarenn {

  std::shared_ptr<Function> import_onnx_model(std::istream& stream);

} //namespace metawarenn

namespace InferenceEngine {

  class CNNNetwork {
  public:
    CNNNetwork(/*const std::shared_ptr<graph::Function>& network*/);
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
