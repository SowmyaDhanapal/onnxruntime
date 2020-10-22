#include <memory>
#include <iostream>

#include "function.h"

namespace graph {

namespace onnx {

  std::shared_ptr<graph::Function> import_onnx_model(std::istream& stream);

} //namespace onnx

} //namespace graph

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
