namespace onnxruntime {
namespace metawarenn {

class ModelBuilder;

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Add the operator to MetaWareNN model
  virtual Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node) ORT_MUST_USE_RESULT = 0;
};

// Generate a lookup table with IOpBuilder delegates
// for different onnx operators
std::unordered_map<std::string, std::shared_ptr<IOpBuilder>> CreateOpBuilders();

} //namespace metawarenn
} //namespace onnxruntime
