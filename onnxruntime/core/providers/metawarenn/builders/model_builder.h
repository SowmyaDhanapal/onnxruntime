#include <core/graph/graph_viewer.h>
#include "core/providers/metawarenn/model.h"


namespace onnxruntime {
namespace metawarenn{

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(const GraphViewer& graph_viewer);
  ~ModelBuilder() = default;

  Status Compile(std::unique_ptr<Model>& model) ORT_MUST_USE_RESULT;
  Status AddOperation(int op) ORT_MUST_USE_RESULT;  

 private:
  const MetaWareNN* metawarenn_{nullptr};
  const GraphViewer& graph_viewer_;
  std::unique_ptr<Model> metawarenn_model_;
  std::unordered_map<std::string, std::shared_ptr<IOpBuilder>> op_builders_;


  Status Prepare() ORT_MUST_USE_RESULT;

  Status AddOperations() ORT_MUST_USE_RESULT;

  IOpBuilder* GetOpBuilder(const Node& node);

};

} // namespace metaware
} //namespace onnxruntime
