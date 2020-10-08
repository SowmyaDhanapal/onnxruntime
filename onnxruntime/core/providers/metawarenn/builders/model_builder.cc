//#include "core/providers/metawarenn/metawarenn_lib/metawarenn_implementation.h" //To access MetawareNN APIs from shared library
#include <core/common/logging/logging.h>
#include "model_builder.h"
#include "op_builder.h"
#include "helper.h"

namespace onnxruntime {
namespace metawarenn{

ModelBuilder::ModelBuilder(const GraphViewer& graph_viewer)
    : metawarenn_(MetaWareNNImplementation()), graph_viewer_(graph_viewer) {
  op_builders_ = CreateOpBuilders();
} 

Status ModelBuilder::Prepare() {
  std::cout<<"\nPrepare!!"<<std::endl;
  metawarenn_model_ = std::unique_ptr<Model>(new Model());
  /* Create and Populate the metawarenn_model_ by adding ops and operands using MetaWareNN API */
  ORT_RETURN_IF_ERROR(AddOperations());
  return Status::OK();
}

IOpBuilder* ModelBuilder::GetOpBuilder(const Node& node) {
  //std::cout<<"\nGetOpBuilder!!"<<std::endl;
  if (!Contains(op_builders_, node.OpType()))
    return nullptr;

  return op_builders_[node.OpType()].get();
}

Status ModelBuilder::AddOperations() {
  std::cout<<"\nAddOperations!!\n"<<std::endl;
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (auto* op_builder = GetOpBuilder(*node)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(*this, *node));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Node [", node->Name(), "], type [", node->OpType(), "] is not supported");
    }
  }

  return Status::OK();
}

Status ModelBuilder::AddOperation(int op) {
  if(op <0 && op>5){
    std::cout<<"Invalid op";
  }

  //std::cout<<"\nAddOperation for OP code - "<<op<<std::endl;
  return Status::OK();
}

Status ModelBuilder::Compile(std::unique_ptr<Model>& model) {
  std::cout<<"\nModelBuilder's Compile"<<std::endl;
  ORT_RETURN_IF_ERROR(Prepare());
  /* Impl for compilation of the model */
  model.reset(metawarenn_model_.release());
  return Status::OK();
}

} // namespace metawarenn
} // namespace onnxruntime
