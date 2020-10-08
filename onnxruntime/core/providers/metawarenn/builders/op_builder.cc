#include "helper.h"
#include "model_builder.h"
#include "op_builder.h"

namespace onnxruntime {
namespace metawarenn {


class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node) override final ORT_MUST_USE_RESULT;

 protected:
  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) ORT_MUST_USE_RESULT = 0;
};    

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node) {
  //std::cout<<"\nAddToModelBuilder!!!"<<std::endl;
  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node));
  return Status::OK();
}

class ConvOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

Status ConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  //std::cout<<"\nConvOpBuilder's AddToModelBuilderImpl!!!"<<std::endl;
  const auto& op_type(node.OpType());
  int32_t op_code;  
  if (op_type == "Conv") {
    op_code = METAWARENN_CONV_2D;
  }
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code));
  return Status::OK();
} 


class PoolOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

Status PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  //std::cout<<"\nPoolOpBuilder's AddToModelBuilderImpl!!!"<<std::endl;
  const auto& op_type(node.OpType());
  int32_t op_code;  
  if (op_type == "GlobalAveragePool") {
    op_code = METAWARENN_GLOBAL_AVERAGE_POOL_2D;
  }
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code));
  return Status::OK();
}

class BinaryOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  //std::cout<<"\nBinaryOpBuilder's AddToModelBuilderImpl!!!"<<std::endl;
  const auto& op_type(node.OpType());
  int32_t op_code;  
  if (op_type == "Add") {
    op_code = METAWARENN_ADD;
  }
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code));
  return Status::OK();
}

class ReluOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

Status ReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  //std::cout<<"\nReluOpBuilder's AddToModelBuilderImpl!!!"<<std::endl;
  const auto& op_type(node.OpType());
  int32_t op_code;  
  if (op_type == "Relu") {
    op_code = METAWARENN_RELU;
  }
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code));
  return Status::OK();
}

class ReshapeOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  //std::cout<<"\nReshapeOpBuilder's AddToModelBuilderImpl!!!"<<std::endl;
  const auto& op_type(node.OpType());
  int32_t op_code;  
  if (op_type == "Reshape") {
    op_code = METAWARENN_RESHAPE;
  }
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code));
  return Status::OK();
}


std::unordered_map<std::string, std::shared_ptr<IOpBuilder>>
CreateOpBuilders() {
  std::cout<<"\nCreateOpBuilders!!!"<<std::endl;
  std::unordered_map<std::string, std::shared_ptr<IOpBuilder>> op_map;

  op_map.emplace("Conv", std::make_shared<ConvOpBuilder>());

  auto pool_op_builder = std::make_shared<PoolOpBuilder>();
  op_map.emplace("GlobalAveragePool", pool_op_builder);

  auto binary_op_builder = std::make_shared<BinaryOpBuilder>();
  op_map.emplace("Add", binary_op_builder);

  op_map.emplace("Relu", std::make_shared<ReluOpBuilder>());
  op_map.emplace("Reshape", std::make_shared<ReshapeOpBuilder>());  

  std::cout<<"\nOp map Created...\n";
  return op_map;
}

}  // namespace metawarenn
}  // namespace onnxruntime
