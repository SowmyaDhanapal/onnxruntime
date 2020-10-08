#include "metawarenn_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {

MetaWareNNExecutionProvider::MetaWareNNExecutionProvider()
    : IExecutionProvider{onnxruntime::kMetaWareNNExecutionProvider} {
}
MetaWareNNExecutionProvider::~MetaWareNNExecutionProvider() {}

static void AppendNodesToSubGraph(const std::vector<NodeIndex>& nodes,
                                    const std::vector<std::string>& inputs,
                                    const std::vector<std::string>& outputs,
                                    std::vector<std::unique_ptr<ComputeCapability>>& result) {
  static size_t op_counter = 0;

  auto meta_def = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->name = "MetaWareNN_" + std::to_string(++op_counter);
  meta_def->domain = kMetaWareNNDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;

  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  sub_graph->nodes = nodes;
  sub_graph->SetMetaDef(meta_def);
  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  std::cout << "\n  --> MetaWareNNExecutionProvider Created SubGraph \n ";
}

std::vector<std::unique_ptr<ComputeCapability>>
MetaWareNNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                         const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Construct subgraph capability from node list
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::cout << "\n MetaWareNNExecutionProvider::GetCapability() \n";
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  static std::set<std::string> metawarenn_supported_ops = {"Add", "Conv", "GlobalAveragePool", "Relu", "Reshape"};
  std::vector<NodeIndex> unsupported_nodes_idxes;

  // This is a list of initializers that MetaWareNNGraph considers as constants. 
  // Example weights, reshape shape etc.
  std::unordered_set<std::string> metawarenn_required_initializers;
  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto& node = graph_viewer.GetNode(node_idx);
    const auto& optype = node->OpType();
    if (metawarenn_supported_ops.count(optype) == 0)
    {
      unsupported_nodes_idxes.push_back(node_idx);
      std::cout << "\nUnsupported node_idx: " << node_idx << "\tnode_name: " << node->Name() << " --- " << node->OpType();
    }
    else
    {
         // Collect inputs that are initializers
      graph_viewer.GetNode(node_idx)->ForEachDef([&metawarenn_required_initializers, &graph_viewer](const onnxruntime::NodeArg& node_arg, bool is_input) {
              if(is_input && graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                metawarenn_required_initializers.insert(node_arg.Name());
              } }, true);
    }
  }
  //TODO - Handle more than 1 subgraph - Now considered all nodes are supported for MobileNet-V1
  if (unsupported_nodes_idxes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    //Fill inputs with names
    std::for_each(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                  [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

    // Initializers need to be part of meta_def->inputs
    std::for_each(metawarenn_required_initializers.begin(), metawarenn_required_initializers.end(),
                  [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

    // Fill outputs with names
    std::for_each(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(),
                  [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

    // Create and add this graph to result.
    AppendNodesToSubGraph(graph_viewer.GetNodesInTopologicalOrder(), inputs, outputs, result);
  }
  return result;
}

common::Status MetaWareNNExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                                  std::vector<NodeComputeInfo>& node_compute_funcs) {
  std::cout << "\n MetaWareNNExecutionProvider::Compile() \n";
  for (const auto* fused_node : fused_nodes) {
    std::cout << "\n  Fused_node_name: " << fused_node->Name() << "\n";
    std::shared_ptr<metawarenn::BackendManager> backend_manager = std::make_shared<metawarenn::BackendManager>(fused_node, *GetLogger());
  }
  NodeComputeInfo compute_info;
  node_compute_funcs.push_back(compute_info);

  //Define these funtions based on MetaWareNN FunctionState
  compute_info.create_state_func = nullptr;//[](){};
  compute_info.compute_func = nullptr;//[](){};
  compute_info.release_state_func = nullptr;//[](){};
  
  return Status::OK();
}
}  // namespace onnxruntime
