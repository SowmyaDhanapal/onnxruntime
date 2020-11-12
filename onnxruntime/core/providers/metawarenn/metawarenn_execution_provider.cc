#include "metawarenn_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/memcpy.h"

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  Status st;
};
}  // namespace

namespace onnxruntime {

constexpr const char* MetaWareNN = "MetaWareNN";

MetaWareNNExecutionProvider::MetaWareNNExecutionProvider()
    : IExecutionProvider{onnxruntime::kMetaWareNNExecutionProvider} {
  DeviceAllocatorRegistrationInfo device_info(
      {OrtMemTypeDefault,
       [](int) {
         return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(MetaWareNN, OrtAllocatorType::OrtDeviceAllocator));
       },
       std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(device_info));

  DeviceAllocatorRegistrationInfo cpu_info{
      OrtMemTypeCPUOutput,
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(
            OrtMemoryInfo(MetaWareNN, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      },
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(cpu_info));
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

// This functions returns vector of all supported_subgraph by metawarenn
static std::vector<std::vector<NodeIndex>>
GetPartitionedSubgraphs(const std::vector<NodeIndex>& topological_order, const std::vector<NodeIndex>& unsupported_nodes_idxes) {
  std::vector<std::vector<NodeIndex>> metawarenn_subgraphs;

  auto prev = topological_order.begin();

  for (const auto& unsup_node : unsupported_nodes_idxes) {
    auto it = std::find(prev, topological_order.end(), unsup_node);
    // Create a cluster vector[supported_node_idx, unsupported_node_idx)
    // and append it to return list.
    std::vector<NodeIndex> this_subgraph{prev, it};
    if (!this_subgraph.empty()) {
      metawarenn_subgraphs.push_back(std::move(this_subgraph));
    }
    // Point prev to node idx past this unsuported node.
    prev = ++it;
  }

  // Tail
  std::vector<NodeIndex> this_subgraph{prev, topological_order.end()};
  if (!this_subgraph.empty()) {
    metawarenn_subgraphs.push_back(std::move(this_subgraph));
  }

  return metawarenn_subgraphs;
}

static void GetInputsOutputsOfSubgraph(const GraphViewer& graph_viewer,
                                       const std::vector<NodeIndex>& nodes,
                                       const std::unordered_set<std::string>& metawarenn_required_initializers,
                                       std::vector<std::string>& nodes_inputs,
                                       std::vector<std::string>& nodes_outputs) {
  std::unordered_set<std::string> input_args;
  std::vector<std::string> ordered_input_args;
  std::unordered_set<std::string> output_args;
  std::unordered_set<std::string> external_output_args;

  for (const auto& node_idx : nodes) {
    const auto& node = graph_viewer.GetNode(node_idx);
    // Collect all inputs and outputs
    node->ForEachDef(
        [&input_args, &ordered_input_args, &output_args](const NodeArg& node_arg, bool is_input) {
          if (is_input) {
            if (!input_args.count(node_arg.Name())) {
              ordered_input_args.push_back(node_arg.Name());
            }
            input_args.insert(node_arg.Name());
          } else {
            output_args.insert(node_arg.Name());
          }
        },
        true);

    // Check if output of this node is used by nodes outside
    // subgraph. If yes add this to cluster outputs
    for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
      const auto& ext_node = graph_viewer.GetNode((*it).Index());

      if (std::find(nodes.begin(), nodes.end(), ext_node->Index()) == nodes.end()) {
        // Node is external to subgraph. Search through its
        // inputs to find the output that is generated by subgraph.
        std::set<std::string> ext_node_inputs;
        ext_node->ForEachDef(
            [&ext_node_inputs](const onnxruntime::NodeArg& arg, bool is_input) {
              if (is_input) {
                ext_node_inputs.insert(arg.Name());
              }
            },
            true);

        for (const auto& out_def : node->OutputDefs()) {
          if (ext_node_inputs.find(out_def->Name()) != ext_node_inputs.end()) {
            external_output_args.insert(out_def->Name());
          }
        }
      }
    }
  }

  //Extract initializers used by subgraph.
  std::unordered_set<std::string> original_graph_inputs;
  for (const auto& node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    original_graph_inputs.insert(node_arg->Name());
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();
  std::vector<std::string> const_inputs;
  for (const auto& in_arg : ordered_input_args) {
    if ((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        metawarenn_required_initializers.count(in_arg)) {
      const_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : ordered_input_args) {
    if (!output_args.count(in_arg) &&
        !((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        metawarenn_required_initializers.count(in_arg))) {
      nodes_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : const_inputs) {
    nodes_inputs.push_back(in_arg);
  }

  std::copy(external_output_args.begin(), external_output_args.end(), std::back_inserter(nodes_outputs));
  for (const auto& node_arg : graph_viewer.GetOutputs()) {
    const auto& name = node_arg->Name();
    if (output_args.count(name) && !external_output_args.count(name)) {
      nodes_outputs.push_back(name);
    }
  }
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
  //static std::set<std::string> metawarenn_supported_ops = {"Add", "Conv", "Relu", "Reshape"};
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
  else { //unsupported_nodes_idxes is not empty
    const auto metawarenn_subgraphs = GetPartitionedSubgraphs(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes_idxes);

    for (const auto& subgraph : metawarenn_subgraphs) {
      std::vector<std::string> subgraph_inputs, subgraph_outputs;
      GetInputsOutputsOfSubgraph(graph_viewer, subgraph, metawarenn_required_initializers, subgraph_inputs, subgraph_outputs);

      if (!subgraph_inputs.empty()) {
        AppendNodesToSubGraph(subgraph, subgraph_inputs, subgraph_outputs, result);
      }
    }
  }
  return result;
}

common::Status MetaWareNNExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                                  std::vector<NodeComputeInfo>& node_compute_funcs) {
  std::cout << "\n MetaWareNNExecutionProvider::Compile() \n";
  for (const auto* fused_node : fused_nodes) {
    std::cout << "\n  Fused_node_name: " << fused_node->Name() << "\n";
    std::shared_ptr<metawarenn_ep::BackendManager> backend_manager = std::make_shared<metawarenn_ep::BackendManager>(fused_node, *GetLogger());
    NodeComputeInfo compute_info;
    //Define these funtions based on MetaWareEV FunctionState
    compute_info.create_state_func =
        [backend_manager](ComputeContext* context, FunctionState* state) {
          MetaWareNNFunctionState* p = new MetaWareNNFunctionState();
          p->allocate_func = context->allocate_func;
          p->destroy_func = context->release_func;
          p->allocator_handle = context->allocator_handle;
          p->backend_manager = backend_manager;
          *state = static_cast<FunctionState>(p);
          return 0;
        };
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      auto function_state = static_cast<MetaWareNNFunctionState*>(state);
      try {
        function_state->backend_manager->Compute(*api, context);
      } catch (const char* msg) {
        return common::Status(common::ONNXRUNTIME, common::FAIL, msg);
      }
      return Status::OK();
    };

    compute_info.release_state_func =
        [](FunctionState state) {
          if (state) {
            MetaWareNNFunctionState* function_state = static_cast<MetaWareNNFunctionState*>(state);
            delete function_state;
          }
        };

    node_compute_funcs.push_back(compute_info);
    }
  
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kMetaWareNNExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kMetaWareNNExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kMetaWareNNExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kMetaWareNNExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static Status RegisterMetaWareNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMetaWareNNExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMetaWareNNExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }
  return Status::OK();
}

KernelRegistryAndStatus GetMetaWareNNKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterMetaWareNNKernels(*ret.kernel_registry);
  return ret;
}

std::shared_ptr<KernelRegistry>
MetaWareNNExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = onnxruntime::GetMetaWareNNKernelRegistry();
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

}  // namespace onnxruntime
