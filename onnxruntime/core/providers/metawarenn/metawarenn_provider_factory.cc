#include "core/providers/metawarenn/metawarenn_provider_factory.h"
#include "metawarenn_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct MetaWareNNProviderFactory : IExecutionProviderFactory {
  MetaWareNNProviderFactory() {}
  ~MetaWareNNProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> MetaWareNNProviderFactory::CreateProvider() {
  std::cout << "\n MetaWareNNProviderFactory - CreateProvider\n";
  return onnxruntime::make_unique<MetaWareNNExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MetaWareNN() {
  std::cout << "\n CreateExecutionProviderFactory_MetaWareNN\n";
  return std::make_shared<onnxruntime::MetaWareNNProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MetaWareNN, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_MetaWareNN());
  std::cout << "\n Added CreateExecutionProviderFactory_MetaWareNN to provider_factories\n";
  return nullptr;
}
