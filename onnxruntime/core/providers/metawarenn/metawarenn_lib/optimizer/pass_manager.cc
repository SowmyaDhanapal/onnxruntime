#include "pass_manager.h"

namespace metawarenn {

namespace optimizer {

PassManager::PassManager() { std::cout << "\n In PassManager Constructor!!"; }
void PassManager::run_passes() {
  for (auto& pass : pass_list) {
    std::cout  << "\n PassName : " << pass->get_name();
    if(auto pass1 = std::dynamic_pointer_cast<DummyPass1>(pass)) {
      std::cout << "\n Value : " << pass1->get_value();
    }
    else if (auto pass2 = std::dynamic_pointer_cast<DummyPass2>(pass)) {
      std::cout << "\n Value : " << pass2->get_value();
    }
    else if (auto remove_reshape_pass = std::dynamic_pointer_cast<RemoveReshape>(pass)) {
      remove_reshape_pass->RunPass();
    }
    else if (auto fuse_relu_pass = std::dynamic_pointer_cast<FuseRelu>(pass)) {
      fuse_relu_pass->RunPass();
    }
    else if (auto convert_layout_pass = std::dynamic_pointer_cast<ConvertLayout>(pass)) {
      convert_layout_pass->RunPass();
    }
  }
}

} //namespace optimizer

} //namespace metawarenn
