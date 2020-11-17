#include "metawarenn_model.h"

namespace metawarenn {

MWNNModel::MWNNModel(ModelProto& onnx_model_proto) {
    model_proto = onnx_model_proto;
    std::cout << "\n In MetawareNN Model Constructor";
    for (auto& id : model_proto.opset_import()) {
        std::cout << "\nid.domain() << " << id.domain() << "id.version() << " << id.version();
}
}
MWNNModel::~MWNNModel() {
    //std::cout << "\nIn MetawareNN Model Destructor";
}

} //namespace metawarenn
