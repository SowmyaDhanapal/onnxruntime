#include "convert_layout.h"

namespace metawarenn {

namespace optimizer {

ConvertLayout::ConvertLayout() {
  set_name("ConvertLayout");
}
ConvertLayout::ConvertLayout(MWNNGraph* mwnn_graph, MWNNTensor mwnn_tensor, bool to_hwc, bool to_chw) {
  set_name("ConvertLayout");
  graph = mwnn_graph;
  tensor = mwnn_tensor;
  CHW_to_HWC = to_hwc;
  HWC_to_CHW = to_chw;
  is_tensor = true;
}
ConvertLayout::ConvertLayout(MWNNGraph* mwnn_graph, MWNNValueInfo mwnn_value_info, bool to_hwc, bool to_chw) {
  set_name("ConvertLayout");
  graph = mwnn_graph;
  value_info = mwnn_value_info;
  CHW_to_HWC = to_hwc;
  HWC_to_CHW = to_chw;
  is_value_info = true;
}
void ConvertLayout::RunPass() {
  if(is_tensor) {
    if(CHW_to_HWC) {
      std::vector<int64_t> dims = tensor.get_dims();
      std::vector<int64_t> new_dims{dims[0], dims[2], dims[3], dims[1]};
      std::vector<float> data = tensor.get_tensor();
      std::vector<float> new_data((dims[0]*dims[1]*dims[2]*dims[3]), 0);
      int channel = dims[1];
      int height = dims[2];
      int width = dims[3];
      // Data layout conversion from CHW to HWC
      for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
          for(int k = 0; k < channel; k++) {
            new_data[(i * width * channel) + (j * channel) + k] = data[(k * height * width) + (i * width) + (j)];
          }
        }
      }
      graph->update_initializer_tensors(tensor.get_name(), new_dims, new_data);
    }
    else if(HWC_to_CHW) {
      std::vector<int64_t> dims = tensor.get_dims();
      std::vector<int64_t> new_dims{dims[0], dims[3], dims[1], dims[2]};
      std::vector<float> data = tensor.get_tensor();
      std::vector<float> new_data((dims[0]*dims[1]*dims[2]*dims[3]), 0);
      int channel = dims[3];
      int height = dims[1];
      int width = dims[2];
      // Data layout conversion from HWC to CHW
      for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
          for(int k = 0; k < channel; k++) {
            new_data[(k * height * width) + (i * width) + j] = data[(i * width * channel) + (j * channel) + k];
          }
        }
      }
      graph->update_initializer_tensors(tensor.get_name(), new_dims, new_data);
    }
  }
  else if(is_value_info) {
    if(CHW_to_HWC) {
      std::vector<int64_t> dims = value_info.get_dims();
      std::vector<int64_t> new_dims{dims[0], dims[2], dims[3], dims[1]};
      graph->update_inputs(value_info.get_name(), new_dims);
    }
    else if(HWC_to_CHW) {
      std::vector<int64_t> dims = value_info.get_dims();
      std::vector<int64_t> new_dims{dims[0], dims[3], dims[1], dims[2]};
      graph->update_inputs(value_info.get_name(), new_dims);
    }
  }
  }
} //namespace optimizer

} //namespace metawarenn
