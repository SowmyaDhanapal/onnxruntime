#include "metawarenn_utils.h"

namespace metawarenn {

void fill_mwnn_tensor_initalizer(std::string input_name, MWNNGraph mwnn_graph, mli_tensor mwnn_initalizer)
{
  std::cout << "\n\nInitializer name: " << input_name;
  mwnn_initalizer.el_type = MLI_EL_FX_16;
  auto weight = mwnn_graph.get_initializer_tensor(input_name);
  auto dims = weight.get_dims();
  mwnn_initalizer.rank = dims.size();
  std::copy(dims.begin(), dims.end(), mwnn_initalizer.shape);
  auto tensor = weight.get_tensor();
  auto abs_max = std::abs(*std::max_element(tensor.begin(), tensor.end()));
  auto abs_min = std::abs(*std::min_element(tensor.begin(), tensor.end()));
  auto max = std::max(abs_max, abs_min);
  mwnn_initalizer.el_params.fx.frac_bits = mwnn_initalizer.el_type - (int)ceil(log2(max)) - 1;
  int wt_buf_size = 1;
  uint8_t i;
  std::cout << "\nDimension size: ";
  for (i = 0; i < dims.size(); i++)
  {
    std::cout << dims[i] << ", ";
    wt_buf_size = wt_buf_size * dims[i];
  }
  uint16_t buffer[wt_buf_size];
  for(std::vector<float>::iterator it = tensor.begin(); it != tensor.end(); ++it)
  {
    buffer[i++] = (uint16_t)(*it * (1 << (mwnn_initalizer.el_params.fx.frac_bits)) + ((*it >= 0)? 0.5f: -0.5f));
  }
  mwnn_initalizer.data.capacity = sizeof(buffer);
  mwnn_initalizer.data.mem.void_p = (void *)buffer;

  std::cout << "\nMax of tensor: " << max;
  std::cout << "\nInt bits : " << (int)ceil(log2(max));
  std::cout << "\nFractional bits : " << (int)mwnn_initalizer.el_params.fx.frac_bits;
  std::cout << "\nInitializer element type : " << mwnn_initalizer.el_type;
  std::cout << "\nInitializer rank : " << mwnn_initalizer.rank;
}

void fill_mwnn_tensor_input(std::string input_name, MWNNGraph mwnn_graph, mli_tensor mwnn_input)
{
  std::cout << "\n\nInput name : " << input_name;
  uint16_t input_buffer[MAX_INPUT_BUF_SIZE] = {};
  mwnn_input.data.capacity = sizeof(input_buffer);
  mwnn_input.data.mem.void_p = (void *)input_buffer;
  assert(mwnn_graph.mwnn_graph_nodes[input_name]!=NULL);
  op::Node node = mwnn_graph.mwnn_graph_nodes[input_name];
  std::cout << "\nInput's data capacity: " << mwnn_input.data.capacity;
}

void convert_to_mwnn_format(MWNNGraph mwnn_graph)
{
    std::cout << "\n======================================================================================================================= \n";
  std::cout << "\n --------------------------------- Conversion to MetaWareNN High Level Graph Format -----------------------------------\n";
  for (auto g_n : mwnn_graph.get_graph_nodes()) {
    std::cout << "\n======================================================================================================================= \n";
    std::string op_type = g_n.get_op_type();
    std::cout << "\nNode name : " << g_n.get_name();
    std::cout << "\nOptype : " << op_type;
    if (op_type == "Conv")
    {
      mli_conv2d_cfg conv_cfg;
      auto strides = g_n.get_attribute_value("strides");
      auto pads = g_n.get_attribute_value("pads");
      auto dilations = g_n.get_attribute_value("dilations");
      conv_cfg.stride_height = strides[0];
      conv_cfg.stride_width = strides[1];
      conv_cfg.padding_bottom = pads[0];
      conv_cfg.padding_top = pads[1];
      conv_cfg.padding_left = pads[2];
      conv_cfg.padding_right = pads[3];
      conv_cfg.dilation_height = dilations[0];
      conv_cfg.dilation_width = dilations[1];
      conv_cfg.relu.type = MLI_RELU_NONE;
      std::cout << "\n\nConfig params:";
      std::cout << "\nstride_height : " << (int)conv_cfg.stride_height;
      std::cout << "\nstride_width : " << (int)conv_cfg.stride_width;
      std::cout << "\npadding_bottom : " << (int)conv_cfg.padding_bottom;
      std::cout << "\npadding_top : " << (int)conv_cfg.padding_top;
      std::cout << "\ndilation_height : " << (int)conv_cfg.dilation_height;
      std::cout << "\ndilation_width : " << (int)conv_cfg.dilation_width;
      mli_tensor input_tensor;
      mli_tensor conv_wt;
      mli_tensor conv_bias;
      std::vector<std::string> inputs = g_n.get_inputs();
      fill_mwnn_tensor_input(inputs[0], mwnn_graph, input_tensor);
      fill_mwnn_tensor_initalizer(inputs[1], mwnn_graph, conv_wt);
      if(inputs.size() == 3)
        fill_mwnn_tensor_initalizer(inputs[2], mwnn_graph, conv_bias);
    }
    else if (op_type =="Relu")
    {
      mli_relu_cfg relu_cfg;
      relu_cfg.type = MLI_RELU_NONE;
      std::cout << "\nrelu_cfg.type : " << relu_cfg.type;
    }
  }
}

} //namespace metawarenn
