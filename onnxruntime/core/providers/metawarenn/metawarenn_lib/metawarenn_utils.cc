#include "metawarenn_utils.h"

namespace metawarenn {

void fill_mwnn_tensor_initalizer(std::string input_name, MWNNGraph mwnn_graph, mli_tensor *mwnn_initalizer)
{
  std::cout << "\n\nInitializer name: " << input_name;
  mwnn_initalizer->el_type = MLI_EL_FX_16;
  auto weight = mwnn_graph.get_initializer_tensor(input_name);
  auto dims = weight.get_dims();
  mwnn_initalizer->rank = dims.size();
  std::copy(dims.begin(), dims.end(), mwnn_initalizer->shape);
  auto tensor = weight.get_tensor();
  auto abs_max = std::abs(*std::max_element(tensor.begin(), tensor.end()));
  auto abs_min = std::abs(*std::min_element(tensor.begin(), tensor.end()));
  auto max = std::max(abs_max, abs_min);
  mwnn_initalizer->el_params.fx.frac_bits = mwnn_initalizer->el_type - (int)ceil(log2(max)) - 1;
  int wt_buf_size = 1;
  uint8_t i;
  std::cout << "\nDimension size: ";
  for (i = 0; i < dims.size(); i++)
  {
    mwnn_initalizer->mem_stride[i] = 0;
    std::cout << dims[i] << ", ";
    wt_buf_size = wt_buf_size * dims[i];
  }
  int16_t *buffer = (int16_t*)malloc(wt_buf_size * sizeof(int16_t));
  int j = 0;
  for(std::vector<float>::iterator it = tensor.begin(); it != tensor.end(); ++it)
  {
    buffer[j++] = (int16_t)(*it * (1 << (mwnn_initalizer->el_params.fx.frac_bits)) + ((*it >= 0)? 0.5f: -0.5f));
  }
  mwnn_initalizer->data.capacity = sizeof(buffer);
  mwnn_initalizer->data.mem.void_p = (void *)buffer;

  std::cout << "\nMax of tensor: " << max;
  std::cout << "\nInt bits : " << (int)ceil(log2(max));
  std::cout << "\nFractional bits : " << (int)mwnn_initalizer->el_params.fx.frac_bits;
  std::cout << "\nInitializer element type : " << mwnn_initalizer->el_type;
  std::cout << "\nInitializer rank : " << mwnn_initalizer->rank;
}

void fill_mwnn_tensor_input(MWNNValueInfo input, mli_tensor *mwnn_tensor)
{
  auto dims = input.get_dims();
  mwnn_tensor->rank = dims.size();
  std::copy(dims.begin()+1, dims.end(), mwnn_tensor->shape);
  mwnn_tensor->el_type = MLI_EL_FX_16;
  mwnn_tensor->el_params.fx.frac_bits = 8;
  uint8_t i;
  for (i = 0; i < dims.size(); i++)
  {
    mwnn_tensor->mem_stride[i] = 0;
  }
  int16_t *input_buffer = (int16_t*)malloc(MAX_INPUT_BUF_SIZE * sizeof(int16_t));
  for(int i = 0; i < MAX_INPUT_BUF_SIZE; i++)
  {
    input_buffer[i++] = (int16_t)5;
  }
  mwnn_tensor->data.capacity = MAX_INPUT_BUF_SIZE * sizeof(int16_t);
  mwnn_tensor->data.mem.void_p = (void *)input_buffer;
  std::cout << "\nInput's data capacity: " << mwnn_tensor->data.capacity;
}

void create_mwnn_tensor_output(mli_tensor *mwnn_tensor)
{
  for(int dim = 0; dim < 3; dim++)
    mwnn_tensor->mem_stride[dim] = 0;
  int16_t *out_buffer = (int16_t*)malloc(MAX_OUTPUT_BUF_SIZE * sizeof(int16_t));
  mwnn_tensor->data.capacity = MAX_OUTPUT_BUF_SIZE * sizeof(int16_t);
  mwnn_tensor->data.mem.void_p = (void *)out_buffer;
  mwnn_tensor->el_params.fx.frac_bits = 8;
  std::cout << "\nOutput's data capacity: " << mwnn_tensor->data.capacity;
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
      mli_tensor output_tensor;
      std::vector<std::string> inputs = g_n.get_inputs();
      if(inputs[0] == mwnn_graph.get_graph_inputs()[0].get_name())
        {
          fill_mwnn_tensor_input(mwnn_graph.get_graph_inputs()[0], &input_tensor);
        }
      fill_mwnn_tensor_initalizer(inputs[1], mwnn_graph, &conv_wt);
      if(inputs.size() == 3)
        fill_mwnn_tensor_initalizer(inputs[2], mwnn_graph, &conv_bias);
      create_mwnn_tensor_output(&output_tensor);
      mli::krn::ref::conv2d_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_CHW, mli::CONV_GENERAL>(
        &input_tensor,
        &conv_wt,
        &conv_bias,
        &conv_cfg, &output_tensor);
      exit(1);
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
