#include "metawarenn_graph.h"
#include "mli_types.h"
#include <cmath>
#include "kernels/include/api/mli_kernels_api.h"
#include "kernels/mli_krn_convolution_ref.h"


namespace metawarenn {

#define MAX_INPUT_BUF_SIZE 224*224*3
#define MAX_OUTPUT_BUF_SIZE 112*112*32

typedef mli_acc40_t mli_fx16_accu_t;

void fill_mwnn_tensor_initalizer(std::string input_name, MWNNGraph mwnn_graph, mli_tensor *mwnn_initalizer);
void fill_mwnn_tensor_input(MWNNValueInfo input, mli_tensor *mwnn_tensor);
void create_output_mwnn_tensor(mli_tensor *output_tensor);
void create_mwnn_tensor_output(MWNNGraph mwnn_graph);
void convert_to_mwnn_format(MWNNGraph mwnn_graph);

} // namespace metawarenn
