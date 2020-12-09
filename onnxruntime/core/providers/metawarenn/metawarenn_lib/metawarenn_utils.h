#include "metawarenn_graph.h"
#include "mli_types.h"
#include <cmath>

namespace metawarenn {

#define MAX_INPUT_BUF_SIZE 224*224*3

void fill_mwnn_tensor_initalizer(std::string input_name, MWNNGraph mwnn_graph, mli_tensor mwnn_initalizer);
void fill_mwnn_tensor_input(std::string input_name, MWNNGraph mwnn_graph, mli_tensor mwnn_input);
void convert_to_mwnn_format(MWNNGraph mwnn_graph);

} // namespace metawarenn
