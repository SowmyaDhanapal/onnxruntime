#pragma once

#include "metawarenn_optimizer.h"

namespace metawarenn {

namespace optimizer {

class RemoveReshape : public MWNNOptimizer {
  public:
    RemoveReshape();
    RemoveReshape(MWNNGraph* mwnn_graph, MWNNNode mwnn_node);
    void RunPass();
  private:
    MWNNGraph *graph;
    MWNNNode node;
    std::set<std::string> producers;
    std::set<std::string> consumers;
};

} //namespace optimizer

} //namespace metawarenn
