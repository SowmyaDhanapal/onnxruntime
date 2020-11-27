#pragma once

#include <iostream>
#include <vector>

namespace metawarenn {

namespace op {

class Node {
  public:
    Node();
    Node(std::string name, std::string node_type);
    std::string name;
    std::string node_type;
};
} //namespace op

} //namespace metawarenn
