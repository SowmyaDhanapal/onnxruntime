#include "node.h"

namespace metawarenn {

namespace op {

Node::Node() { }
Node::Node(std::string name, std::string node_type) {
  name = name;
  node_type = node_type;
  std::cout << "\n Name : " << name << " Type : " << node_type;
}

} //namespace op

} //namespace metawarenn
