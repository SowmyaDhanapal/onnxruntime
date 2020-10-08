#include <core/graph/graph.h>
#include <string>

#include "core/providers/metawarenn/metawarenn_lib/NeuralNetworksTypes.h"

template <class Map, class Key>
inline bool Contains(const Map& map, const Key& key) {
  return map.find(key) != map.end();
}
