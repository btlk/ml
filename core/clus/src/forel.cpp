#include <clus/forel.h>
#include <utils/utils.h>

using namespace clus;
using namespace utils;

ForElClusterizer::ForElClusterizer(double rad)
  : radius_(rad) {
  // empty body
}

std::vector<double> ForElClusterizer::fit(
  const std::vector<std::vector<double>>& data) {
  int size = data.size();
  int dim = data[0].size();
  int cur_clus = 1;
  std::vector<double> masks(size, 0.0);
  while (true) {
    int cur_ind = std::rand() % size;
    for (int i = (cur_ind + 1) % size; i != cur_ind; i = (i + 1) % size) {
      if (masks[i] == 0.0) {
        cur_ind = i;
        break;
      }
    }
    if (masks[cur_ind] != 0.0) {
      break;
    }
    auto cur_point = data[cur_ind];
    auto center(cur_point);
    do {
      cur_point = center;
      center.assign(dim, 0.0);
      int count = 0;
      for (int i = 0; i < size; ++i) {
        if (masks[i] == 0.0 || masks[i] == cur_clus) {
          if (ChebyshovDist(cur_point, data[i]) <= radius_) {
            for (int j = 0; j < dim; ++j) {
              center[j] += data[i][j];
            }
            masks[i] = cur_clus;
            ++count;
          } else {
            masks[i] = 0.0;
          }
        } 
      }
      for (int i = 0; i < dim; ++i) {
        center[i] /= count;
      }
    } while (ChebyshovDist(center, cur_point) > 1e-8);
    ++cur_clus;
  }

  return masks;
}

ForElClusterizer::~ForElClusterizer() {
  // empty body
}