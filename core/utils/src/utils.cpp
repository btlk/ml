#include <utils/utils.h>

#include <cmath>
#include <algorithm>

using namespace utils;

std::vector<double> ElementSum(
  const std::vector<double>& a,
  const std::vector<double>& b) {
  int size = a.size();
  if (size != b.size()) {
    throw std::runtime_error("Vector sizes dont match");
  }
  auto ans(a);
  for (int i = 0; i < size; ++i) {
    ans[i] += b[i];
  }
  return ans;
}