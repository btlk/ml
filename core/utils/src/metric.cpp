#include <utils/metric.h>

#include <cmath>
#include <algorithm>

double utils::L2Dist(
  const std::vector<double>& a,
  const std::vector<double>& b) {
  double ans = 0.0;
  int size = a.size();
  if (size != b.size()) {
    throw std::runtime_error("Vector sizes dont match");
  }
  double tmp_var = 0.0;
  for (int i = 0; i < size; ++i) {
    tmp_var = a[i] - b[i];
    ans += tmp_var * tmp_var;
  }
  return std::sqrt(ans);
}

double utils::L2Dist(
  double* a,
  double* b,
  int size) {
  double ans = 0.0;
  double tmp_var = 0.0;
  for (int i = 0; i < size; ++i) {
    tmp_var = a[i] - b[i];
    ans += tmp_var * tmp_var;
  }
  return std::sqrt(ans);
}

double utils::ChebyshovDist(
  const std::vector<double>& a,
  const std::vector<double>& b) {
  double ans = 0.0;
  int size = a.size();
  if (size != b.size()) {
    throw std::runtime_error("Vector sizes dont match");
  }
  for (int i = 0; i < size; ++i) {
    ans = std::max(ans, std::fabs(a[i] - b[i]));
  }
  return ans;
}

double utils::ChebyshovDist(
  double* a,
  double* b,
  int size) {
  double ans = 0.0;
  for (int i = 0; i < size; ++i) {
    ans = std::max(ans, std::fabs(a[i] - b[i]));
  }
  return ans;
}