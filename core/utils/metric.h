#pragma once
#ifndef INCLUDE_METRIC_H
#define INCLUDE_METRIC_H

#include <vector>
#include <functional>

namespace utils {

using metric = std::function<double(
  const std::vector<double>&, 
  const std::vector<double>&
)>;

double L2Dist(
  const std::vector<double>& a,
  const std::vector<double>& b);

double L2Dist(
  double* a,
  double* b,
  int size);

double ChebyshovDist(
  const std::vector<double>& a,
  const std::vector<double>& b);

double ChebyshovDist(
  double* a,
  double* b,
  int size);

} // namespace utils

#endif // INCLUDE_METRIC_H
