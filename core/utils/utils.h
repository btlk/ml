#pragma once
#ifndef INCLUDE_UTILS_H
#define INCLUDE_UTILS_H

#include <utils/data.h>
#include <utils/metric.h>
#include <utils/initializer.h>

#include <vector>

namespace utils {

std::vector<double> ElementSum(
  const std::vector<double>&, 
  const std::vector<double>&);

}

#endif // INCLUDE_UTILS_H