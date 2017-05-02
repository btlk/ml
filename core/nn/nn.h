#pragma once
#ifndef INCLUDE_NN_H
#define INCLUDE_NN_H

#include <nn/layers.h>
#include <nn/optimizer.h>

#include <vector>

namespace nn {

std::vector<double> feedSingle(
  std::vector<double>& data,
  Layer* input_layer);

std::vector<std::vector<double>> feedMultiple(
  std::vector<std::vector<double>>& data,
  Layer* input_layer);

} // namespace nn

#endif // INCLUDE_NN_H