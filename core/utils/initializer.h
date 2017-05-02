#pragma once
#ifndef INCLUDE_INITIALIZER_H
#define INCLUDE_INITIALIZER_H

#include <vector>

namespace utils {

void GaussianInitializer(double* arr, int n, double m, double s);

void ZeroInitializer(double* arr, int n);

void UniformInitializer(double* arr, int n, double low, double high);

void UniformInitializer(int* arr, int n, int low, int high);

std::vector<double> GaussianDistribution(int n, double m, double s);

std::vector<double> UniformDistribution(int n, double low, double high);

std::vector<int> UniformDistribution(int n, int low, int high);

} // namespace utils

#endif // INCLUDE_INITIALIZER_H