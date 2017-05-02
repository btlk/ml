#include <utils/initializer.h>

#include <chrono>
#include <random>

void utils::GaussianInitializer(
  double* arr, int n, double m, double s) {
  unsigned seed = (unsigned)std::chrono::system_clock::now()
                    .time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(m, s);
  for (int i = 0; i < n; ++i) {
    arr[i] = distribution(generator);
  }
}

void utils::ZeroInitializer(
  double* arr, int n) {
  for (int i = 0; i < n; ++i) {
    arr[i] = 0.0;
  }
}

void utils::UniformInitializer(
  double* weights, int n, double low, double high) {
  unsigned seed = (unsigned)std::chrono::system_clock::now()
                    .time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(low, high);
  for (int i = 0; i < n; ++i) {
    weights[i] = distribution(generator);
  }
}

void utils::UniformInitializer(
  int* arr, int n, int low, int high) {
  unsigned seed = (unsigned)std::chrono::system_clock::now()
                    .time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(low, high);
  for (int i = 0; i < n; ++i) {
    arr[i] = distribution(generator);
  }
}

std::vector<double> utils::GaussianDistribution(
  int n, double m, double s) {
  std::vector<double> ans(n, 0);
  GaussianInitializer(ans.data(), n, m, s);
  return ans;
}

std::vector<double> utils::UniformDistribution(
  int n, double low, double high) {
  std::vector<double> ans(n, 0);
  UniformInitializer(ans.data(), n, low, high);
  return ans;
}

std::vector<int> utils::UniformDistribution(
  int n, int low, int high) {
  std::vector<int> ans(n, 0);
  UniformInitializer(ans.data(), n, low, high);
  return ans;
}