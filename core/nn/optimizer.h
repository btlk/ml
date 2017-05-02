#pragma once
#ifndef INCLUDE_OPTIMIZER_H
#define INCLUDE_OPTIMIZER_H

#include <nn/nn.h>
#include <vector>

namespace nn {

/* ------------------------------------------- */
/* ---------------- interface ---------------- */
/* ------------------------------------------- */

class Optimizer {
public:
  Optimizer(Layer* input, Layer* output);
  Optimizer(const Optimizer& obj) = delete;
  Optimizer(Optimizer&& obj) = delete;
  Optimizer& operator=(const Optimizer& obj) = delete;
  Optimizer& operator=(Optimizer&& obj) = delete;

  virtual void fit(
    std::vector<std::vector<double>>& train_data,
    std::vector<double>& train_labels,
    std::vector<std::vector<double>>& test_data,
    std::vector<double>& test_labels,
    int epochs,
    int test_freq) = 0;

  double getError(
    std::vector<std::vector<double>>& data,
    std::vector<double>& labels);

  virtual ~Optimizer();

protected:
  Layer* input_ = nullptr;
  Layer* output_ = nullptr;
};

/* ------------------------------------------- */
/* ------------- specializations ------------- */
/* ------------------------------------------- */

class SGDOptimizer : public Optimizer {
public:
  SGDOptimizer(
    Layer* input,
    Layer* output, 
    double l_rate = 0.001,
    double mom_rate = 0.9,
    double l_rate_decay = 1.0);

  virtual void fit(
    std::vector<std::vector<double>>& train_data,
    std::vector<double>& train_labels,
    std::vector<std::vector<double>>& test_data,
    std::vector<double>& test_labels,
    int epochs = 10000,
    int test_freq = 100) override;

  virtual ~SGDOptimizer() override;

protected:
  std::vector<double> getErrorGrad(
    std::vector<double>& signal,
    double label);

protected:
  double l_rate_;
  double mom_rate_;
  double l_rate_decay_;
};

class SOMOptimizer : Optimizer {
public:
  SOMOptimizer(Layer* som_layer);

  virtual void fit(
    std::vector<std::vector<double>>& train_data,
    std::vector<double>& train_labels,
    std::vector<std::vector<double>>& test_data,
    std::vector<double>& test_labels,
    int epochs,
    int test_freq);

  double getError(
    std::vector<std::vector<double>>& data,
    std::vector<double>& labels);

  virtual ~SOMOptimizer();

public:
  std::vector<std::vector<double>> getMap();

private:
  void initializeWeights(
    std::vector<std::vector<double>>& train_data);

  int getWinnerPosition(
    std::vector<double>& elem,
    int som_size);
};

} // namespace nn

#endif // INCLUDE_OPTIMIZER_H
