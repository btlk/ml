#include <nn/optimizer.h>
#include <utils/metric.h>

#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

using namespace nn;
using namespace utils;

/* ---------------- Optimizer ---------------- */

Optimizer::Optimizer(
  Layer* input, Layer* output) {
  if (input == nullptr) {
    throw std::runtime_error(
      "Input function must be initilized");
  }
  if (output == nullptr) {
    throw std::runtime_error(
      "Output function must be initilized");
  }

  input_ = input;
  output_ = output;
}

double Optimizer::getError(
  std::vector<std::vector<double>>& data,
  std::vector<double>& labels) {
  int err = 0;
  auto ans = feedMultiple(data, input_);
  for (int i = 0; i < (int)labels.size(); ++i) {
    double cur_ans = ans[i][0];
    double cur_ans_i = 0;
    for (int j = 1; j < (int)ans[i].size(); ++j) {
      if (ans[i][j] > cur_ans) {
        cur_ans = ans[i][j];
        cur_ans_i = j;
      }
    }
    if (cur_ans_i != (int)labels[i]) {
      ++err;
    }
  }
  return 1.0 * err / labels.size();
}

Optimizer::~Optimizer() {
  // empty body
}

/* --------------- SGDOptimizer -------------- */

SGDOptimizer::SGDOptimizer(
    Layer* input,
    Layer* output,
    double l_rate,
    double mom_rate,
    double l_rate_decay)
  : Optimizer(input, output)
  , l_rate_(l_rate)
  , mom_rate_(mom_rate)
  , l_rate_decay_(l_rate_decay) {
  // empty body
}

void SGDOptimizer::fit(
  std::vector<std::vector<double>>& train_data,
  std::vector<double>& train_labels,
  std::vector<std::vector<double>>& test_data,
  std::vector<double>& test_labels,
  int epochs,
  int test_freq) {
  unsigned seed = (unsigned)std::chrono::system_clock::now()
    .time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> uni(0, (int)train_data.size() - 1);
  std::vector<double> result;
  std::vector<double> erGrad;
  double cur_trn_err = 1.0;
  double cur_tst_err = 0.0;
  double min_trn_err = 1.0;
  double min_tst_err = 1.0;
  int small_test_freq = std::max(1, test_freq / 10);
  for (int i = 0; i < epochs; ++i) {
    if (i % small_test_freq == 0) {
      cur_trn_err = getError(train_data, train_labels);
      min_trn_err = std::min(min_trn_err, cur_trn_err);
      if (i % test_freq == 0) {
        cur_tst_err = getError(test_data, test_labels);
        min_tst_err = std::min(min_tst_err, cur_tst_err);
        std::cout << "LR " << l_rate_ << " ";
        std::cout << "Epoch " << i << " TrnE = "
                  << cur_trn_err * 100 << " % ";
        std::cout << " TstE = " << cur_tst_err * 100 
                  << " %" << std::endl;
      }
      if (cur_trn_err < 1e-4) {
        cur_tst_err = getError(test_data, test_labels);
        min_tst_err = std::min(min_tst_err, cur_tst_err);
        std::cout << "Training finished, epoch " << i
                  << " test error = " << cur_tst_err * 100 << " %" << std::endl;
        return;
      }
    }
    int pat_ind = uni(rng);
    Layer* cur_layer = output_;
    result = feedSingle(train_data[pat_ind], input_);
    erGrad = getErrorGrad(result, train_labels[pat_ind]);
    cur_layer->setBwdInputs(erGrad.data());
    while (true) {
      cur_layer->backward();
      if (cur_layer->prev_ != nullptr) {
        cur_layer = cur_layer->prev_;
      } else {
        break;
      }
    }
    cur_layer = output_;
    while (true) {
      WeightedLayer* cur_ptr = dynamic_cast<WeightedLayer*>(cur_layer);
      if (cur_ptr != nullptr) {
        cur_ptr->add_momentum(mom_rate_, l_rate_);
        cur_ptr->fix_weights(mom_rate_, l_rate_);
      }
      if (cur_layer->prev_ != nullptr) {
        cur_layer = cur_layer->prev_;
      } else {
        break;
      }
    }
    l_rate_ *= l_rate_decay_;
  }

  std::cout << "Training finished, epoch " << epochs
            << ", minimal error = " << min_trn_err * 100 << " %" << std::endl;
  std::cout << "Training finished, epoch " << epochs
            << ", minimal error = " << min_tst_err * 100 << " %" << std::endl;
}

std::vector<double> SGDOptimizer::getErrorGrad(
  std::vector<double>& signal,
  double label) {
  std::vector<double> ans(output_->getNumOutputs(), 0.0);
  for (int i = 0; i < (int)ans.size(); ++i) {
    if ((int)label == i) {
      ans[i] = signal[i] - 1;
    } else {
      ans[i] = signal[i];
    }
  }

  return ans;
}

SGDOptimizer::~SGDOptimizer() {
  // empty body
}

/* --------------- SOMOptimizer -------------- */

SOMOptimizer::SOMOptimizer(
  Layer* som_layer)
  : Optimizer(som_layer, som_layer) {
  // empty body
}

void SOMOptimizer::fit(
  std::vector<std::vector<double>>& train_data,
  std::vector<double>& train_labels,
  std::vector<std::vector<double>>& test_data,
  std::vector<double>& test_labels,
  int epochs,
  int test_freq) {
  WeightedLayer* we = dynamic_cast<WeightedLayer*>(input_);
  int som_dim = we->getNumOutputs();
  int som_side = (int)(std::sqrt(som_dim) + 1e-6);

  initializeWeights(train_data);
  double** weights = we->getWeightsInternal();
  int elem_count = (int)train_data.size();
  int elem_dim = we->getNumInputs();
  double sig0 = 1.0;
  for (int i = 0; i < epochs; ++i) {
    if (i % 2500 == 0) {
      std::cout << "Epoch " << i << '\n';
    }
    int elem_ind = std::rand() % elem_count;
    int best_i = getWinnerPosition(train_data[elem_ind], som_dim);
    double best_x = best_i / som_side;
    double best_y = best_i % som_side;
    double sig = sig0 / (1.0 + 1.0 * i / epochs);
    for (int j = 0; j < som_side; ++j) {
      for (int k = 0; k < som_side; ++k) {
        double metr = L2Dist({best_x, best_y}, {(double) j, (double) k});
        double hj = std::exp(- metr * metr / (2.0 * sig * sig));
        int cur_ind = j * som_side + k;
        for (int l = 0; l < elem_dim; ++l) {
          weights[cur_ind][l] += hj * (train_data[elem_ind][l] - weights[cur_ind][l]);
        }
      } // for k
    } // for j

  } // for i
}

SOMOptimizer::~SOMOptimizer() {
  // empty body
}

std::vector<std::vector<double>> SOMOptimizer::getMap() {
  WeightedLayer* we = dynamic_cast<WeightedLayer*>(input_);
  return we->getWeights();
}

void SOMOptimizer::initializeWeights(
  std::vector<std::vector<double>>& data) {
  int som_dim = input_->getNumOutputs();
  int elem_dim = input_->getNumInputs();
  int num_elems = (int)data.size();
  WeightedLayer* we = dynamic_cast<WeightedLayer*>(input_);
  double** weights = we->getWeightsInternal();
  for (int i = 0; i < som_dim; ++i) {
    int elem_ind = std::rand() % num_elems;
    for (int k = 0; k < elem_dim; ++k) {
      weights[i][k] = data[elem_ind][k];
    }
  }
}

int SOMOptimizer::getWinnerPosition(
  std::vector<double>& elem,
  int som_size) {
  WeightedLayer* we = dynamic_cast<WeightedLayer*>(input_);
  auto weights = we->getWeightsInternal();
  int elem_dim = elem.size();
  double best_dist = L2Dist(elem.data(), weights[0], elem_dim);
  int best_ind = 0;
  for (int i = 1; i < som_size; ++i) {
    double cur_dist = L2Dist(elem.data(), weights[i], elem_dim);
    if (cur_dist < best_dist) {
      best_dist = cur_dist;
      best_ind = i;
    }
  }

  return best_ind;
}