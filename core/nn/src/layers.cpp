#include <nn/layers.h>
#include <utils/initializer.h>

#include <exception>
#include <limits>
#include <algorithm>

using namespace nn;
using namespace utils;

/* ------------------ Layer ------------------ */

Layer::Layer(int num_inputs, int num_outputs)
  : num_inputs_(num_inputs) {
  if (num_outputs == 0) {
    num_outputs_ = num_inputs_;
  } else {
    num_outputs_ = num_outputs;
  }
  fwd_in_ = nullptr;
  fwd_out_ = new double[num_outputs_];

  bwd_in_ = nullptr;
  bwd_out_ = new double[num_outputs_];
  bwd_out_prev_ = new double[num_outputs_];
  ZeroInitializer(bwd_out_, num_outputs_);
}

Layer::Layer(Layer* prev, int num_outputs) {
  if (prev == nullptr) {
    throw std::runtime_error("Attempt to use nullptr as a previous layer");
  }

  prev_ = prev;
  prev_->next_ = this;
  num_inputs_ = prev->num_outputs_;
  if (num_outputs == 0) {
    num_outputs_ = num_inputs_;
  } else {
    num_outputs_ = num_outputs;
  }
  fwd_in_ = prev->fwd_out_;
  fwd_out_ = new double[num_outputs_];

  bwd_in_ = nullptr;
  bwd_out_ = new double[num_outputs_];
  bwd_out_prev_ = new double[num_outputs_];
  ZeroInitializer(bwd_out_, num_outputs_);
  prev_->bwd_in_ = bwd_out_;
}

void Layer::setFwdInputs(double* inputs) {
  fwd_in_ = inputs;
}

void Layer::setBwdInputs(double* inputs) {
  bwd_in_ = inputs;
}

std::vector<double> Layer::getFwdInputs() const {
  return std::vector<double>(fwd_in_, fwd_in_ + num_inputs_);
}

std::vector<double> Layer::getFwdOutputs() const {
  return std::vector<double>(fwd_out_, fwd_out_ + num_outputs_);
}

std::vector<double> Layer::getBwdInputs() const {
  return std::vector<double>(bwd_in_, bwd_in_ + next_->getNumOutputs());
}

std::vector<double> Layer::getBwdOutputs() const {
  return std::vector<double>(bwd_out_, bwd_out_ + num_outputs_);
}

int Layer::getNumInputs() const {
  return num_inputs_;
}

int Layer::getNumOutputs() const {
  return num_outputs_;
}

Layer::~Layer() {
  delete[] fwd_out_;
  delete[] bwd_out_;
}

/* -------------- WeightedLayer -------------- */

WeightedLayer::WeightedLayer(int num_inputs, 
                             int num_outputs,
                             int ker_width, 
                             int ker_height)
  : Layer(num_inputs, num_outputs) {
  if (ker_width == 0) {
    ker_width_ = num_outputs;
  } else {
    ker_width_ = ker_width;
  }
  if (ker_height == 0) {
    ker_height_ = num_inputs;
  } else {
    ker_height_ = ker_height;
  }
  weights_ = new double*[ker_width_];
  for (int i = 0; i < ker_width_; ++i) {
    weights_[i] = new double[ker_height_];
    UniformInitializer(weights_[i], ker_height_, -4, 4);
  }
  biases_ = new double[ker_width_];
  ZeroInitializer(biases_, ker_width_);
  //UniformInitializer(biases_, ker_width_, -1, 1);
}

WeightedLayer::WeightedLayer(Layer* prev,
                            int num_outputs,
                            int ker_width,
                            int ker_height)
  : Layer(prev, num_outputs) {
  if (ker_width == 0) {
    ker_width_ = num_outputs;
  } else {
    ker_width_ = ker_width;
  }
  if (ker_height == 0) {
    ker_height_ = prev->getNumOutputs();
  } else {
    ker_height_ = ker_height;
  }
  weights_ = new double*[ker_width_];
  for (int i = 0; i < ker_width_; ++i) {
    weights_[i] = new double[ker_height_];
    UniformInitializer(weights_[i], ker_height_, -4, 4);
  }
  biases_ = new double[ker_width_];
  ZeroInitializer(biases_, ker_width_);
  //UniformInitializer(biases_, ker_width_, -1, 1);
}

WeightedLayer::~WeightedLayer() {
  for (int i = 0; i < ker_width_; ++i) {
    delete[] weights_[i];
  }
  delete[] weights_;
  delete[] biases_;
}

void WeightedLayer::add_momentum(double alpha, double eta) {
  for (int i = 0; i < ker_width_; ++i) {
    biases_[i] -= alpha * 2 * eta * bwd_out_prev_[i];
    for (int j = 0; j < ker_height_; ++j) {
      weights_[i][j] -= alpha * eta * bwd_out_prev_[i];
    }
  }
}

void WeightedLayer::fix_weights(double alpha, double eta) {
  for (int i = 0; i < ker_width_; ++i) {
    biases_[i] -= (1.0 - alpha) * 2 * eta * bwd_out_[i];
    for (int j = 0; j < ker_height_; ++j) {
      weights_[i][j] -= (1.0 - alpha) * eta * bwd_out_[i];
    }
  }
}

std::vector<std::vector<double>> WeightedLayer::getWeights() const {
  std::vector<std::vector<double>> ans(ker_width_);
  for (int i = 0; i < ker_width_; ++i) {
    ans[i] = std::vector<double>(weights_[i], weights_[i] + ker_height_);
  }
  return ans;
}

std::vector<double> WeightedLayer::getBiases() const {
  return std::vector<double>(biases_, biases_ + ker_height_);
}

double** WeightedLayer::getWeightsInternal() const {
  return weights_;
}

double* WeightedLayer::getBiasesInternal() const {
  return biases_;
}

/* --------------- NeuronLayer --------------- */

NeuronLayer::NeuronLayer(Layer* prev) 
  : Layer(prev) {
  // empty body
}

NeuronLayer::~NeuronLayer() {
  // empty body
}

/* ----------- FullyConnectedLayer ----------- */

FullyConnectedLayer::FullyConnectedLayer(
  int num_inputs, int num_outputs)
  : WeightedLayer(num_inputs, num_outputs, num_outputs, num_inputs) {
  // empty body
}

FullyConnectedLayer::FullyConnectedLayer(
  Layer* prev, int num_outputs)
  : WeightedLayer(prev, num_outputs) {
  // empty body
}

void FullyConnectedLayer::forward() {
  for (int i = 0; i < num_outputs_; ++i) {
    fwd_out_[i] = biases_[i];
    for (int j = 0; j < num_inputs_; ++j) {
      fwd_out_[i] += fwd_in_[j] * weights_[i][j];
    }
  }
}

void FullyConnectedLayer::backward() {
  for (int i = 0; i < num_outputs_; ++i) {
    bwd_out_prev_[i] = bwd_out_[i];
    bwd_out_[i] = bwd_in_[i];
  }
}

FullyConnectedLayer::~FullyConnectedLayer() {
  // empty body
}

/* --------------- SigmoidLayer -------------- */

SigmoidLayer::SigmoidLayer(Layer* prev)
  : NeuronLayer(prev) {
  // empty body
}

void SigmoidLayer::forward() {
  for (int i = 0; i < num_inputs_; ++i) {
    fwd_out_[i] = 1.0 / (1.0 + std::exp(-fwd_in_[i]));
  }
}

void SigmoidLayer::backward() {
  WeightedLayer* next = dynamic_cast<WeightedLayer*>(next_);
  if (next != nullptr) {
    auto next_weights = next->getWeightsInternal();
    for (int i = 0; i < num_outputs_; ++i) {
      bwd_out_prev_[i] = bwd_out_[i];
      bwd_out_[i] = 0.0;
      for (int j = 0; j < next->getNumOutputs(); ++j) {
        bwd_out_[i] += bwd_in_[j] * next_weights[j][i];
      }
    }
  } else {
    for (int i = 0; i < num_outputs_; ++i) {
      bwd_out_[i] = bwd_in_[i];
    }
  }
  for (int i = 0; i < num_outputs_; ++i) {
    bwd_out_[i] *= (fwd_out_[i] * (1.0 - fwd_out_[i]));
  }
}

SigmoidLayer::~SigmoidLayer() {
  // empty body
}

/* ---------------- ReluLayer ---------------- */

ReluLayer::ReluLayer(Layer* prev)
  : NeuronLayer(prev) {
  // empty body
}

void ReluLayer::forward() {
  for (int i = 0; i < num_inputs_; ++i) {
    fwd_out_[i] = std::max(0.01 * fwd_in_[i], fwd_in_[i]);
  }
}

void ReluLayer::backward() {
  WeightedLayer* next = dynamic_cast<WeightedLayer*>(next_);
  if (next != nullptr) {
    auto next_weights = next->getWeightsInternal();
    for (int i = 0; i < num_outputs_; ++i) {
      bwd_out_[i] = 0.0;
      for (int j = 0; j < next->getNumInputs(); ++j) {
        bwd_out_[i] += next_weights[j][i] * bwd_in_[j];
      }
    }
  } else {
    for (int i = 0; i < num_outputs_; ++i) {
      bwd_out_[i] = bwd_in_[i];
    }
  }
  for (int i = 0; i < num_outputs_; ++i) {
    if (fwd_in_[i] < 0.0) {
      bwd_out_[i] *= 0.01;
    }
  }
}

ReluLayer::~ReluLayer() {
  // empty body
}

/* ---------------- TanhLayer ---------------- */

TanhLayer::TanhLayer(Layer* prev)
  : NeuronLayer(prev) {
  // empty body
}

void TanhLayer::forward() {
  for (int i = 0; i < num_inputs_; ++i) {
    fwd_out_[i] = std::tanh(fwd_in_[i]);
  }
}

void TanhLayer::backward() {
  WeightedLayer* next = dynamic_cast<WeightedLayer*>(next_);
  if (next != nullptr) {
    auto next_weights = next->getWeightsInternal();
    for (int i = 0; i < num_outputs_; ++i) {
      bwd_out_[i] = 0.0;
      for (int j = 0; j < next->getNumInputs(); ++j) {
        bwd_out_[i] += next_weights[j][i] * bwd_in_[j];
      }
    }
  } else {
    for (int i = 0; i < num_outputs_; ++i) {
      bwd_out_[i] = bwd_in_[i];
    }
  }
  for (int i = 0; i < num_inputs_; ++i) {
    bwd_out_[i] *= (1.0 - fwd_out_[i] * fwd_out_[i]);
  }
}

TanhLayer::~TanhLayer() {
  // empty body
}

/* --------------- SoftmaxLayer -------------- */

//SoftmaxLayer::SoftmaxLayer(Layer* prev)
//  : NeuronLayer(prev) {
//  // empty body
//}
//
//void SoftmaxLayer::forward() {
//  double in_sum = 0.0;
//  double in_max = fwd_in_[0];
//  for (int i = 1; i < num_inputs_; ++i) {
//    in_max = std::max(in_max, fwd_in_[i]);
//  }
//  for (int i = 0; i < num_inputs_; ++i) {
//    fwd_out_[i] = std::exp(fwd_in_[i] - in_max);
//    in_sum += fwd_out_[i];
//  }
//  for (int i = 0; i < num_inputs_; ++i) {
//    fwd_out_[i] /= in_sum;
//  }
//}
//
//void SoftmaxLayer::backward() {
//  for (int i = 0; i < num_outputs_; ++i) {
//    bwd_out_[i] = fwd_out_[i] * (1.0 - fwd_out_[i]) * fwd_in_[i];
//    for (int j = 0; j < i; ++j) {
//      bwd_out_[i] -= fwd_out_[i] * fwd_out_[j] * fwd_in_[j];
//    }
//    for (int j = i + 1; j < num_outputs_; ++j) {
//      bwd_out_[i] -= fwd_out_[i] * fwd_out_[j] * fwd_in_[j];
//    } 
//  }
//  for (int i = 0; i < num_outputs_; ++i) {
//    bwd_out_[i] *= bwd_in_[i];
//  }
//}
//
//SoftmaxLayer::~SoftmaxLayer() {
//  // empty body
//}
