#include <nn/nn.h>

using namespace nn;

std::vector<double> nn::feedSingle(
  std::vector<double>& data,
  Layer* input_layer) {
  if (input_layer == nullptr) {
    throw std::runtime_error("Input layer must be initialized");
  }

  int layer_input_size = input_layer->getNumInputs();
  if (layer_input_size != (int)data.size()) {
    throw std::runtime_error("Data size must be equal to layer's input size");
  }

  Layer* cur_layer = input_layer;
  cur_layer->setFwdInputs(data.data());

  while (true) {
    cur_layer->forward();
    if (cur_layer->next_ != nullptr) {
      cur_layer = cur_layer->next_;
    } else {
      return cur_layer->getFwdOutputs();
    }
  }
}

std::vector<std::vector<double>> nn::feedMultiple(
  std::vector<std::vector<double>>& data,
  Layer* input_layer) {
  std::vector<std::vector<double>> ans(data.size());
  for (int i = 0; i < (int)data.size(); ++i) {
    ans[i] = feedSingle(data[i], input_layer);
  }

  return ans;
}
