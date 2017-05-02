#pragma once
#ifndef INCLUDE_LAYERS_H
#define INCLUDE_LAYERS_H

#include <vector>
#include <memory>

namespace nn {

/* -------------------------------------------- */
/* ---------------- interfaces ---------------- */
/* -------------------------------------------- */

class Layer {
public:
  Layer(int num_inputs, int num_outputs = 0);
  Layer(Layer* prev, int num_outputs = 0);
  Layer(const Layer& obj) = delete;
  Layer(Layer&& obj) = delete;
  Layer& operator=(const Layer& obj) = delete;
  Layer& operator=(Layer&& obj) = delete;

  virtual void forward() = 0;
  virtual void backward() = 0;
  void setFwdInputs(double* inputs);
  void setBwdInputs(double* inputs);

  std::vector<double> getFwdInputs() const;
  std::vector<double> getFwdOutputs() const;
  std::vector<double> getBwdInputs() const;
  std::vector<double> getBwdOutputs() const;
  int getNumInputs() const;
  int getNumOutputs() const;

  virtual ~Layer();

public:
  Layer* prev_ = nullptr;
  Layer* next_ = nullptr;

protected:
  double* fwd_in_ = nullptr;
  double* fwd_out_ = nullptr;

  double* bwd_in_ = nullptr;
  double* bwd_out_ = nullptr;
  double* bwd_out_prev_ = nullptr;

  int num_inputs_ = 0;
  int num_outputs_ = 0;
};

class WeightedLayer : public Layer {
public:
  WeightedLayer(int num_inputs, 
                int num_outputs, 
                int ker_width = 0, 
                int ker_height = 0);
  WeightedLayer(Layer* prev, 
                int num_outputs, 
                int ker_width = 0, 
                int ker_height = 0);

  virtual void forward() = 0;
  virtual void backward() = 0;
  virtual void add_momentum(double alpha, double eta);
  virtual void fix_weights(double alpha, double eta);
  std::vector<double> getBiases() const;
  std::vector<std::vector<double>> getWeights() const;

  double* getBiasesInternal() const;
  double** getWeightsInternal() const;

  virtual ~WeightedLayer();

protected:
  double* biases_ = nullptr;
  double** weights_ = nullptr;

  int ker_width_ = 0;
  int ker_height_ = 0;
};

class NeuronLayer : public Layer {
public:
  NeuronLayer(Layer* prev);

  virtual void forward() = 0;
  virtual void backward() = 0;

  virtual ~NeuronLayer();
};

/* -------------------------------------------- */
/* -------------- specializations ------------- */
/* -------------------------------------------- */

/* -------------- WeightedLayer --------------- */

class FullyConnectedLayer : public WeightedLayer {
public:
  FullyConnectedLayer(int num_inputs, int num_outputs);
  FullyConnectedLayer(Layer* prev, int num_outputs);

  virtual void forward() override;
  virtual void backward() override;

  virtual ~FullyConnectedLayer();
}; 

/* --------------- NeuronLayer ---------------- */

class SigmoidLayer : public NeuronLayer {
public:
  SigmoidLayer(Layer* prev);

  virtual void forward() override;
  virtual void backward() override;

  virtual ~SigmoidLayer();
};

class ReluLayer : public NeuronLayer {
public:
  ReluLayer(Layer* prev);

  virtual void forward() override;
  virtual void backward() override;

  virtual ~ReluLayer();
};

class TanhLayer : public NeuronLayer {
public:
  TanhLayer(Layer* prev);

  virtual void forward() override;
  virtual void backward() override;

  virtual ~TanhLayer();
};

// IT IS BUGGY AF
//class SoftmaxLayer : public NeuronLayer {
//public:
//  SoftmaxLayer(Layer* prev);
//
//  virtual void forward() override;
//  virtual void backward() override;
//
//  virtual ~SoftmaxLayer();
//};

} // namespace nn

#endif // INCLUDE_LAYERS_H