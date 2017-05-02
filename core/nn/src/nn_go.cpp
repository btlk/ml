#include <iostream>

#include <nn/nn.h>
#include <utils/data.h>
#include <tclap/CmdLine.h>

using namespace nn;
using namespace utils;
using namespace std;

int main(int argc, char** argv) {
  string train_data_path = "";
  string test_data_path = "";
  int input_size = 4;
  int output_size = 4;

  try {
    TCLAP::CmdLine cmd("NN runner app");

    TCLAP::ValueArg<std::string> trDataPathArg(
      "d", "train-path", "Path to train data csv", 
      true, "", "string", cmd);    
    TCLAP::ValueArg<std::string> tsDataPathArg(
      "v", "test-path", "Path to test data csv",
      true, "", "string", cmd);
    TCLAP::ValueArg<std::string> inputSizeArg(
      "i", "input-size", "Input size",
      true, "", "int", cmd);
    TCLAP::ValueArg<std::string> outputSizeArg(
      "o", "output-size", "Output size",
      true, "", "int", cmd);

    cmd.parse(argc, argv);

    train_data_path = trDataPathArg.getValue();
    test_data_path = tsDataPathArg.getValue();
    input_size = atoi(inputSizeArg.getValue().c_str());
    output_size = atoi(outputSizeArg.getValue().c_str());
  } catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }

  vector<vector<double>> train_data; 
  vector<double> train_labels;
  ParseDataCSV(train_data_path, train_data, train_labels);

  vector<vector<double>> test_data;
  vector<double> test_labels;
  ParseDataCSV(test_data_path, test_data, test_labels);

  FullyConnectedLayer fc1(input_size, 25);
  SigmoidLayer fc1_neuron(&fc1);
  FullyConnectedLayer fc2(&fc1_neuron, 15);
  SigmoidLayer fc2_neuron(&fc2);
  FullyConnectedLayer fc3(&fc2_neuron, output_size);
  SigmoidLayer fc3_neuron(&fc3);
  //FullyConnectedLayer fc4(&fc3_neuron, output_size);
  //SigmoidLayer fc4_neuron(&fc4);
  //SoftmaxLayer output(&fc2_neuron);

  double learning_rate = 0.08;
  double momentum = 0.9;
  double rate_decay = 1.0 - 1e-6;
  int epochs = 100000;
  int test_frequency = 1000;

  SGDOptimizer opt(&fc1,
                   &fc3_neuron,
                   learning_rate,
                   momentum,
                   rate_decay);

  opt.fit(train_data, train_labels, test_data, test_labels, epochs, test_frequency);

  return 0;
}
