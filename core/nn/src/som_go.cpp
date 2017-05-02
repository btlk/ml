#include <iostream>
#include <fstream>

#include <nn/nn.h>
#include <utils/data.h>
#include <tclap/CmdLine.h>

using namespace nn;
using namespace utils;
using namespace std;

void DumpSOM(
  const std::vector<std::vector<double>>& data,
  const std::string& output_path) {
  std::ofstream ofstr(output_path);
  std::stringstream ss;
  std::string cur_str;
  int size = data.size();
  int dim = data[0].size();
  for (int i = 0; i < size; ++i) {
    ss << data[i][0];
    for (int j = 1; j < dim; ++j) {
      ss << ',' << data[i][j];
    }
    ss >> cur_str;
    ss.clear();
    ofstr << cur_str << '\n';
  }
}

int main(int argc, char** argv) {
  string train_data_path = "";
  string output_path = "";
  int input_size = 4;
  int net_size = 4;

  try {
    TCLAP::CmdLine cmd("NN runner app");

    TCLAP::ValueArg<std::string> trDataPathArg(
      "d", "train-path", "Path to train data csv", 
      true, "", "string", cmd);    
    TCLAP::ValueArg<std::string> outputPathArg(
      "o", "test-path", "Path to output file",
      true, "", "string", cmd);
    TCLAP::ValueArg<std::string> inputSizeArg(
      "i", "input-size", "Input size",
      true, "", "int", cmd);
    TCLAP::ValueArg<std::string> netSizeArg(
      "n", "net-size", "Net size",
      true, "", "int", cmd);

    cmd.parse(argc, argv);

    train_data_path = trDataPathArg.getValue();
    output_path = outputPathArg.getValue();
    input_size = atoi(inputSizeArg.getValue().c_str());
    net_size = atoi(netSizeArg.getValue().c_str());
  } catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }

  vector<vector<double>> train_data; 
  vector<double> train_labels;
  ParseDataCSV(train_data_path, train_data, train_labels);

  vector<vector<double>> test_data; // by design
  vector<double> test_labels; // by design

  FullyConnectedLayer som_layer(input_size, net_size * net_size);

  double learning_rate = 0.08;
  int epochs = 100000;
  int test_frequency = -1;

  SOMOptimizer opt(&som_layer);

  opt.fit(train_data, train_labels, test_data, test_labels, epochs, test_frequency);

  auto som = opt.getMap();
  DumpSOM(som, output_path);

  return 0;
}
