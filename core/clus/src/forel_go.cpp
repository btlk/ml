#include <iostream>

#include <utils/data.h>

#include <clus/forel.h>
#include <tclap/CmdLine.h>

using namespace clus;
using namespace utils;
using namespace std;

int main(int argc, char** argv) {
  string train_data_path = "";
  double radius = 1.0;
  int input_size = 4;
  string output_path = "";

  try {
    TCLAP::CmdLine cmd("NN runner app");

    TCLAP::ValueArg<std::string> trDataPathArg(
      "d", "train-path", "Path to train data csv", 
      true, "", "string", cmd);
    TCLAP::ValueArg<std::string> radArg(
      "r", "radius", "Cluster radius",
      true, "", "double", cmd);
    TCLAP::ValueArg<std::string> inputSizeArg(
      "i", "input-size", "Input size",
      true, "", "int", cmd);
    TCLAP::ValueArg<std::string> outputPathArg(
      "o", "output-size", "Output path",
      true, "", "double", cmd);

    cmd.parse(argc, argv);

    train_data_path = trDataPathArg.getValue();
    radius = stod(radArg.getValue());
    input_size = atoi(inputSizeArg.getValue().c_str());
    output_path = outputPathArg.getValue();
  } catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }

  vector<vector<double>> data; 
  vector<double> labels;
  ParseDataCSV(train_data_path, data, labels);

  ForElClusterizer clusterizer(radius);
  auto clusterization = clusterizer.fit(data);

  DumpDataToCSV(data, clusterization, output_path);

  return 0;
}
