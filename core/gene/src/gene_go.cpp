#include <iostream>
#include <cmath>

#include <gene/gene.h>
#include <tclap/CmdLine.h>

using namespace gene;
using namespace std;

const double PI = 3.14159265;

double foo_rast(const vector<double>& v) {
  int n = (int)v.size();
  double val = 10 * n;
  for (int i = 0; i < n; ++i) {
    val += v[i] * v[i] - 10 * std::cos(2.0 * PI * v[i]);
  }
  return val;
}

std::vector<double> foo_rast_deriv(const std::vector<double>& v) {
  int n = (int)v.size();
  std::vector<double> ans(n, 0.0);
  for (int i = 0; i < n; ++i) {
    ans[i] = 2.0 * (v[i] + PI * 10 * std::sin(2.0 * PI * v[i]));
  }
  return ans;
}

double foo_booth(const vector<double>& v) {
  int n = (int)v.size();
  double val = (v[0] + 2 * v[1] - 7) * (v[0] + 2 * v[1] - 7) 
    + (2 * v[0] - v[1] - 5) * (2 * v[0] - v[1] - 5);
  return val;
}

std::vector<double> foo_booth_deriv(const std::vector<double>& v) {
  int n = (int)v.size();
  std::vector<double> ans(n, 0.0);
  ans[0] = 2 * (v[0] + 2 * v[1] - 7) + 4 * (2 * v[0] + v[1] - 5);
  ans[1] = 4 * (v[0] + 2 * v[1] - 7) + 2 * (2 * v[0] + v[1] - 5);
  return ans;
}

double foo_sphere(const vector<double>& v) {
  int n = (int)v.size();
  double val = 0.0;
  for (int i = 0; i < n; ++i) {
    val += v[0] * v[0];
  }
  return val;
}

std::vector<double> foo_sphere_deriv(const std::vector<double>& v) {
  int n = (int)v.size();
  std::vector<double> ans(n, 0.0);
  for (int i = 0; i < n; ++i) {
    ans[i] = 2.0 * v[i];
  }
  return ans;
}

double foo_beale(const vector<double>& v) {
  int n = (int)v.size();
  double val = (1.5 - v[0] + v[0] * v[1]) * (1.5 - v[0] + v[0] * v[1])
    + (2.25 - v[0] +v[0] * v[1] * v[1]) * (2.25 - v[0] + v[0] * v[1] * v[1])
    + (2.625 - v[0] + v[0] * v[1] * v[1] * v[1])
    * (2.625 - v[0] + v[0] * v[1] * v[1] * v[1]);
  return val;
}

std::vector<double> foo_beale_deriv(const std::vector<double>& v) {
  int n = (int)v.size();
  std::vector<double> ans(n, 0.0);
  ans[0] = 2 * (1.5 - v[0] + v[0] * v[1]) * (v[1] - 1)
    + 2 * (2.25 - v[0] + v[0] * v[1] * v[1]) * (v[1] * v[1] - 1)
    + 2 * (2.625 - v[0] + v[0] * v[1] * v[1] * v[1]) * (v[1] * v[1] * v[1] - 1);
  ans[1] = 2 * (1.5 - v[0] + v[0] * v[1]) * v[0]
    + 2 * (2.25 - v[0] + v[0] * v[1] * v[1]) * 2 * v[0] * v[1]
    + 2 * (2.625 - v[0] + v[0] * v[1] * v[1] * v[1]) * 3 * v[0] * v[1] * v[1];
  return ans;
}

int main(int argc, char** argv) {
  int foo_dim = 2;
  int pop_size = 20;
  double dim_min = -10;
  double dim_max = 10;
  double mut_eps = 1e-3;
  double mut_thresh = 0.2;
  double gd_eta = 5 * 1e-2; // rastrigin: 5 * 1e-2, other: 1 * 1e-2
  int epochs = 10000;
  bool early_stop = false;

  try {
    TCLAP::CmdLine cmd("Genesis runner app");

    TCLAP::ValueArg<std::string> fooDimArg(
      "f", "foo-dim", "Function dimensionality",
      true, "", "int", cmd);
    TCLAP::ValueArg<std::string> popSizeArg(
      "p", "pop-size", "Population size",
      false, "20", "int", cmd);
    TCLAP::ValueArg<std::string> dimMinArg(
      "l", "low", "Lower border of space",
      false, "-10", "int", cmd);
    TCLAP::ValueArg<std::string> dimMaxArg(
      "m", "max", "Higher border of space",
      false, "10", "int", cmd);
    TCLAP::ValueArg<std::string> mutEpsArg(
      "v", "value", "Mutation value",
      false, "1e-3", "double", cmd);
    TCLAP::ValueArg<std::string> mutThreshArg(
      "t", "thresh", "Mutation threshold",
      false, "0.3", "double", cmd);
    TCLAP::ValueArg<std::string> epochsArg(
      "e", "epochs", "Epochs count",
      true, "", "int", cmd);
    TCLAP::SwitchArg earlyStopArg(
      "s", "stop", "Early stop flag",
      cmd, false);

    cmd.parse(argc, argv);
    foo_dim = stoi(fooDimArg.getValue());
    pop_size = stoi(popSizeArg.getValue());
    epochs = stoi(epochsArg.getValue());
    dim_min = atoi(dimMinArg.getValue().c_str());
    dim_max = atoi(dimMaxArg.getValue().c_str());
    mut_eps = atoi(mutEpsArg.getValue().c_str());
    mut_thresh = atoi(mutThreshArg.getValue().c_str());
    early_stop = earlyStopArg.getValue();

  } catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    return -1;
  }

  HybridGenesis gen(dim_min, dim_max, mut_eps, mut_thresh, 
    foo_rast, foo_dim, pop_size,
    foo_rast_deriv, gd_eta);

  gen.simulate(epochs, early_stop);
  auto best = gen.getBestCandidate();

  cout << '\n' << "Best minimum: ";
  cout << best.first << endl;
  cout << "Coordinates: " << endl;
  for (int i = 0; i < best.second.size(); ++i) {
    cout << best.second[i] << " ";
  }
  cout << endl;

  return 0;
}
