#include <gene/gene.h>
#include <utils/initializer.h>

#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <chrono>
#include <iostream>
#include <queue>

using namespace gene;
using namespace utils;

/* ----------------- Genesis ----------------- */

Genesis::Genesis(
  double mn, double mx, double me, double mt,
  FitnessFunction ff, int fd, int cp) 
  : dim_min_(mn)
  , dim_max_(mx)
  , mut_eps_(me)
  , mut_thresh_(mt)
  , ffunc_(ff)
  , ffunc_dim_(fd)
  , population_(cp)
  , candidates_() {
  // empty body
}

Candidate Genesis::getBestCandidate() {
  return *candidates_.begin();
}

Genesis::~Genesis() {
  // empty body
}

/* -------------- HybridGenesis -------------- */

HybridGenesis::HybridGenesis(
  double mn, double mx, double me, double mt,
  FitnessFunction ff, int fd, int cp, 
  FFDerivative fder, double eta)
  : Genesis(mn, mx, me, mt, ff, fd, cp)
  , ffunc_der_(fder)
  , eta_(eta) {
  // empty body
}

void HybridGenesis::simulate(int epochs, bool early_stop) {
  candidates_.clear();
  std::vector<std::vector<double>> initial_cands(population_);
  for (int i = 0; i < population_; ++i) {
    initial_cands[i] = UniformDistribution(ffunc_dim_, dim_min_, dim_max_);
  }
  for (int i = 0; i < population_; ++i) {
    candidates_.insert(
      std::make_pair(ffunc_(initial_cands[i]), initial_cands[i])
    );
  }

  if (early_stop) {
    double break_thresh = 1e-9;
    std::vector<double> cache = UniformDistribution(1000, -1.0, 1.0);
    int cache_ind = 0;
    double cache_sum = 0;
    for (int i = 0; i < epochs; ++i) {
      crossoverAndMutate(population_ / 2, mut_eps_, mut_thresh_);
      cache_ind = i % cache.size();
      cache[cache_ind] = getBestCandidate().first;
      std::cout << cache[cache_ind] << std::endl;
      cache_sum = 0;
      for (int j = 0; j < cache.size(); ++j) {
        cache_sum += cache[j];
      }
      if (fabs(cache_sum / cache.size() - cache[cache_ind]) < break_thresh) {
        std::cout << "\nAborting simulation due stagnation." << std::endl;
        break;
      }
    }
  } else {
    for (int i = 0; i < epochs; ++i) {
      crossoverAndMutate(population_ / 2, mut_eps_, mut_thresh_);
      if (i % 100 == 0)
        std::cout << getBestCandidate().first << std::endl;
    }
  }
}

void HybridGenesis::crossoverAndMutate(int quantity, double eps, double chance) {
  std::vector<std::vector<double>> children (quantity * 2, std::vector<double>(ffunc_dim_));

  // breeding phase
  std::vector<int> partners = UniformDistribution(quantity, 0, quantity - 1);
  std::vector<int> crosspoints = UniformDistribution(quantity, 0, ffunc_dim_ - 1);
  int i = 0;
  for (auto it = candidates_.begin(); i < quantity; ++i, ++it) {
    auto partner = partners[i] > 0 ? std::next(it, partners[i]) : it;
    for (int j = 0; j < crosspoints[i]; ++j) {
      children[i][j] = it->second[j];
      children[quantity + i][j] = partner->second[j];
    }
    for (int j = crosspoints[i]; j < ffunc_dim_; ++j) {
      children[i][j] = partner->second[j];
      children[quantity + i][j] = it->second[j];
    }
  }

  // mutation and gradient descent
  double roll = 0.0;
  double mut = 0.0;
  for (i = 0; i < quantity * 2; ++i) {
    for (int j = 0; j < ffunc_dim_; ++j) {
      roll = (double) std::rand() / RAND_MAX;
      if (roll > chance) {
        roll = (double) std::rand() / RAND_MAX;
        mut = eps * 0.1 + (double) std::rand() / RAND_MAX * eps;
        children[i][j] += roll > 0.5 ? mut : -mut;
      }
    }
    applyGradientDescent(children[i]);
    candidates_.insert(std::make_pair(ffunc_(children[i]), children[i]));
  }

  // remove worst
  auto last_iter = std::next(candidates_.begin(), quantity * 2 - 1);
  candidates_.erase(last_iter, candidates_.end());
}

void HybridGenesis::applyGradientDescent(std::vector<double>& cand) {
  int n = (int)cand.size();
  auto etas = UniformDistribution(n, eta_ * 1e-2, eta_);
  auto grads = ffunc_der_(cand);
  for (int i = 0; i < n; ++i) {
    cand[i] -= etas[i] * grads[i];
  }
}

HybridGenesis::~HybridGenesis() {
  // empty body
}