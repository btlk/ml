#pragma once
#ifndef INCLUDE_GENE_H
#define INCLUDE_GENE_H

#include <vector>
#include <set>
#include <functional>

namespace gene {

/* ------------------------------------------- */
/* ---------------- interface ---------------- */
/* ------------------------------------------- */

using FitnessFunction = std::function<
  double(const std::vector<double>&)>;
using FFDerivative = std::function<
  std::vector<double>(const std::vector<double>&)>;
using Candidate = std::pair<double, std::vector<double>>;

class Genesis {
public:
  Genesis(double mn, double mx, double me, double mt, 
          FitnessFunction ff, int fd, int cp);
  Genesis(const Genesis& obj) = delete;
  Genesis(Genesis&& obj) = delete;
  Genesis& operator=(const Genesis& obj) = delete;
  Genesis& operator=(Genesis&& obj) = delete;

  virtual void simulate(int epochs, bool flag) = 0;
  Candidate getBestCandidate();

  virtual ~Genesis();

private:
  virtual void crossoverAndMutate(int quantity, double eps, double chance) = 0;

protected:
  double dim_min_;
  double dim_max_;
  double mut_eps_;
  double mut_thresh_;

  int ffunc_dim_;
  FitnessFunction ffunc_;

  int population_;
  std::multiset<Candidate> candidates_;
};

/* ------------------------------------------- */
/* ------------- specializations ------------- */
/* ------------------------------------------- */

class HybridGenesis : public Genesis {
public:
  HybridGenesis(double mn, double mx, double me, double mt,
                FitnessFunction ff, int fd, int cp, 
                FFDerivative fder, double eta);

  virtual void simulate(int epochs, bool flag) override;

  virtual ~HybridGenesis() override;

private:
  virtual void crossoverAndMutate(int quantity, double eps, double chance) override;
  void applyGradientDescent(std::vector<double>& cand);

protected:
  double eta_;

  FFDerivative ffunc_der_;
};

} // namespace gene

#endif // INCLUDE_GENE_H
