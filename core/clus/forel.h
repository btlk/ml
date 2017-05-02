#pragma once
#ifndef INCLUDE_FOREL_H
#define INCLUDE_FOREL_H

#include <clus/clus.h>

namespace clus {

class ForElClusterizer : public Clusterizer {
public:
  ForElClusterizer(double rad);

  virtual std::vector<double> fit(
    const std::vector<std::vector<double>>& data) override;

  virtual ~ForElClusterizer();

protected:
  double radius_;
};

} // namespace clus

#endif // INCLUDE_FOREL_H