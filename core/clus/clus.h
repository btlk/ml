#pragma once
#ifndef INCLUDE_CLUS_H
#define INCLUDE_CLUS_H

#include <clus/clus.h>
#include <vector>

namespace clus {

/* ------------------------------------------- */
/* ---------------- interface ---------------- */
/* ------------------------------------------- */

class Clusterizer {
public:
  Clusterizer() = default;
  Clusterizer(const Clusterizer& obj) = delete;
  Clusterizer(Clusterizer&& obj) = delete;
  Clusterizer& operator=(const Clusterizer& obj) = delete;
  Clusterizer& operator=(Clusterizer&& obj) = delete;

  virtual std::vector<double> fit(
    const std::vector<std::vector<double>>& data) = 0;

  virtual ~Clusterizer();
};

} // namespace clus

#endif // INCLUDE_CLUS_H
