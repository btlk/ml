#pragma once
#ifndef INCLUDE_DATA_H
#define INCLUDE_DATA_H

#include <string>
#include <vector>

namespace utils {

void ParseDataCSV(const std::string& file_path, 
                  std::vector<std::vector<double>>& data, 
                  std::vector<double>& labels);

void DumpDataToCSV(const std::vector<std::vector<double>>& data,
                   const std::vector<double>& labels,
                   const std::string& output_path);
                   

} // namespace utils

#endif // INCLUDE_DATA_H