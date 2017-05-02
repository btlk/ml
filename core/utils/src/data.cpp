#include <utils/data.h>

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iterator>

template<typename Out>
void internal_split(const std::string &s, 
                    char delim, 
                    Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = stod(item);
  }
}

std::vector<double> split(const std::string &s, 
                          char delim) {
  std::vector<double> elems;
  internal_split(s, delim, std::back_inserter(elems));
  return elems;
}

void utils::ParseDataCSV(const std::string& file_path,
                         std::vector<std::vector<double>>& data,
                         std::vector<double>& labels) {
  std::ifstream ifstr(file_path);
  std::string cur_str;
  while (std::getline(ifstr, cur_str)) {
    std::vector<double> splitted = split(cur_str, ',');
    labels.emplace_back(splitted[0]);
    data.emplace_back(splitted.begin() + 1, splitted.end());
  }
}

void utils::DumpDataToCSV(
  const std::vector<std::vector<double>>& data,
  const std::vector<double>& labels,
  const std::string& output_path) {
  std::ofstream ofstr(output_path);
  std::stringstream ss;
  std::string cur_str;
  int size = data.size();
  int dim = data[0].size();
  if (size != labels.size()) {
    throw std::runtime_error("Data size doesnt match labels size");
  }
  for (int i = 0; i < size; ++i) {
    ss << labels[i];
    for (int j = 0; j < dim; ++j) {
      ss << ',' << data[i][j];
    }
    ss >> cur_str;
    ss.clear();
    ofstr << cur_str << '\n';
  }
}