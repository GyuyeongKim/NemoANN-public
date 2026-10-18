#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pipeann {

  class FilterStore {
   public:
    void load(const std::string &path) {
      std::ifstream in(path, std::ios::binary);
      if (!in) {
        throw std::runtime_error("Cannot open " + path);
      }
      uint32_t n;
      in.read((char *) &n, sizeof(n));
      labels_.resize(n);
      in.read((char *) labels_.data(), n);
      if (!in) {
        throw std::runtime_error("Cannot read " + path);
      }
    }

    bool passes(uint32_t id, uint8_t label) const {
      if (id >= labels_.size()) {
        return true;
      }
      return labels_[id] == label;
    }

    size_t size() const {
      return labels_.size();
    }

   private:
    std::vector<uint8_t> labels_;
  };

}  // namespace pipeann
