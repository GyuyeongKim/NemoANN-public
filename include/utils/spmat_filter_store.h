#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/log.h"

namespace pipeann {

  class SpmatFilterStore {
   public:
    void load(const std::string &path) {
      std::ifstream in(path, std::ios::binary);
      if (!in) {
        throw std::runtime_error("Cannot open " + path);
      }

      int64_t nrow, ncol, nnz;
      in.read((char *) &nrow, sizeof(int64_t));
      in.read((char *) &ncol, sizeof(int64_t));
      in.read((char *) &nnz, sizeof(int64_t));

      std::vector<int64_t> indptr(nrow + 1);
      in.read((char *) indptr.data(), (nrow + 1) * sizeof(int64_t));
      std::vector<int32_t> indices(nnz);
      in.read((char *) indices.data(), nnz * sizeof(int32_t));
      std::vector<float> data(nnz);
      in.read((char *) data.data(), nnz * sizeof(float));

      n_nodes_ = (uint32_t) nrow;
      indptr_.assign(nrow + 1, 0);
      for (int64_t i = 0; i < nrow; i++) {
        int64_t cnt = 0;
        for (int64_t j = indptr[i]; j < indptr[i + 1]; j++) {
          cnt += (data[j] != 0.0f);
        }
        indptr_[i + 1] = indptr_[i] + cnt;
      }

      labels_.resize(indptr_[nrow]);
      for (int64_t i = 0; i < nrow; i++) {
        int64_t pos = indptr_[i];
        for (int64_t j = indptr[i]; j < indptr[i + 1]; j++) {
          if (data[j] != 0.0f) {
            labels_[pos++] = (uint32_t) indices[j];
          }
        }
        std::sort(labels_.begin() + indptr_[i], labels_.begin() + indptr_[i + 1]);
      }

      double mb = (indptr_.size() * sizeof(int64_t) + labels_.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
      LOG(INFO) << "Filter store: " << n_nodes_ << " nodes, " << labels_.size() << " labels, " << std::fixed
                << std::setprecision(1) << mb << " MB";
    }

    bool passes(uint32_t id, const void *filter_data) const {
      if (id >= n_nodes_ || filter_data == nullptr) {
        return true;
      }

      uint32_t q_count;
      memcpy(&q_count, filter_data, sizeof(uint32_t));
      if (q_count == 0) {
        return true;
      }
      const uint32_t *q = (const uint32_t *) ((const char *) filter_data + sizeof(uint32_t));

      int64_t begin = indptr_[id];
      uint32_t n_count = (uint32_t) (indptr_[id + 1] - begin);
      if (n_count == 0) {
        return false;
      }

      uint32_t qi = 0, ni = 0;
      while (qi < q_count && ni < n_count) {
        uint32_t ql = q[qi];
        uint32_t nl = labels_[begin + ni];
        if (ql == nl) {
          qi++;
          ni++;
        } else if (nl < ql) {
          ni++;
        } else {
          return false;
        }
      }
      return qi == q_count;
    }

   private:
    uint32_t n_nodes_ = 0;
    std::vector<int64_t> indptr_;
    std::vector<uint32_t> labels_;
  };

}  // namespace pipeann
