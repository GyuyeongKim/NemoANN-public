#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "utils/log.h"

namespace pipeann {

  class NbrStore {
   public:
    NbrStore() = default;
    ~NbrStore() {
      release();
    }

    NbrStore(const NbrStore &) = delete;
    NbrStore &operator=(const NbrStore &) = delete;

    void init(uint64_t n_nodes, uint32_t max_nbrs) {
      release();
      n_nodes_ = n_nodes;
      max_nbrs_ = max_nbrs;
      stride_ = 1 + max_nbrs;
      buf_.assign(n_nodes * stride_, 0);
      data_ = buf_.data();
    }

    void set(uint32_t id, uint32_t nnbrs, const uint32_t *nbrs) {
      uint32_t *p = buf_.data() + (uint64_t) id * stride_;
      uint32_t keep = std::min(nnbrs, max_nbrs_);
      p[0] = keep;
      memcpy(p + 1, nbrs, keep * sizeof(uint32_t));
    }

    void get(uint32_t id, uint32_t &nnbrs, const uint32_t *&nbrs) const {
      const uint32_t *p = data_ + (uint64_t) id * stride_;
      nnbrs = p[0];
      nbrs = p + 1;
    }

    bool save(const std::string &path) const {
      FILE *f = fopen(path.c_str(), "wb");
      if (f == nullptr) {
        LOG(ERROR) << "Cannot write " << path;
        return false;
      }
      uint64_t header[3] = {n_nodes_, max_nbrs_, stride_};
      fwrite(header, sizeof(uint64_t), 3, f);
      fwrite(data_, sizeof(uint32_t), n_nodes_ * stride_, f);
      fclose(f);
      LOG(INFO) << "Neighbor store saved to " << path;
      return true;
    }

    bool load(const std::string &path) {
      release();
      int fd = open(path.c_str(), O_RDONLY);
      if (fd < 0) {
        return false;
      }
      uint64_t header[3];
      if (::read(fd, header, sizeof(header)) != (ssize_t) sizeof(header)) {
        close(fd);
        return false;
      }
      n_nodes_ = header[0];
      max_nbrs_ = (uint32_t) header[1];
      stride_ = header[2];

      struct stat st;
      fstat(fd, &st);
      map_size_ = st.st_size;
      map_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
      close(fd);
      if (map_ == MAP_FAILED) {
        map_ = nullptr;
        LOG(ERROR) << "mmap failed for " << path;
        return false;
      }
      data_ = reinterpret_cast<const uint32_t *>(static_cast<const char *>(map_) + sizeof(header));
      LOG(INFO) << "Neighbor store: " << n_nodes_ << " nodes, R_max " << max_nbrs_ << ", " << size_gb() << " GB";
      return true;
    }

    double size_gb() const {
      return (double) n_nodes_ * stride_ * sizeof(uint32_t) / (1024.0 * 1024.0 * 1024.0);
    }

   private:
    void release() {
      if (map_ != nullptr) {
        munmap(map_, map_size_);
        map_ = nullptr;
      }
      buf_.clear();
      buf_.shrink_to_fit();
      data_ = nullptr;
    }

    uint64_t n_nodes_ = 0;
    uint32_t max_nbrs_ = 0;
    uint64_t stride_ = 0;
    std::vector<uint32_t> buf_;
    const uint32_t *data_ = nullptr;
    void *map_ = nullptr;
    size_t map_size_ = 0;
  };

}  // namespace pipeann
