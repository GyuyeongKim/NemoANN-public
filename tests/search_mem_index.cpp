#include <omp.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ssd_index.h"
#include "index.h"
#include "nbr/dummy_nbr.h"
#include "linux_aligned_file_reader.h"
#include "utils.h"

namespace {
  std::vector<uint8_t> load_labels(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Cannot open " + path);
    }
    uint32_t n;
    in.read((char *) &n, sizeof(n));
    std::vector<uint8_t> labels(n);
    in.read((char *) labels.data(), n);
    return labels;
  }

  void load_gt(const std::string &path, std::vector<uint32_t> &ids, uint64_t &nq, uint64_t &k) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Cannot open " + path);
    }
    uint32_t n, d;
    in.read((char *) &n, sizeof(n));
    in.read((char *) &d, sizeof(d));
    nq = n;
    k = d;
    ids.resize(nq * k);
    in.read((char *) ids.data(), nq * k * sizeof(uint32_t));
  }

  double recall(const std::vector<uint32_t> &gt, uint64_t gt_k, const std::vector<uint32_t> &res, uint64_t nq,
                uint64_t k) {
    double total = 0;
    uint64_t counted = 0;
    for (uint64_t i = 0; i < nq; i++) {
      std::set<uint32_t> truth;
      for (uint64_t j = 0; j < std::min(gt_k, k); j++) {
        uint32_t id = gt[i * gt_k + j];
        if (id != std::numeric_limits<uint32_t>::max()) {
          truth.insert(id);
        }
      }
      if (truth.empty()) {
        continue;
      }
      uint64_t hits = 0;
      for (uint64_t j = 0; j < k; j++) {
        hits += truth.count(res[i * k + j]);
      }
      total += (double) hits / truth.size();
      counted++;
    }
    return counted > 0 ? total / counted : 0.0;
  }

  template<typename T>
  int run(int argc, char **argv) {
    int arg = 2;
    std::string index_prefix = argv[arg++];
    uint32_t num_threads = std::atoi(argv[arg++]);
    std::string query_file = argv[arg++];
    std::string base_labels_file = argv[arg++];
    std::string query_labels_file = argv[arg++];
    std::string gt_file = argv[arg++];
    uint64_t K = std::atoi(argv[arg++]);

    std::vector<uint64_t> Ls;
    for (int i = arg; i < argc; i++) {
      uint64_t L = std::atoi(argv[i]);
      if (L >= K) {
        Ls.push_back(L);
      }
    }
    if (Ls.empty()) {
      std::cerr << "No valid L" << std::endl;
      return -1;
    }

    T *query = nullptr;
    size_t nq, dim;
    pipeann::load_bin<T>(query_file, query, nq, dim);
    std::vector<uint8_t> base_labels = load_labels(base_labels_file);
    std::vector<uint8_t> query_labels = load_labels(query_labels_file);
    std::vector<uint32_t> gt;
    uint64_t gt_nq, gt_k;
    load_gt(gt_file, gt, gt_nq, gt_k);
    if (query_labels.size() < nq || gt_nq < nq) {
      std::cerr << "Query labels or ground truth do not cover all queries" << std::endl;
      return -1;
    }

    pipeann::Metric metric = pipeann::Metric::L2;
    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    pipeann::SSDIndex<T> disk_index(metric, reader, new pipeann::DummyNeighbor<T>(metric), true, nullptr);
    auto t_load = std::chrono::high_resolution_clock::now();
    auto &index = *disk_index.load_to_mem(index_prefix);
    std::cout << "Loaded index into memory in " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_load).count() << " s"
              << std::endl;
    omp_set_num_threads(num_threads);

    std::cout << std::setw(6) << "L" << std::setw(10) << "QPS" << std::setw(10) << "Lat_us" << std::setw(10)
              << "Recall@" + std::to_string(K) << std::endl;

    std::vector<uint32_t> res(nq * K);
    std::vector<pipeann::QueryStats> stats(nq);
    for (uint64_t L : Ls) {
      std::fill(res.begin(), res.end(), 0);

      auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) nq; i++) {
        uint8_t ql = query_labels[i];
        std::vector<uint32_t> cand(L);
        std::vector<float> cand_dists(L);
        index.search(query + i * dim, L, L, cand.data(), cand_dists.data(), &stats[i]);
        uint32_t *out = res.data() + i * K;
        size_t found = 0;
        for (size_t j = 0; j < L && found < K; j++) {
          if (cand[j] < base_labels.size() && base_labels[cand[j]] == ql) {
            out[found++] = cand[j];
          }
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();

      double qps = nq / std::chrono::duration<double>(t1 - t0).count();
      std::cout << std::setw(6) << L << std::setw(10) << std::setprecision(0) << qps << std::setw(10)
                << std::setprecision(1) << num_threads * 1e6 / qps << std::setw(10) << std::setprecision(4)
                << recall(gt, gt_k, res, nq, K) << std::endl;
    }

    delete[] query;
    return 0;
  }
}  // namespace

int main(int argc, char **argv) {
  if (argc < 10) {
    std::cerr << "Usage: " << argv[0]
              << " <uint8|int8|float> <index_prefix> <threads> <query.bin> <base_labels.bin> <query_labels.bin>"
                 " <gt.bin> <K> <L1> [L2 ...]"
              << std::endl;
    return -1;
  }
  std::string type = argv[1];
  if (type == "uint8") {
    return run<uint8_t>(argc, argv);
  } else if (type == "int8") {
    return run<int8_t>(argc, argv);
  } else if (type == "float") {
    return run<float>(argc, argv);
  }
  std::cerr << "Unknown type " << type << std::endl;
  return -1;
}
