#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ssd_index.h"
#include "linux_aligned_file_reader.h"
#include "utils.h"

namespace {
  double rss_gb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
      if (line.compare(0, 6, "VmRSS:") == 0) {
        long kb = 0;
        sscanf(line.c_str(), "VmRSS: %ld", &kb);
        return kb / (1024.0 * 1024.0);
      }
    }
    return 0;
  }

  bool is_spmat(const std::string &path) {
    const std::string ext = ".spmat";
    return path.size() >= ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0;
  }

  struct QueryFilters {
    bool multi = false;
    std::vector<uint8_t> single;
    std::vector<std::vector<uint32_t>> sets;

    void load(const std::string &path) {
      std::ifstream in(path, std::ios::binary);
      if (!in) {
        throw std::runtime_error("Cannot open " + path);
      }
      multi = is_spmat(path);
      if (!multi) {
        uint32_t n;
        in.read((char *) &n, sizeof(n));
        single.resize(n);
        in.read((char *) single.data(), n);
        return;
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
      sets.resize(nrow);
      for (int64_t i = 0; i < nrow; i++) {
        std::vector<uint32_t> labels;
        for (int64_t j = indptr[i]; j < indptr[i + 1]; j++) {
          if (data[j] != 0.0f) {
            labels.push_back((uint32_t) indices[j]);
          }
        }
        std::sort(labels.begin(), labels.end());
        sets[i].push_back((uint32_t) labels.size());
        sets[i].insert(sets[i].end(), labels.begin(), labels.end());
      }
    }

    size_t size() const {
      return multi ? sets.size() : single.size();
    }

    const void *get(size_t i) const {
      return multi ? (const void *) sets[i].data() : (const void *) &single[i];
    }
  };

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
  struct StoreSelector : pipeann::AbstractSelector {
    const pipeann::SSDIndex<T> *index;
    explicit StoreSelector(const pipeann::SSDIndex<T> *index) : index(index) {
    }
    bool is_member(uint32_t id, const void *query_labels, const void *) override {
      return index->check_filter(id, query_labels);
    }
  };

  enum class System { DiskANN, PipeANN, NemoANN };

  template<typename T>
  int run(int argc, char **argv) {
    int arg = 2;
    std::string index_prefix = argv[arg++];
    uint32_t num_threads = std::atoi(argv[arg++]);
    uint32_t beam_width = std::atoi(argv[arg++]);
    std::string query_file = argv[arg++];
    std::string base_labels = argv[arg++];
    std::string query_labels = argv[arg++];
    std::string gt_file = argv[arg++];
    uint64_t K = std::atoi(argv[arg++]);
    std::string system_name = argv[arg++];
    uint32_t mem_L = std::atoi(argv[arg++]);
    uint32_t r_max = std::atoi(argv[arg++]);

    System system;
    if (system_name == "diskann") {
      system = System::DiskANN;
    } else if (system_name == "pipeann") {
      system = System::PipeANN;
    } else if (system_name == "nemoann") {
      system = System::NemoANN;
    } else {
      std::cerr << "Unknown system " << system_name << std::endl;
      return -1;
    }

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

    QueryFilters filters;
    filters.load(query_labels);
    if (filters.size() < nq) {
      std::cerr << "Fewer query labels than queries" << std::endl;
      return -1;
    }

    std::vector<uint32_t> gt;
    uint64_t gt_nq, gt_k;
    load_gt(gt_file, gt, gt_nq, gt_k);
    if (gt_nq < nq) {
      std::cerr << "Fewer ground-truth rows than queries" << std::endl;
      return -1;
    }

    pipeann::Metric metric = pipeann::Metric::L2;
    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<pipeann::SSDIndex<T>> index(
        new pipeann::SSDIndex<T>(metric, reader, pipeann::get_nbr_handler<T>(metric, "pq"), true));
    if (index->load(index_prefix.c_str(), num_threads) != 0) {
      return -1;
    }
    if (mem_L != 0) {
      index->load_mem_index(index_prefix + "_mem.index");
    }
    omp_set_num_threads(num_threads);

    if (is_spmat(base_labels)) {
      index->load_spmat_filter_store(base_labels);
    } else {
      index->load_filter_store(base_labels);
    }

    if (system == System::NemoANN) {
      index->load_nbr_store(r_max);
      index->enable_tunneling();
    }

    StoreSelector<T> selector(index.get());

    std::cout << std::setw(6) << "L" << std::setw(10) << "QPS" << std::setw(10) << "Lat_us" << std::setw(10)
              << "Recall@" + std::to_string(K) << std::setw(10) << "IOs" << std::setw(10) << "Tunnels"
              << std::setw(10) << "IO_us" << std::setw(11) << "Tunnel_us" << std::setw(10) << "Proc_us"
              << std::setw(9) << "RSS_GB" << std::endl;

    std::vector<uint32_t> res(nq * K);
    std::vector<pipeann::QueryStats> stats(nq);
    for (uint64_t L : Ls) {
      std::fill(res.begin(), res.end(), 0);
      std::fill(stats.begin(), stats.end(), pipeann::QueryStats{});

      auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) nq; i++) {
        const void *f = filters.get(i);
        uint32_t *out = res.data() + i * K;
        if (system == System::NemoANN) {
          index->pipe_search(query + i * dim, K, mem_L, L, out, nullptr, beam_width, &stats[i], nullptr, f);
        } else if (system == System::PipeANN) {
          index->pipe_search(query + i * dim, K, mem_L, L, out, nullptr, beam_width, &stats[i], &selector, f);
        } else {
          std::vector<uint32_t> cand(L);
          std::vector<float> cand_dists(L);
          index->beam_search(query + i * dim, L, mem_L, L, cand.data(), cand_dists.data(), beam_width, &stats[i]);
          size_t found = 0;
          for (size_t j = 0; j < L && found < K; j++) {
            if (selector.is_member(cand[j], f, nullptr)) {
              out[found++] = cand[j];
            }
          }
        }
      }
      auto t1 = std::chrono::high_resolution_clock::now();

      double sec = std::chrono::duration<double>(t1 - t0).count();
      double qps = nq / sec;
      auto mean = [&](double pipeann::QueryStats::*field) {
        return pipeann::get_mean_stats(stats.data(), nq, [field](const pipeann::QueryStats &s) { return s.*field; });
      };

      std::cout << std::setw(6) << L << std::setw(10) << std::fixed << std::setprecision(0) << qps << std::setw(10)
                << std::setprecision(1) << num_threads * 1e6 / qps << std::setw(10) << std::setprecision(4)
                << recall(gt, gt_k, res, nq, K) << std::setw(10) << std::setprecision(2)
                << mean(&pipeann::QueryStats::n_ios) << std::setw(10) << mean(&pipeann::QueryStats::n_tunnels)
                << std::setw(10) << std::setprecision(0) << mean(&pipeann::QueryStats::bd_io_us) << std::setw(11)
                << mean(&pipeann::QueryStats::bd_tunnel_us) << std::setw(10)
                << mean(&pipeann::QueryStats::bd_process_us) << std::setw(9) << std::setprecision(2) << rss_gb()
                << std::endl;
    }

    delete[] query;
    return 0;
  }
}  // namespace

int main(int argc, char **argv) {
  if (argc < 14) {
    std::cerr << "Usage: " << argv[0]
              << " <uint8|int8|float> <index_prefix> <threads> <W> <query.bin> <base_labels> <query_labels>"
                 " <gt.bin> <K> <diskann|pipeann|nemoann> <mem_L> <R_max> <L1> [L2 ...]"
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
