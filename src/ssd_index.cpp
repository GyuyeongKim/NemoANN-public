#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>

#include <omp.h>
#include <cmath>
#include "distance.h"
#include "nbr/nbr.h"
#include "ssd_index_defs.h"
#include "utils/timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include "utils/tsl/robin_set.h"

namespace pipeann {
  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader,
                              AbstractNeighbor<T> *nbr_handler, bool tags, IndexBuildParameters *parameters)
      : reader(file_reader), nbr_handler(nbr_handler), metric(m), enable_tags(tags) {
    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

    if (parameters != nullptr) {
      this->params = *parameters;
      params.print();
    }
    LOG(INFO) << "Use " << nbr_handler->get_name() << " as neighbor handler, metric: " << get_metric_str(m);
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << pipeann::cache.cache.size();

    if (load_flag) {
      this->destroy_buffers();
      reader->close();
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(uint64_t n_threads) {
    uint64_t n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    this->thread_data_queue.null_T = nullptr;
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
      this->reader->register_buf(data->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_buffers() {
    while (!this->thread_data_bufs.empty()) {
      auto buf = this->thread_data_bufs.back();
      pipeann::aligned_free((void *) buf->coord_scratch);
      pipeann::aligned_free((void *) buf->sector_scratch);
      pipeann::aligned_free((void *) buf->nbr_vec_scratch);
      pipeann::aligned_free((void *) buf->nbr_ctx_scratch);
      pipeann::aligned_free((void *) buf->aligned_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_query_T);
      this->thread_data_bufs.pop_back();
      this->thread_data_queue.pop();
      delete buf;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(const std::string &mem_index_path) {
    if (!this->load_flag) {
      LOG(ERROR) << "Index is not loaded. Please load the index first.";
      exit(-1);
    }
    mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(this->metric, this->meta_.data_dim);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, uint32_t num_threads) {
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
    this->disk_index_file = disk_index_file;

    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!file_exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->load_page_layout(index_prefix, meta_.nnodes_per_sector, meta_.npoints);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector,
                                           const uint64_t num_points) {
    std::string partition_file = index_prefix + "_partition.bin.aligned";
    id2loc_.resize(num_points);  // pre-allocate space first.
    loc2id_.resize(cur_loc);     // pre-allocate space first.

    if (file_exists(partition_file)) {
      LOG(INFO) << "Loading partition file " << partition_file;
      std::ifstream part(partition_file);
      uint64_t C, partition_nums, nd;
      part.read((char *) &C, sizeof(uint64_t));
      part.read((char *) &partition_nums, sizeof(uint64_t));
      part.read((char *) &nd, sizeof(uint64_t));
      if (nnodes_per_sector <= 1 || C != nnodes_per_sector) {
        // graph reordering is useful only when nnodes_per_sector > 1.
        LOG(ERROR) << "partition information not correct.";
        exit(-1);
      }
      LOG(INFO) << "Partition meta: C: " << C << " partition_nums: " << partition_nums;

      uint64_t page_offset = loc_sector_no(0);
      auto st = std::chrono::high_resolution_clock::now();

      constexpr uint64_t n_parts_per_read = 1024 * 1024;
      std::vector<unsigned> part_buf(n_parts_per_read * (1 + nnodes_per_sector));
      for (uint64_t p = 0; p < partition_nums; p += n_parts_per_read) {
        uint64_t nxt_p = std::min(p + n_parts_per_read, partition_nums);
        part.read((char *) part_buf.data(), sizeof(unsigned) * n_parts_per_read * (1 + nnodes_per_sector));
#pragma omp parallel for schedule(dynamic)
        for (uint64_t i = p; i < nxt_p; ++i) {
          uint32_t base = (i - p) * (1 + nnodes_per_sector);
          uint32_t s = part_buf[base];  // size of this partition
          for (uint32_t j = 0; j < s; ++j) {
            uint64_t id = part_buf[base + 1 + j];
            uint64_t loc = i * nnodes_per_sector + j;
            id2loc_[id] = loc;
            loc2id_[loc] = id;
          }
          for (uint32_t j = s; j < nnodes_per_sector; ++j) {
            loc2id_[i * nnodes_per_sector + j] = kInvalidID;
          }
        }
      }
      auto et = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Page layout loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count()
                << " ms";
    } else {
      LOG(INFO) << partition_file << " does not exist, use equal partition mapping";
// use equal mapping for id2loc and page_layout.
#ifndef NO_MAPPING
#pragma omp parallel for
      for (size_t i = 0; i < meta_.npoints; ++i) {
        id2loc_[i] = i;
        loc2id_[i] = i;
      }
      for (size_t i = meta_.npoints; i < this->cur_loc; ++i) {
        loc2id_[i] = kInvalidID;
      }
#endif
    }
    LOG(INFO) << "Page layout loaded.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
      LOG(INFO) << "Loaded " << tags.size() << " tags";
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_nbr_store(uint32_t max_nbrs) {
    std::string prefix = disk_index_file.substr(0, disk_index_file.rfind("_disk.index"));
    std::string path = prefix + "_nbr_store_R" + std::to_string(max_nbrs) + ".bin";

    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
      LOG(INFO) << "Loading neighbor store from " << path;
      nbr_store_ = std::make_unique<NbrStore>();
      if (nbr_store_->load(path)) {
        return;
      }
      nbr_store_.reset();
    }

    LOG(INFO) << "Building neighbor store (R_max = " << max_nbrs << ") from " << disk_index_file;
    nbr_store_ = std::make_unique<NbrStore>();
    nbr_store_->init(meta_.npoints, max_nbrs);

    void *ctx = reader->get_ctx();
    char *sector_buf;
    pipeann::alloc_aligned((void **) &sector_buf, size_per_io, SECTOR_LEN);

    const uint64_t step = meta_.nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN);
    const uint64_t end = loc_sector_no(meta_.nnodes_per_sector > 0 ? meta_.npoints + meta_.nnodes_per_sector - 1
                                                                   : meta_.npoints);
    for (uint64_t sec = 1; sec < end; sec += step) {
      std::vector<IORequest> reqs = {IORequest(sec * SECTOR_LEN, size_per_io, sector_buf, 0, 0, sector_buf)};
      reader->read(reqs, ctx, false);

      uint64_t first = sector_to_loc(sec, 0);
      uint64_t n = meta_.nnodes_per_sector > 0 ? meta_.nnodes_per_sector : 1;
      for (uint64_t off = 0; off < n; off++) {
        uint64_t loc = first + off;
        if (loc >= meta_.npoints) {
          break;
        }
        uint32_t id = (loc < loc2id_.size()) ? loc2id_[loc] : loc;
        if (id >= meta_.npoints) {
          continue;
        }
        DiskNode<T> node = node_from_page(sector_buf, meta_.nnodes_per_sector > 0 ? off : 0);
        nbr_store_->set(id, node.nnbrs, node.nbrs);
      }
    }
    pipeann::aligned_free(sector_buf);
    LOG(INFO) << "Neighbor store built: " << meta_.npoints << " nodes, " << nbr_store_->size_gb() << " GB";
    nbr_store_->save(path);
  }

  template<typename T, typename TagT>
  Index<T, TagT> *SSDIndex<T, TagT>::load_to_mem(const std::string &filename) {
    std::string disk_index_file = std::string(filename) + "_disk.index";
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);  // ensure that loc_sector_no is correct.

    Index<T, TagT> *mem_index = new Index<T, TagT>(this->metric, meta.data_dim);

    mem_index->_ep = meta.entry_point;
    mem_index->range = meta.range;
    mem_index->_nd = meta.npoints;
    mem_index->resize(meta.npoints);

    reader->open(disk_index_file, true, false);

    const uint64_t kLocsPerRead = meta_.nnodes_per_sector > 0 ? ROUND_UP(131072, meta_.nnodes_per_sector)
                                                              : 131072;  // process this many locs per iteration

    // Calculate buffer size based on whether we have large or small nodes
    uint64_t sectors_per_batch = meta_.nnodes_per_sector > 0
                                     ? kLocsPerRead / meta_.nnodes_per_sector
                                     : kLocsPerRead * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN);
    uint64_t buf_size_per_batch = sectors_per_batch * SECTOR_LEN;

    char *buf;
    pipeann::alloc_aligned((void **) &buf, buf_size_per_batch, SECTOR_LEN);

    for (uint64_t loc_st = 0; loc_st < meta.npoints; loc_st += kLocsPerRead) {
      uint64_t loc_ed = std::min(meta.npoints, loc_st + kLocsPerRead);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, buf, 0, 0));
      reader->read(read_reqs, reader->get_ctx(), false);

#pragma omp parallel for
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc;
#pragma omp critical
        {
          mem_index->_location_to_tag[id] = id;
          mem_index->_tag_to_location[id] = id;
        }

        uint64_t loc_sector = loc_sector_no(loc);
        auto page_rbuf = buf + (loc_sector - st_sector) * SECTOR_LEN;
        DiskNode<T> node = node_from_page(page_rbuf, loc);

        // load data and nhood.
        memcpy(mem_index->_data.data() + id * meta.data_dim, node.coords, meta.data_dim * sizeof(T));
        std::vector<uint32_t> nhood;

        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          nhood.push_back(node.nbrs[i]);
        }
        mem_index->_final_graph[id] = nhood;
      }

      LOG(INFO) << "Loaded " << loc_ed << "/" << meta.npoints << " nodes.";
    }
    return mem_index;
  }

  template<typename T, typename TagT>
  TagT SSDIndex<T, TagT>::id2tag(uint32_t id) {
#ifdef NO_MAPPING
    return id;  // use ID to replace tags.
#else
    TagT ret;
    if (tags.find(id, ret)) {
      return ret;
    } else {
      return id;
    }
#endif
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::id2loc(uint32_t id) {
#ifdef NO_MAPPING
    return id;
#else
    id2loc_resize_mu_.lock_shared();
    if (unlikely(id >= id2loc_.size())) {
      LOG(ERROR) << "id " << id << " is out of range " << id2loc_.size();
      crash();
      return kInvalidID;
    }
    uint32_t ret = id2loc_[id];
    id2loc_resize_mu_.unlock_shared();
    return ret;
#endif
  }

  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors) {
    std::vector<uint32_t> to_lock;
    to_lock.assign(neighbors.begin(), neighbors.end());
    if (target != kInvalidID) {
      to_lock.push_back(target);
    }
    // Sort and deduplicate.
    std::sort(to_lock.begin(), to_lock.end());
    auto last = std::unique(to_lock.begin(), to_lock.end());
    to_lock.erase(last, to_lock.end());
    return to_lock;
  }

  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                                    const std::vector<uint32_t> &neighbors, bool rd) {
#ifndef READ_ONLY_TESTS
    std::vector<uint32_t> to_lock = get_to_lock_idx(target, neighbors);
    for (auto &id : to_lock) {
      rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
    }
    return to_lock;
#else
    return std::vector<uint32_t>();
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table,
                                     const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
    for (auto &id : to_lock) {
      lock_table.unlock(id);
    }
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock) {
#ifndef READ_ONLY_TESTS
    lock_table.unlock(to_lock);
#endif
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
