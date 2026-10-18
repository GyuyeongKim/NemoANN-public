#pragma once
#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include <string>
#include <set>
#include <omp.h>

#include "aligned_file_reader.h"
#include "ssd_index_defs.h"
#include "filter/selector.h"
#include "utils/concurrent_queue.h"
#include "utils/lock_table.h"
#include "utils/percentile_stats.h"
#include "nbr/nbr.h"
#include "utils.h"
#include "index.h"
#include "utils/filter_store.h"
#include "utils/spmat_filter_store.h"
#include "utils/nbr_store.h"

namespace pipeann {
  template<typename T, typename TagT = uint32_t>
  class SSDIndex {
   public:
    std::unique_ptr<Index<T, uint32_t>> mem_index_;  // in-memory navigation graph

    // Index metadata (consolidated).
    SSDIndexMetadata<T> meta_;

    std::atomic<uint64_t> cur_id, cur_loc;

    SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader,
             AbstractNeighbor<T> *nbr = new PQNeighbor<T>(), bool tags = false,
             IndexBuildParameters *parameters = nullptr);

    ~SSDIndex();

    // Use node_from_page() to create instances for DiskNode.
    // Params: location (id2loc first if using ID), the page-aligned buffer.
    // The offset is calculated in the constructor.
    inline DiskNode<T> node_from_page(char *page_buf, uint32_t loc) {
      return DiskNode<T>(page_buf, loc, meta_);
    }

    // Size of the data region in a DiskNode.
    inline uint64_t node_label_size() const {
      return meta_.label_size;
    }

    // Unaligned offset to location.
    inline uint64_t u_loc_offset(uint64_t loc) {
      return loc * meta_.max_node_len;  // compacted store.
    }

    inline uint64_t u_loc_offset_nbr(uint64_t loc) {
      return loc * meta_.max_node_len + meta_.data_dim * sizeof(T);
    }

    // Avoid integer overflow when * SECTOR_LEN.
    inline uint64_t loc_sector_no(uint64_t loc) {
      return 1 + (meta_.nnodes_per_sector > 0 ? loc / meta_.nnodes_per_sector
                                              : loc * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) {
      return meta_.nnodes_per_sector == 0
                 ? (sector_no - 1) / DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN)  // sector_off == 0.
                 : (sector_no - 1) * meta_.nnodes_per_sector + sector_off;
    }

    void init_metadata(const SSDIndexMetadata<T> &meta) {
      meta.print();
      this->meta_ = meta;
      this->cur_id = this->cur_loc = meta.npoints;
      this->aligned_dim = ROUND_UP(meta_.data_dim, 8);
      this->params.R = meta.range;
      this->size_per_io = SECTOR_LEN * (meta_.nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));
      LOG(INFO) << "Size per IO: " << size_per_io;

      // Aligned.
      if (meta_.nnodes_per_sector != 0 && meta_.npoints % meta_.nnodes_per_sector != 0) {
        cur_loc += meta_.nnodes_per_sector - (meta_.npoints % meta_.nnodes_per_sector);
      }
      LOG(INFO) << "Cur location: " << this->cur_loc;

      if (params.L == 0) {
        params.set(meta.range, meta.range + 32, 384, 1.2, 0, true, 4);
        params.print();
      }
    }

    void init_query_buf(QueryBuffer<T> &buf) {
      pipeann::alloc_aligned((void **) &buf.coord_scratch, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      pipeann::alloc_aligned((void **) &buf.sector_scratch, MAX_N_SECTOR_READS * size_per_io, SECTOR_LEN);
      pipeann::alloc_aligned((void **) &buf.nbr_vec_scratch,
                             MAX_N_EDGES * AbstractNeighbor<T>::MAX_BYTES_PER_NBR * sizeof(uint8_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_ctx_scratch, ROUND_UP(nbr_handler->query_ctx_size(), 256), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_dist_scratch, MAX_N_EDGES * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_query_T, this->aligned_dim * sizeof(T), 8 * sizeof(T));

      buf.visited = new tsl::robin_set<uint64_t>(4096);
      buf.page_visited = new tsl::robin_set<unsigned>(4096);

      memset(buf.sector_scratch, 0, MAX_N_SECTOR_READS * SECTOR_LEN);
      memset(buf.coord_scratch, 0, this->aligned_dim * sizeof(T));
      memset(buf.aligned_query_T, 0, this->aligned_dim * sizeof(T));
    }

    QueryBuffer<T> *pop_query_buf(const T *query) {
      QueryBuffer<T> *data = this->thread_data_queue.pop();
      while (data == this->thread_data_queue.null_T) {
        this->thread_data_queue.wait_for_push_notify();
        data = this->thread_data_queue.pop();
      }

      if (likely(query != nullptr)) {
        if (this->metric == Metric::COSINE) {
          // Data has been normalized. Normalize search vector too.
          pipeann::normalize_data(data->aligned_query_T, query, meta_.data_dim);
        } else {
          memcpy(data->aligned_query_T, query, meta_.data_dim * sizeof(T));
        }
      }
      return data;
    }

    void push_query_buf(QueryBuffer<T> *data) {
      this->thread_data_queue.push(data);
      this->thread_data_queue.push_notify_all();
    }

    // Load compressed data, and obtains the handle to the disk-resident index.
    int load(const char *index_prefix, uint32_t num_threads);

    void load_mem_index(const std::string &mem_index_path);

    // Load disk index to memory index.
    Index<T, TagT> *load_to_mem(const std::string &filename);

    size_t beam_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       AbstractSelector *selector = nullptr, const void *filter_data = nullptr);

    void load_filter_store(const std::string &path) {
      filter_store_ = std::make_unique<FilterStore>();
      filter_store_->load(path);
      LOG(INFO) << "Filter store: " << filter_store_->size() << " nodes";
    }

    void load_spmat_filter_store(const std::string &path) {
      spmat_filter_ = std::make_unique<SpmatFilterStore>();
      spmat_filter_->load(path);
    }

    void load_nbr_store(uint32_t max_nbrs);

    void enable_tunneling() {
      if (!nbr_store_ || (!filter_store_ && !spmat_filter_)) {
        LOG(ERROR) << "Tunneling needs a filter store and a neighbor store.";
        crash();
      }
      tunneling_ = true;
    }

    bool check_filter(uint32_t id, const void *filter_data) const {
      if (spmat_filter_) {
        return spmat_filter_->passes(id, filter_data);
      }
      if (filter_store_ && filter_data != nullptr) {
        return filter_store_->passes(id, *(const uint8_t *) filter_data);
      }
      return true;
    }

   private:
    // Derived/runtime metadata not stored in SSDIndexMetadata.
    uint64_t aligned_dim = 0;
    uint64_t size_per_io = 0;

    // File reader and index file path.
    std::shared_ptr<AlignedFileReader> &reader;
    std::string disk_index_file;

    // Neighbor handler.
    AbstractNeighbor<T> *nbr_handler;

    // Distance comparator.
    pipeann::Metric metric;
    std::shared_ptr<Distance<T>> dist_cmp;

    IndexBuildParameters params;

    // Thread-specific scratch buffers.
    ConcurrentQueue<QueryBuffer<T> *> thread_data_queue;
    std::vector<QueryBuffer<T> *> thread_data_bufs;  // pre-allocated thread data
    uint64_t max_nthreads;

    // Locking tables for concurrency control.
    pipeann::SparseLockTable<uint64_t> idx_lock_table;
    std::shared_mutex merge_lock;

    // ID to location mapping.
    // Concurrency control is done in lock_idx.
    // Only resize should be protected.
    std::vector<uint32_t> id2loc_;
    pipeann::ReaderOptSharedMutex id2loc_resize_mu_;

    // Location to ID mapping.
    // If nnodes_per_sector >= 1, page_layout[i * nnodes_per_sector + j] is the id of the j-th node in the i-th page.
    // ElseIf nnodes_per_sector == 0, page_layout[i] is the id of the i-th node (starting from loc_sector_no(i)).
    std::vector<uint32_t> loc2id_;

    // Tag support.
    // If ID == tag, then it is not stored.
    libcuckoo::cuckoohash_map<uint32_t, TagT> tags;

    // Flags.
    bool load_flag = false;    // already loaded.
    bool enable_tags = false;

    std::unique_ptr<FilterStore> filter_store_;
    std::unique_ptr<SpmatFilterStore> spmat_filter_;
    std::unique_ptr<NbrStore> nbr_store_;
    bool tunneling_ = false;

    void init_buffers(uint64_t nthreads);
    void destroy_buffers();

    // Load id2loc and loc2id (i.e., page_layout), to support index reordering.
    void load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector = 0,
                          const uint64_t num_points = 0);

    void load_tags(const std::string &tag_file, size_t offset = 0);

    void do_beam_search(const T *vec, uint32_t mem_L, uint32_t Lsize, const uint32_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, QueryStats *stats = nullptr);

    // ID, loc, page mapping.
    TagT id2tag(uint32_t id);

    uint32_t id2loc(uint32_t id);

    std::vector<uint32_t> get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors);

    std::vector<uint32_t> lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock);
  };
}  // namespace pipeann
