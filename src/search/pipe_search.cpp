#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#ifndef USE_AIO
#include "liburing.h"
#endif

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

namespace pipeann {
  struct io_t {
    Neighbor nbr;
    unsigned page_id;
    unsigned loc;
    IORequest *read_req;
    bool operator>(const io_t &rhs) const {
      return nbr.distance > rhs.nbr.distance;
    }

    bool operator<(const io_t &rhs) const {
      return nbr.distance < rhs.nbr.distance;
    }

    bool finished() {
      return read_req->finished;
    }
  };

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pipe_search(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT *res_tags, float *distances,
                                        const uint64_t beam_width, QueryStats *stats, AbstractSelector *selector,
                                        const void *filter_data) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to current vector for comparison
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer;
    std::vector<Neighbor> retset(mem_L + this->params.R + l_search * 10);
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(l_search * 10);

#ifndef OVERLAP_INIT
    nbr_handler->initialize_query(query, query_buf);
#endif

    auto compute_exact_dists_and_push = [&](const DiskNode<T> &node, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node.coords, meta_.data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      // Only add to full_retset if no filter or filter condition is satisfied.
      if (selector == nullptr || selector->is_member(id, filter_data, node.labels)) {
        full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      }
      return cur_expanded_dist;
    };

    uint64_t n_computes = 0;
    auto compute_and_push_nbrs = [&](DiskNode<T> &node, unsigned &nk) {
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < node.nnbrs; ++m) {
        if (visited.find(node.nbrs[m]) == visited.end()) {
          node.nbrs[nbors_cand_size++] = node.nbrs[m];
          visited.insert(node.nbrs[m]);
        }
      }

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        nbr_handler->compute_dists(query_buf, node.nbrs, nbors_cand_size);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node.nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }

          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          if (cur_list_size < l_search) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          if (r < nk)
            nk = r;
        }
      }
    };

    auto add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids, float *dists) {
      for (uint64_t i = 0; i < n_ids; ++i) {
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    if (stats != nullptr) {
      stats->io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
    }
    // search in in-memory index.

    int64_t cur_beam_width = std::min(4ul, beam_width);  // before converge.
    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

#ifdef OVERLAP_INIT
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      add_to_retset(mem_tags.data(), std::min((uint64_t) mem_L, l_search), mem_dists.data());
    } else {
      // cannot overlap.
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
      add_to_retset(&meta_.entry_point_id, 1, dist_scratch);
    }
#else
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      nbr_handler->compute_dists(query_buf, mem_tags.data(), mem_L);
      add_to_retset(mem_tags.data(), std::min((uint64_t) mem_L, l_search), dist_scratch);
    } else {
      nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
      add_to_retset(&meta_.entry_point_id, 1, dist_scratch);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);
#endif

    const bool tunnel = this->tunneling_ && filter_data != nullptr;

    std::queue<io_t> on_flight_ios;
    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;

      // lock the corresponding page.
      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
      const unsigned loc = id2loc(item.id), pid = loc_sector_no(loc);

      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), meta_.max_node_len,
                      sector_scratch);
      reader->send_read_no_alloc(req, ctx);

      on_flight_ios.push(io_t{item, pid, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };

    auto tunnel_through = [&](unsigned &marker) {
      auto st = std::chrono::high_resolution_clock::now();
      retset[marker].flag = false;
      retset[marker].visited = true;

      uint32_t nnbrs;
      const uint32_t *nbrs;
      this->nbr_store_->get(retset[marker].id, nnbrs, nbrs);
      uint32_t cand[MAX_N_EDGES];
      unsigned n_cand = 0;
      for (unsigned m = 0; m < nnbrs; m++) {
        if (visited.find(nbrs[m]) == visited.end()) {
          cand[n_cand++] = nbrs[m];
          visited.insert(nbrs[m]);
        }
      }
      if (n_cand > 0) {
        nbr_handler->compute_dists(query_buf, cand, n_cand);
        for (unsigned m = 0; m < n_cand; m++) {
          float d = dist_scratch[m];
          if (cur_list_size < l_search || d < retset[cur_list_size - 1].distance) {
            Neighbor nn(cand[m], d, true);
            auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
            if (cur_list_size < l_search)
              ++cur_list_size;
            if (r < marker)
              marker = r;
          }
        }
      }

      if (stats != nullptr) {
        stats->n_tunnels++;
        auto ed = std::chrono::high_resolution_clock::now();
        stats->bd_tunnel_us += std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st).count() / 1000.0;
      }
    };

    std::unordered_map<unsigned, DiskNode<T>> id_buf_map;
    auto poll_all = [&]() -> std::pair<int, int> {
      auto st = std::chrono::high_resolution_clock::now();
      // poll once.
      reader->poll_all(ctx);
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        id_buf_map.insert(std::make_pair(io.nbr.id, node_from_page((char *) io.read_req->buf, io.loc)));
        io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;
        // unlock the corresponding page.
        this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
      }
      if (stats != nullptr) {
        auto ed = std::chrono::high_resolution_clock::now();
        stats->bd_io_us += std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st).count() / 1000.0;
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0, marker = 0;
      while (marker < cur_list_size && n_sent < n) {
        while (marker < cur_list_size /* pool size */ &&
               (retset[marker].flag == false /* on flight */ ||
                id_buf_map.find(retset[marker].id) != id_buf_map.end() /* already read */)) {
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          ++marker;
        }
        if (marker >= cur_list_size) {
          break;  // nothing to send.
        }
        if (tunnel && !this->check_filter(retset[marker].id, filter_data)) {
          tunnel_through(marker);
          continue;
        }
        n_sent += send_read_req(retset[marker]);
      }
      return n_sent != 0;  // nothing to send.
    };

    auto calc_best_node = [&]() -> int {  // if converged.
      auto st = std::chrono::high_resolution_clock::now();
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      for (marker = 0; marker < cur_list_size; ++marker) {
        if (!retset[marker].visited && id_buf_map.find(retset[marker].id) != id_buf_map.end()) {
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          retset[marker].visited = true;
          auto it = id_buf_map.find(retset[marker].id);
          auto &[id, node] = *it;
          compute_exact_dists_and_push(node, id);
          compute_and_push_nbrs(node, nk);
          break;
        }
      }

      /* guess the first unvisited vector (eager) */
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && retset[i].flag /* not on-fly */
            && id_buf_map.find(retset[i].id) == id_buf_map.end() /* not already read */) {
          first_unvisited_eager = i;
          break;
        }
      }
      if (stats != nullptr) {
        auto ed = std::chrono::high_resolution_clock::now();
        stats->bd_process_us += std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st).count() / 1000.0;
      }
      return first_unvisited_eager;
    };

    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };

    auto cpu2_st = std::chrono::high_resolution_clock::now();
#ifdef OVERLAP_INIT
    if (this->tunneling_ && mem_L != 0) {
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, mem_tags.data(), mem_L);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
#endif
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned marker = 0, max_marker = 0;
#ifdef OVERLAP_INIT
    if (likely(mem_L != 0) && !this->tunneling_) {
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, mem_tags.data(), mem_L);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
#endif

    int cur_n_in = 0, cur_tot = 0;
    while (get_first_unvisited() != -1) {
      // poll to heap (best-effort) -> calc best from heap (skip if heap is empty) -> send IO (if can send) -> ...
      auto [n_in, n_out] = poll_all();

      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        // converged, tune beam width.
        constexpr double kWasteThreshold = 0.1;  // 0.1 * 10
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }
      }

      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
        send_best_read_req(1);
      }
      marker = calc_best_node();
      max_marker = std::max(max_marker, marker);
    }

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
    }
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // copy k_search values
    uint64_t t = 0;
    for (uint64_t i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;  // deduplicate.
      }
      res_tags[t] = id2tag(full_retset[i].id);
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
      }
      t++;
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
