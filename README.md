# NemoANN

NemoANN is an SSD-based graph index for filtered approximate nearest neighbor search.
It checks the query's filter in memory before it reads a candidate from the SSD.
A candidate that passes is read and scored as usual.
A candidate that fails is never read, but the search still moves through it by using an in-memory copy of its neighbor list and the PQ distances that are already in memory.
We call this graph tunneling.
The graph index and its on-disk layout are not modified, and the predicate is given at query time.

This repository contains the implementation and the experiment scripts for the paper

> Nakyung Lee, Soobin Cho, Jiwoong Park, and Gyuyeong Kim.
> **NemoANN: I/O-Efficient Filtered Vector Search on SSDs with Graph Tunneling.**
> ACM Symposium on Cloud Computing (SoCC '26), 2026.

NemoANN is built on [PipeANN](https://github.com/thustorage/PipeANN) (commit `6a0403c`).
An earlier version of this work appeared on arXiv under the name GateANN.

## Key results

On BigANN-100M at 10% selectivity and 90% Recall@10 with 32 threads:

- NemoANN issues about 10x fewer SSD reads than post-filtering.
- Its throughput is 8.0x that of PipeANN with post-filtering and 5.0x that of PipeANN-Filter.
- At 5% selectivity, the gain over PipeANN grows to 13.1x.
- It keeps the original index, which is 14x smaller on SSD than the densified index of PipeANN-Filter.

On BigANN-1B, NemoANN reaches 9.7K QPS against 1.7K QPS for PipeANN (5.7x).

## Repository layout

| Path | Contents |
|------|----------|
| `include/`, `src/` | PipeANN core and NemoANN |
| `tests/` | Index build and search binaries |
| `scripts/` | Build, data preparation, and experiment scripts |
| `patches/` | Fix applied to the official DiskANN for the Filtered-DiskANN baseline |
| `third_party/liburing` | liburing 2.10 (git submodule) |

The NemoANN code is in the following files.

| File | Role |
|------|------|
| `include/utils/filter_store.h` | Single-label filter store |
| `include/utils/spmat_filter_store.h` | Multi-label filter store (a node passes if it has every query label) |
| `include/utils/nbr_store.h` | Neighbor store |
| `src/search/pipe_search.cpp` | Filter check before SSD submission and graph tunneling |
| `src/ssd_index.cpp` | Neighbor store construction (`load_nbr_store`) |

## Requirements

The paper's testbed is a server with two Intel Xeon Silver 4514Y CPUs (32 cores), 256 GB of DDR5 memory, and a Samsung PM9A3 2 TB NVMe SSD, running Ubuntu 22.04 with Linux 5.15.

NemoANN needs Linux with `io_uring`, GCC 9 or later, CMake 3.13 or later, a BLAS library (MKL or OpenBLAS), tcmalloc, and Python 3 with `numpy` and `faiss-cpu`.
On Ubuntu 22.04, `scripts/install_deps.sh` installs everything, including the packages needed to build the two baselines.

| Dataset | Disk index | Neighbor store (R_max = 32) |
|---------|-----------:|----------------------------:|
| BigANN-100M | 58.5 GB | 12.3 GB |
| DEEP-100M | 82 GB | 12.3 GB |
| BigANN-1B | 683 GB | 123 GB |

The neighbor store comes on top of the 32-byte PQ codes that PipeANN already keeps in memory (3.2 GB per 100M vectors).

## Build

```bash
git clone --recursive <repository URL> nemoann
cd nemoann
./scripts/install_deps.sh
./scripts/build.sh
```

`build.sh` builds liburing and then the binaries in `build/tests`.

## Datasets

```bash
./scripts/prepare.sh bigann100M
./scripts/prepare.sh deep100M
./scripts/prepare.sh bigann1B
```

For each dataset, `prepare.sh` performs four steps.

1. It downloads the first N vectors of the BigANN or DEEP base set and the public 10K query set.
2. It builds the Vamana disk index with 32-byte PQ (R = 96 and L = 128 for the 100M datasets, R = 128 and L = 200 for BigANN-1B).
3. It builds PipeANN's in-memory navigation graph from a 1% sample.
4. It generates labels and the exact filtered ground truth with `scripts/make_labels.py`.

Each base vector receives one of C labels uniformly at random, where C = 100 / selectivity (in percent), and each query receives one label in the same way.
The random seeds are fixed, so the label files are identical to the ones used in the paper.
The ground truth holds, for every query, the 10 nearest base vectors by exact L2 distance among the vectors that carry the query's label.
BigANN-100M gets labels for 5%, 10%, and 20% selectivity, and the other datasets get 10%.

Data goes to `data/` and results go to `results/`.
Set `NEMO_DATA` and `NEMO_RESULTS` to use other locations, and `NEMO_BUILD_MEM` to change the memory budget of index construction (default 200 GB).

## Running a search

```bash
build/tests/search_disk_index <uint8|int8|float> <index_prefix> <threads> <W> <query.bin> \
    <base_labels> <query_labels> <gt.bin> <K> <diskann|pipeann|nemoann> <mem_L> <R_max> <L1> [L2 ...]
```

- `diskann` runs synchronous beam search, `pipeann` runs PipeANN's pipelined search, and both filter the results afterwards.
  `nemoann` runs PipeANN's pipelined search with the filter check and graph tunneling.
- Labels are either `.bin` files with one `uint8` label per vector or `.spmat` files with a set of labels per vector.
- `R_max` is the number of neighbors per node in the neighbor store and is used only by `nemoann`.
  The store is built from the disk index on first use and saved as `<index_prefix>_nbr_store_R<R_max>.bin`.

For every L, the binary prints the throughput, the mean latency (threads divided by throughput), Recall@K, the mean number of SSD reads and tunneled nodes per query, a time breakdown, and the resident memory.

```bash
build/tests/search_disk_index uint8 data/bigann100M/index/bigann100M 32 32 data/bigann100M/query.u8bin \
    data/bigann100M/labels/sel10_base.bin data/bigann100M/labels/sel10_query.bin \
    data/bigann100M/labels/sel10_gt.bin 10 nemoann 10 32 200
```

`build/tests/search_mem_index` runs in-memory Vamana search with post-filtering on the same index.

## Reproducing the key results

| Script | Paper | What it runs |
|--------|-------|--------------|
| `run_main.sh` | Fig. 5, Fig. 7(a), Sec. 5.2.1, 5.2.3 | DiskANN, PipeANN, and NemoANN on BigANN-100M and DEEP-100M at 10% selectivity |
| `run_rmax.sh` | Fig. 6, Sec. 5.2.2 | NemoANN with R_max of 4, 8, 16, 32, 48, and 64 |
| `run_billion.sh` | Fig. 8, Sec. 5.2.4 | DiskANN, PipeANN, and NemoANN on BigANN-1B |
| `run_vamana.sh` | Fig. 9, Sec. 5.2.5 | In-memory Vamana with post-filtering |
| `run_fdiskann.sh` | Fig. 10, Sec. 5.2.6 | Filtered-DiskANN |
| `run_selectivity.sh` | Fig. 11, Fig. 7(b), Sec. 5.3.1 | PipeANN and NemoANN on BigANN-100M at 5%, 10%, and 20% selectivity |
| `run_pipeann_filter.sh` | Figs. 5, 6, 7(a), 10, 11 | PipeANN-Filter |

Every script sweeps the search list size L with 1 and 32 threads, appends one row per point to `results/<script>.tsv`, and keeps the raw output in `results/logs/`.
Set `THREADS` to change the thread counts.
The summary script reports throughput and SSD reads at a given recall, interpolated along each L sweep, together with NemoANN's speedup.

```bash
./scripts/summarize.py --recall 0.9
```

The experiments use the settings of the paper.
DiskANN uses W = 8, and PipeANN, NemoANN, and PipeANN-Filter use W = 32.
PipeANN and NemoANN start from the in-memory navigation graph (`mem_L = 10`), and NemoANN uses R_max = 32.
All runs use the 10K public queries and report Recall@10.
The scripts run each point of DiskANN, PipeANN, NemoANN, and PipeANN-Filter in a new process, and sweep L in one process for Vamana and Filtered-DiskANN.

### Baselines

`run_pipeann_filter.sh` builds PipeANN-Filter from the upstream [PipeANN](https://github.com/thustorage/PipeANN) repository (commit `6c7d9da` by default).
The index uses R = 96, R_dense = 1100, L = 128, and 32-byte PQ.
PipeANN-Filter stores labels inside each 8 KB record, so every label set needs its own index, which takes about 820 GB for a 100M dataset.
It runs without an in-memory navigation graph (`mem_L = 0`).
Set `SELS` to choose the selectivities.

`run_fdiskann.sh` builds the official [DiskANN](https://github.com/microsoft/DiskANN) at commit `f198a8a` with `patches/diskann-label-parse.patch`.
The patch replaces a scan in label-file parsing that stalls index loading for single-label files, and it does not change the search.
The FilteredVamana index uses R = 96 and a filtered build list of 100 for BigANN-100M and 128 for DEEP-100M.

## Citation

```bibtex
@inproceedings{nemoann,
  author    = {Lee, Nakyung and Cho, Soobin and Park, Jiwoong and Kim, Gyuyeong},
  title     = {{NemoANN}: {I/O}-Efficient Filtered Vector Search on {SSDs} with Graph Tunneling},
  booktitle = {Proceedings of the ACM Symposium on Cloud Computing (SoCC '26)},
  year      = {2026},
  address   = {Singapore},
  publisher = {ACM},
  doi       = {10.1145/3828161.3857852}
}
```

## License

NemoANN is released under the MIT License.
See `NOTICE.txt` for the PipeANN license and third-party notices.
