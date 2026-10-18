#!/usr/bin/env python3
import argparse
import os
import struct
import time

import faiss
import numpy as np

DTYPES = {".u8bin": np.uint8, ".i8bin": np.int8, ".fbin": np.float32}


def open_vectors(path):
    dtype = DTYPES[os.path.splitext(path)[1]]
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(int(n), int(d)))


def write_labels(path, labels):
    with open(path, "wb") as f:
        f.write(struct.pack("I", len(labels)))
        f.write(labels.tobytes())


def merge(best_d, best_i, d, i, k):
    d = np.concatenate([best_d, d], axis=1)
    i = np.concatenate([best_i, i], axis=1)
    order = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, order, axis=1), np.take_along_axis(i, order, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("query")
    p.add_argument("out_prefix")
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--chunk", type=int, default=50_000_000)
    p.add_argument("--rng", choices=["pcg", "legacy"], default="pcg")
    args = p.parse_args()

    t0 = time.time()
    base = open_vectors(args.base)
    query = open_vectors(args.query)
    n, nq = base.shape[0], query.shape[0]

    if args.rng == "pcg":
        base_labels = np.random.default_rng(42).integers(0, args.classes, size=n, dtype=np.uint8)
        query_labels = np.random.default_rng(43).integers(0, args.classes, size=nq, dtype=np.uint8)
    else:
        np.random.seed(42)
        base_labels = np.random.randint(0, args.classes, size=n, dtype=np.uint8)
        query_labels = np.random.randint(0, args.classes, size=nq, dtype=np.uint8)
    write_labels(args.out_prefix + "_base.bin", base_labels)
    write_labels(args.out_prefix + "_query.bin", query_labels)
    print(f"labels: {n} base, {nq} query, {args.classes} classes", flush=True)

    faiss.omp_set_num_threads(os.cpu_count())
    xq = np.ascontiguousarray(query, dtype=np.float32)
    best_d = np.full((nq, args.k), np.inf, dtype=np.float32)
    best_i = np.full((nq, args.k), -1, dtype=np.int64)
    by_class = [np.flatnonzero(query_labels == c) for c in range(args.classes)]

    for start in range(0, n, args.chunk):
        end = min(n, start + args.chunk)
        chunk_labels = base_labels[start:end]
        for c, qsel in enumerate(by_class):
            if len(qsel) == 0:
                continue
            members = np.flatnonzero(chunk_labels == c)
            if len(members) == 0:
                continue
            xb = np.ascontiguousarray(base[start + members], dtype=np.float32)
            d, i = faiss.knn(xq[qsel], xb, min(args.k, len(members)))
            i = np.where(i >= 0, start + members[np.maximum(i, 0)], -1)
            if d.shape[1] < args.k:
                pad = args.k - d.shape[1]
                d = np.pad(d, ((0, 0), (0, pad)), constant_values=np.inf)
                i = np.pad(i, ((0, 0), (0, pad)), constant_values=-1)
            best_d[qsel], best_i[qsel] = merge(best_d[qsel], best_i[qsel], d, i, args.k)
        print(f"ground truth: {end}/{n} base vectors ({time.time() - t0:.0f} s)", flush=True)

    ids = np.where(best_i >= 0, best_i, 0xFFFFFFFF).astype(np.uint32)
    with open(args.out_prefix + "_gt.bin", "wb") as f:
        f.write(struct.pack("II", nq, args.k))
        f.write(ids.tobytes())
    print(f"wrote {args.out_prefix}_gt.bin ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
