#!/usr/bin/env python3
import struct
import sys

import numpy as np


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = int(np.fromfile(src, dtype=np.uint32, count=1)[0])
    labels = np.memmap(src, dtype=np.uint8, mode="r", offset=4, shape=(n,))
    step = 10_000_000

    if dst.endswith(".spmat"):
        with open(dst, "wb") as f:
            f.write(struct.pack("qqq", n, int(labels.max()) + 1, n))
            for s in range(0, n + 1, step):
                f.write(np.arange(s, min(n + 1, s + step), dtype=np.int64).tobytes())
            for s in range(0, n, step):
                f.write(np.asarray(labels[s:s + step], dtype=np.int32).tobytes())
            for s in range(0, n, step):
                f.write(np.ones(min(step, n - s), dtype=np.float32).tobytes())
    elif dst.endswith(".txt"):
        with open(dst, "w") as f:
            for s in range(0, n, step):
                f.write("\n".join(map(str, labels[s:s + step].tolist())) + "\n")
    else:
        sys.exit("output must end with .spmat or .txt")


if __name__ == "__main__":
    main()
