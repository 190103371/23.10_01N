
import os
import time
import platform
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 190103371
ARRAY_SIZE = 20_000_000
PARALLEL_FRACTION = 0.95

# Shared data accessible by worker processes
_shared_memory = None
_shared_array = None


def initialize_worker(shm_name, array_size):
    """Attach each worker process to the shared array."""
    global _shared_memory, _shared_array

    _shared_memory = shared_memory.SharedMemory(name=shm_name)
    _shared_array = np.ndarray(
        (array_size,),
        dtype=np.float64,
        buffer=_shared_memory.buf
    )


def sum_chunk(bounds):
    """Sum one assigned section of the array."""
    start, end = bounds
    return float(np.sum(_shared_array[start:end], dtype=np.float64))


def amdahl_speedup(workers):
    """Theoretical speedup according to Amdahl's Law."""
    p = PARALLEL_FRACTION
    return 1.0 / ((1.0 - p) + p / workers)


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted(set([
        1, 2, 4, max_cores, max_cores * 2
    ]))

    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"Logical CPU Cores: {max_cores}")
    print(f"Array Size: {ARRAY_SIZE}")
    print(f"Parallel Fraction (P): {PARALLEL_FRACTION:.2f}")
    print()

    # Generate deterministic input outside the benchmark timer.
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    # Shared memory avoids copying the 160 MB array to every process.
    shm = shared_memory.SharedMemory(
        create=True,
        size=data.nbytes
    )

    shared_array = np.ndarray(
        data.shape,
        dtype=np.float64,
        buffer=shm.buf
    )
    shared_array[:] = data

    results = []

    try:
        for workers in worker_counts:
            # Divide the array into non-overlapping sections.
            boundaries = np.linspace(
                0, ARRAY_SIZE, workers + 1, dtype=np.int64
            )
            chunks = [
                (int(boundaries[i]), int(boundaries[i + 1]))
                for i in range(workers)
            ]

            # Pool startup is excluded from the timed interval.
            with mp.Pool(
                    processes=workers,
                    initializer=initialize_worker,
                    initargs=(shm.name, ARRAY_SIZE)
            ) as pool:
                start = time.perf_counter()

                partial_sums = pool.map(sum_chunk, chunks)
                total = sum(partial_sums)

                elapsed_ms = (
                                     time.perf_counter() - start
                             ) * 1000.0

            results.append({
                "workers": workers,
                "time_ms": elapsed_ms,
                "sum": total
            })

    finally:
        shm.close()
        shm.unlink()

    # Use the measured one-worker run as the baseline.
    baseline_ms = next(
        result["time_ms"]
        for result in results
        if result["workers"] == 1
    )

    print(
        f"{'Cores':>8} | {'Wall Time (ms)':>16} | "
        f"{'Speedup':>10} | {'Amdahl Upper Bound':>19}"
    )
    print("-" * 65)

    for result in results:
        workers = result["workers"]
        elapsed_ms = result["time_ms"]
        speedup = baseline_ms / elapsed_ms
        theoretical = amdahl_speedup(workers)

        print(
            f"{workers:8d} | {elapsed_ms:16.3f} | "
            f"{speedup:10.3f} | {theoretical:19.3f}"
        )

    print()
    print(f"Serial baseline sum: {results[0]['sum']:.10f}")
    print(f"Baseline time (ms): {baseline_ms:.3f}")
    print("Benchmark completed.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
