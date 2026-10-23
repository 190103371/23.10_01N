
import os
import platform
import time
import numpy as np

# Student information
STUDENT_ID = 190103371
ARRAY_SIZE = 20_000_000

def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU Model: {platform.processor() or platform.uname().processor or 'Not available'}")
    print(f"Logical CPU Cores: {os.cpu_count()}")

    # Start measuring the complete baseline execution
    start_time = time.perf_counter()

    # Generate 20 million float64 values using Student ID as seed
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    print(f"Array Size: {data.size}")
    print(f"Array Data Type: {data.dtype}")

    # Serial sum: one Python thread, no parallel reduction
    total = 0.0
    for value in data:
        total += float(value)

    # Population standard deviation, calculated serially
    mean = total / ARRAY_SIZE
    squared_deviations = 0.0

    for value in data:
        difference = float(value) - mean
        squared_deviations += difference * difference

    std_dev = (squared_deviations / ARRAY_SIZE) ** 0.5

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"Sum: {total:.10f}")
    print(f"Mean: {mean:.10f}")
    print(f"Standard Deviation: {std_dev:.10f}")
    print(f"Execution Time (ms): {elapsed_ms:.3f}")

if __name__ == "__main__":
    main()
