
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr std::size_t N = 30'000'000;
constexpr int REPEATS = 3;

struct Result {
    double milliseconds;
    double checksum;
};

template <typename Function>
Result run_parallel(int thread_count, Function function) {
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    auto start = Clock::now();

    for (int t = 0; t < thread_count; ++t) {
        std::size_t begin = N * t / thread_count;
        std::size_t end = N * (t + 1) / thread_count;

        threads.emplace_back([=, &function]() {
            function(begin, end);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto finish = Clock::now();

    // Checksum is calculated outside the timed section.
    // The benchmark kernels must not include reduction overhead.
    return {
        std::chrono::duration<double, std::milli>(
            finish - start).count(),
        0.0
    };
}

template <typename Function>
double median_time(int thread_count, Function function) {
    std::vector<double> times;

    for (int r = 0; r < REPEATS; ++r) {
        times.push_back(run_parallel(thread_count, function).milliseconds);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main() {
    std::cout << std::fixed << std::setprecision(3);

    const unsigned int detected = std::thread::hardware_concurrency();
    const int max_threads = detected == 0 ? 4 : static_cast<int>(detected);

    // Keep the vector sizes identical for both kernels.
    std::vector<double> A(N);
    std::vector<double> B(N);

    for (std::size_t i = 0; i < N; ++i) {
        A[i] = 0.1 + static_cast<double>(i % 1000) * 0.0001;
    }

    std::cout << "Task 7: Memory Bandwidth Saturation\n";
    std::cout << "Elements: " << N << '\n';
    std::cout << "Array size: "
              << (N * sizeof(double)) / (1024.0 * 1024.0)
              << " MiB per array\n";
    std::cout << "Detected logical CPUs: " << max_threads << "\n\n";

    std::vector<int> thread_counts = {1, 2};
    if (max_threads > 2) {
        thread_counts.push_back(max_threads);
    } else if (max_threads == 2) {
        thread_counts.push_back(2);
    }

    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(
        std::unique(thread_counts.begin(), thread_counts.end()),
        thread_counts.end()
    );

    // Kernel 1: memory-bound vector copy.
    std::cout << "Kernel 1: Memory-bound vector copy\n";
    std::cout << "Threads | Median ms | Speedup | Efficiency | GB/s\n";

    double copy_baseline = 0.0;

    for (int thread_count : thread_counts) {
        auto copy_kernel = [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                B[i] = A[i];
            }
        };

        double ms = median_time(thread_count, copy_kernel);

        if (thread_count == 1) {
            copy_baseline = ms;
        }

        double speedup = copy_baseline / ms;
        double efficiency = speedup / thread_count;
        // Copy reads N doubles and writes N doubles.
        double gb_per_second =
            (2.0 * N * sizeof(double)) / (ms / 1000.0) / 1e9;

        std::cout << std::setw(7) << thread_count << " | "
                  << std::setw(9) << ms << " | "
                  << std::setw(7) << speedup << " | "
                  << std::setw(10) << efficiency * 100.0 << "% | "
                  << std::setw(7) << gb_per_second << '\n';
    }

    double copy_checksum = 0.0;
    for (std::size_t i = 0; i < N; i += 1000) {
        copy_checksum += B[i];
    }
    std::cout << "Sample checksum: " << copy_checksum << "\n\n";

    // Kernel 2: compute-bound transcendental math.
    std::cout << "Kernel 2: Compute-bound trigonometric function\n";
    std::cout << "Threads | Median ms | Speedup | Efficiency\n";

    double compute_baseline = 0.0;

    for (int thread_count : thread_counts) {
        auto compute_kernel = [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                B[i] = std::sin(A[i]) * std::cos(A[i])
                     + std::tan(A[i]);
            }
        };

        double ms = median_time(thread_count, compute_kernel);

        if (thread_count == 1) {
            compute_baseline = ms;
        }

        double speedup = compute_baseline / ms;
        double efficiency = speedup / thread_count;

        std::cout << std::setw(7) << thread_count << " | "
                  << std::setw(9) << ms << " | "
                  << std::setw(7) << speedup << " | "
                  << std::setw(10) << efficiency * 100.0 << "%\n";
    }

    double compute_checksum = 0.0;
    for (std::size_t i = 0; i < N; i += 1000) {
        compute_checksum += B[i];
    }

    std::cout << "Sample checksum: " << compute_checksum << "\n\n";
    std::cout << "Interpretation:\n";
    std::cout << "- Copy reads and writes large arrays, so memory bandwidth "
                 "can limit scaling.\n";
    std::cout << "- Trigonometric operations require more computation per "
                 "element.\n";
    std::cout << "- Use the measured speedup and efficiency to identify "
                 "which kernel stops scaling first.\n";

    return 0;
}
