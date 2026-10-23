
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr int TASK_COUNT = 1000;
constexpr int WORKER_COUNT = 4;

using TimePoint = Clock::time_point;

struct Result {
    double total_ms = 0.0;
    double completion_delta_ms = 0.0;
    std::array<double, WORKER_COUNT> worker_ms{};
    uint64_t checksum = 0;
};

// Simulate an irregular task using the workload specified in the assignment.
// Each task performs approximately 0, 0.1, ... 1.9 ms of work.
void do_work(int index, uint64_t& checksum) {
    const auto delay = std::chrono::microseconds(
        100 * (index % 20)
    );

    const auto start = Clock::now();

    // Sleep may last longer than requested because of OS scheduling.
    std::this_thread::sleep_for(delay);

    // Keep a small deterministic result for correctness checking.
    checksum += static_cast<uint64_t>(index + 1);
    const volatile double value = std::sin(index * 0.001);
    (void)value;

    (void)start;
}

Result run_static() {
    std::array<uint64_t, WORKER_COUNT> sums{};
    std::array<TimePoint, WORKER_COUNT> finished{};
    std::vector<std::thread> threads;

    const auto start = Clock::now();

    for (int worker = 0; worker < WORKER_COUNT; ++worker) {
        threads.emplace_back([&, worker]() {
            const int begin = worker * TASK_COUNT / WORKER_COUNT;
            const int end = (worker + 1) * TASK_COUNT / WORKER_COUNT;

            for (int i = begin; i < end; ++i) {
                do_work(i, sums[worker]);
            }

            finished[worker] = Clock::now();
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    const auto end = Clock::now();

    const auto earliest = *std::min_element(finished.begin(), finished.end());
    const auto latest = *std::max_element(finished.begin(), finished.end());

    Result result;
    result.total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    result.completion_delta_ms =
        std::chrono::duration<double, std::milli>(latest - earliest).count();

    for (int i = 0; i < WORKER_COUNT; ++i) {
        result.worker_ms[i] =
            std::chrono::duration<double, std::milli>(
                finished[i] - start
            ).count();
        result.checksum += sums[i];
    }

    return result;
}

Result run_dynamic(int chunk_size) {
    std::array<uint64_t, WORKER_COUNT> sums{};
    std::array<TimePoint, WORKER_COUNT> finished{};
    std::atomic<int> next_task{0};
    std::vector<std::thread> threads;

    const auto start = Clock::now();

    for (int worker = 0; worker < WORKER_COUNT; ++worker) {
        threads.emplace_back([&, worker]() {
            while (true) {
                const int begin =
                    next_task.fetch_add(chunk_size, std::memory_order_relaxed);

                if (begin >= TASK_COUNT) {
                    break;
                }

                const int end = std::min(begin + chunk_size, TASK_COUNT);

                for (int i = begin; i < end; ++i) {
                    do_work(i, sums[worker]);
                }
            }

            finished[worker] = Clock::now();
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    const auto end = Clock::now();

    const auto earliest = *std::min_element(finished.begin(), finished.end());
    const auto latest = *std::max_element(finished.begin(), finished.end());

    Result result;
    result.total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    result.completion_delta_ms =
        std::chrono::duration<double, std::milli>(latest - earliest).count();

    for (int i = 0; i < WORKER_COUNT; ++i) {
        result.worker_ms[i] =
            std::chrono::duration<double, std::milli>(
                finished[i] - start
            ).count();
        result.checksum += sums[i];
    }

    return result;
}

void print_result(const char* name, const Result& result) {
    std::cout << name << '\n';
    std::cout << "Total execution time: "
              << result.total_ms << " ms\n";
    std::cout << "Thread completion delta: "
              << result.completion_delta_ms << " ms\n";
    std::cout << "Worker completion times (from start): ";

    for (int i = 0; i < WORKER_COUNT; ++i) {
        std::cout << result.worker_ms[i];

        if (i + 1 < WORKER_COUNT) {
            std::cout << ", ";
        }
    }

    std::cout << " ms\n";
    std::cout << "Checksum: " << result.checksum << "\n\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Task 4: Work Stealing & Scheduling Under Skew\n";
    std::cout << "Tasks: " << TASK_COUNT
              << ", workers: " << WORKER_COUNT << "\n\n";

    const Result static_result = run_static();
    const Result dynamic_1 = run_dynamic(1);
    const Result dynamic_5 = run_dynamic(5);

    print_result("Strategy A: Static equal blocks", static_result);
    print_result("Strategy B: Dynamic scheduling, chunk size 1", dynamic_1);
    print_result("Strategy C: Dynamic scheduling, chunk size 5", dynamic_5);

    const uint64_t expected =
        static_cast<uint64_t>(TASK_COUNT) * (TASK_COUNT + 1) / 2;

    std::cout << "Expected checksum: " << expected << '\n';
    std::cout << "All checksums match: "
              << ((static_result.checksum == expected &&
                   dynamic_1.checksum == expected &&
                   dynamic_5.checksum == expected)
                      ? "YES" : "NO")
              << '\n';

    return 0;
}
