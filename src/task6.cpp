
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr int THREAD_COUNT = 4;
constexpr uint64_t INCREMENTS_PER_THREAD = 2'000'000;
constexpr uint64_t EXPECTED =
    THREAD_COUNT * INCREMENTS_PER_THREAD;

// Deliberately unsafe demonstration.
// volatile does NOT make concurrent access thread-safe.
volatile uint64_t unsafe_counter = 0;

double elapsed_ms(Clock::time_point start, Clock::time_point finish) {
    return std::chrono::duration<double, std::milli>(
        finish - start
    ).count();
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Task 6: Synchronization Cost & Data Loss\n";
    std::cout << "Threads: " << THREAD_COUNT << '\n';
    std::cout << "Increments per thread: "
              << INCREMENTS_PER_THREAD << '\n';
    std::cout << "Expected final value: " << EXPECTED << "\n\n";

    // Version A: unsynchronized shared counter.
    unsafe_counter = 0;

    auto start = Clock::now();
    std::vector<std::thread> unsafe_threads;
    unsafe_threads.reserve(THREAD_COUNT);

    for (int t = 0; t < THREAD_COUNT; ++t) {
        unsafe_threads.emplace_back([]() {
            for (uint64_t i = 0; i < INCREMENTS_PER_THREAD; ++i) {
                ++unsafe_counter;
            }
        });
    }

    for (auto& thread : unsafe_threads) {
        thread.join();
    }

    auto finish = Clock::now();

    const double unsafe_ms = elapsed_ms(start, finish);
    const uint64_t actual = unsafe_counter;
    const uint64_t lost = actual <= EXPECTED ? EXPECTED - actual : 0;
    const double lost_percent =
        100.0 * static_cast<double>(lost) / EXPECTED;

    std::cout << "Version A: Unsynchronized counter\n";
    std::cout << "Actual final value: " << actual << '\n';
    std::cout << "Updates lost (estimated): " << lost << '\n';
    std::cout << "Lost percentage (estimated): "
              << lost_percent << "%\n";
    std::cout << "Wall-clock time: " << unsafe_ms << " ms\n\n";

    // Version B: correct atomic counter.
    std::atomic<uint64_t> atomic_counter{0};

    start = Clock::now();
    std::vector<std::thread> atomic_threads;
    atomic_threads.reserve(THREAD_COUNT);

    for (int t = 0; t < THREAD_COUNT; ++t) {
        atomic_threads.emplace_back([&]() {
            for (uint64_t i = 0; i < INCREMENTS_PER_THREAD; ++i) {
                atomic_counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& thread : atomic_threads) {
        thread.join();
    }

    finish = Clock::now();

    const double atomic_ms = elapsed_ms(start, finish);
    const uint64_t atomic_actual = atomic_counter.load();

    std::cout << "Version B: Atomic counter\n";
    std::cout << "Actual final value: " << atomic_actual << '\n';
    std::cout << "Updates lost: "
              << (atomic_actual == EXPECTED ? 0 : EXPECTED - atomic_actual)
              << '\n';
    std::cout << "Wall-clock time: " << atomic_ms << " ms\n\n";

    std::cout << "Synchronization timing comparison\n";
    std::cout << "Atomic / unsynchronized time ratio: ";

    if (unsafe_ms > 0.0) {
        std::cout << atomic_ms / unsafe_ms << "x\n";
    } else {
        std::cout << "N/A\n";
    }

    std::cout << "\nNote: the unsynchronized version has a C++ data race "
                 "and therefore undefined behavior. Its lost-update "
                 "count is an illustrative observation, not a portable "
                 "guarantee. The atomic version is thread-safe.\n";

    return atomic_actual == EXPECTED ? 0 : 1;
}
