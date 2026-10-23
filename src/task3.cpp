
#include <windows.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <algorithm>

constexpr uint64_t ITERATIONS = 50'000'000;
constexpr size_t CACHE_LINE = 64;

// Counters stored next to each other in memory.
// Threads update different counters, but they may share cache lines.
struct UnpaddedCounter {
    volatile uint64_t value = 0;
};

// Each counter occupies its own cache line.
struct alignas(CACHE_LINE) PaddedCounter {
    volatile uint64_t value = 0;
};

unsigned int physical_core_count() {
    DWORD length = 0;

    GetLogicalProcessorInformationEx(
        RelationProcessorCore, nullptr, &length
    );

    if (length == 0) {
        return std::max(1u, std::thread::hardware_concurrency());
    }

    std::vector<char> buffer(length);

    auto info =
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            buffer.data()
        );

    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore, info, &length)) {
        return std::max(1u, std::thread::hardware_concurrency());
    }

    unsigned int cores = 0;
    DWORD offset = 0;

    while (offset < length) {
        auto entry =
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                buffer.data() + offset
            );

        if (entry->Relationship == RelationProcessorCore) {
            ++cores;
        }

        if (entry->Size == 0) {
            break;
        }

        offset += entry->Size;
    }

    return std::max(1u, cores);
}

template <typename Function>
double measure_ms(Function function) {
    auto start = std::chrono::steady_clock::now();

    function();

    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

int main() {
    const unsigned int workers = physical_core_count();

    std::cout << "Task 3: False Sharing Experiment\n";
    std::cout << "Physical CPU cores: " << workers << '\n';
    std::cout << "Iterations per thread: "
              << ITERATIONS << "\n\n";

    // Experiment A: adjacent counters.
    std::vector<UnpaddedCounter> shared_counts(workers);

    double unpadded_ms = measure_ms([&]() {
        std::vector<std::thread> threads;
        threads.reserve(workers);

        for (unsigned int i = 0; i < workers; ++i) {
            threads.emplace_back([&, i]() {
                for (uint64_t j = 0; j < ITERATIONS; ++j) {
                    ++shared_counts[i].value;
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    });

    uint64_t unpadded_total = 0;

    for (const auto& counter : shared_counts) {
        unpadded_total += counter.value;
    }

    // Experiment B: cache-line-aligned counters.
    std::vector<PaddedCounter> padded_counts(workers);

    double padded_ms = measure_ms([&]() {
        std::vector<std::thread> threads;
        threads.reserve(workers);

        for (unsigned int i = 0; i < workers; ++i) {
            threads.emplace_back([&, i]() {
                for (uint64_t j = 0; j < ITERATIONS; ++j) {
                    ++padded_counts[i].value;
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    });

    uint64_t padded_total = 0;

    for (const auto& counter : padded_counts) {
        padded_total += counter.value;
    }

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Experiment A: Unpadded counters\n";
    std::cout << "Time (ms): " << unpadded_ms << '\n';
    std::cout << "Total count: " << unpadded_total << "\n\n";

    std::cout << "Experiment B: Cache-line-padded counters\n";
    std::cout << "Time (ms): " << padded_ms << '\n';
    std::cout << "Total count: " << padded_total << "\n\n";

    std::cout << "Unpadded / padded time ratio: ";

    if (padded_ms > 0.0) {
        std::cout << unpadded_ms / padded_ms << "x\n";
    } else {
        std::cout << "N/A\n";
    }

    std::cout << "\nExpected count per experiment: "
              << ITERATIONS * static_cast<uint64_t>(workers)
              << '\n';

    std::cout << "\nNote: a ratio above 1 means the unpadded "
                 "version was slower in this run. Results depend "
                 "on CPU architecture and system load.\n";

    return 0;
}
