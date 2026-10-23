
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr double A = 2.5;
constexpr double B = 1.25;
constexpr int REPEATS = 5;

const std::vector<std::size_t> SIZES = {
    100,
    1'000,
    10'000,
    100'000,
    1'000'000,
    10'000'000
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void serial_operation(
    const std::vector<double>& x,
    std::vector<double>& y
) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = A * x[i] + B;
    }
}

void parallel_operation(
    const std::vector<double>& x,
    std::vector<double>& y,
    unsigned int workers
) {
    const std::size_t n = x.size();
    workers = std::max(
        1u,
        std::min(workers, static_cast<unsigned int>(n))
    );

    std::vector<std::thread> threads;
    threads.reserve(workers);

    const std::size_t chunk = (n + workers - 1) / workers;

    for (unsigned int t = 0; t < workers; ++t) {
        const std::size_t begin = t * chunk;
        const std::size_t end = std::min(begin + chunk, n);

        if (begin >= n) {
            break;
        }

        threads.emplace_back([&, begin, end]() {
            for (std::size_t i = begin; i < end; ++i) {
                y[i] = A * x[i] + B;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename Function>
double measure_ms(Function function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();

    return std::chrono::duration<double, std::milli>(
        finish - start
    ).count();
}

struct BenchmarkResult {
    std::size_t n;
    double serial_ms;
    double parallel_ms;
    double speedup;
    bool parallel_faster;
};

int main() {
    const unsigned int workers = std::max(
        1u,
        std::thread::hardware_concurrency()
    );

    std::cout << "Task 5: The Overhead Crossover Point\n";
    std::cout << "Available logical threads: " << workers << '\n';
    std::cout << "Repeats per measurement: " << REPEATS << '\n';
    std::cout << "Operation: y[i] = a * x[i] + b\n\n";

    std::cout << std::fixed << std::setprecision(4);

    std::vector<BenchmarkResult> results;

    for (const std::size_t n : SIZES) {
        std::vector<double> x(n);
        std::vector<double> y(n, 0.0);

        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i % 1000) / 1000.0;
        }

        // Warm up both implementations before timing.
        serial_operation(x, y);
        parallel_operation(x, y, workers);

        std::vector<double> serial_times;
        std::vector<double> parallel_times;
        serial_times.reserve(REPEATS);
        parallel_times.reserve(REPEATS);

        for (int r = 0; r < REPEATS; ++r) {
            serial_times.push_back(
                measure_ms([&]() {
                    serial_operation(x, y);
                })
            );

            parallel_times.push_back(
                measure_ms([&]() {
                    parallel_operation(x, y, workers);
                })
            );
        }

        const double serial_ms = median(serial_times);
        const double parallel_ms = median(parallel_times);
        const double speedup =
            parallel_ms > 0.0 ? serial_ms / parallel_ms : 0.0;

        results.push_back({
            n,
            serial_ms,
            parallel_ms,
            speedup,
            parallel_ms < serial_ms
        });

        std::cout << "N = " << n << '\n';
        std::cout << "  Serial median:   " << serial_ms << " ms\n";
        std::cout << "  Parallel median: " << parallel_ms << " ms\n";
        std::cout << "  Speedup:         " << speedup << "x\n";
        std::cout << "  Faster:          "
                  << (parallel_ms < serial_ms ? "PARALLEL" : "SERIAL")
                  << "\n\n";
    }

    std::cout << "=== Crossover analysis ===\n";

    bool found = false;

    for (std::size_t i = 1; i < results.size(); ++i) {
        if (!results[i - 1].parallel_faster &&
            results[i].parallel_faster) {
            std::cout << "Observed crossover between N = "
                      << results[i - 1].n
                      << " and N = " << results[i].n << ".\n";
            found = true;
        }
    }

    if (!found) {
        const auto first_faster = std::find_if(
            results.begin(),
            results.end(),
            [](const BenchmarkResult& r) {
                return r.parallel_faster;
            }
        );

        if (first_faster == results.end()) {
            std::cout
                << "No crossover to faster parallel execution was "
                   "observed in the tested sizes.\n";
        } else if (first_faster == results.begin()) {
            std::cout
                << "Parallel execution was already faster at the "
                   "smallest tested size.\n";
        } else {
            std::cout
                << "Parallel execution first became faster at N = "
                << first_faster->n << ".\n";
        }
    }

    std::cout
        << "Note: crossover is determined from median timings at "
           "the tested sizes; it is not an exact threshold for every N.\n";

    return 0;
}
