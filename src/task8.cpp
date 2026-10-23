
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr int N = 1024;
constexpr int REPEATS = 3;

using Matrix = std::vector<double>;

void multiply_naive(
    const Matrix& A,
    const Matrix& B,
    Matrix& C
) {
    std::fill(C.begin(), C.end(), 0.0);

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            const double a = A[i * N + k];

            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void multiply_tiled(
    const Matrix& A,
    const Matrix& B,
    Matrix& C,
    int tile
) {
    std::fill(C.begin(), C.end(), 0.0);

    for (int ii = 0; ii < N; ii += tile) {
        for (int kk = 0; kk < N; kk += tile) {
            for (int jj = 0; jj < N; jj += tile) {
                const int i_end = std::min(ii + tile, N);
                const int k_end = std::min(kk + tile, N);
                const int j_end = std::min(jj + tile, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        const double a = A[i * N + k];

                        for (int j = jj; j < j_end; ++j) {
                            C[i * N + j] +=
                                a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

double checksum(const Matrix& M) {
    double sum = 0.0;

    for (double value : M) {
        sum += value;
    }

    return sum;
}

double max_difference(
    const Matrix& A,
    const Matrix& B
) {
    double difference = 0.0;

    for (std::size_t i = 0; i < A.size(); ++i) {
        difference = std::max(
            difference,
            std::abs(A[i] - B[i])
        );
    }

    return difference;
}

template <typename Function>
double benchmark(Function function) {
    std::vector<double> times;

    for (int r = 0; r < REPEATS; ++r) {
        auto start = Clock::now();
        function();
        auto finish = Clock::now();

        times.push_back(
            std::chrono::duration<double, std::milli>(
                finish - start
            ).count()
        );
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main() {
    std::cout << std::fixed << std::setprecision(3);

    Matrix A(N * N);
    Matrix B(N * N);
    Matrix C_naive(N * N);
    Matrix C16(N * N);
    Matrix C64(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] =
                static_cast<double>((i + j) % 100) / 100.0;

            B[i * N + j] =
                static_cast<double>((i * 3 + j * 7) % 100) / 100.0;
        }
    }

    std::cout << "Task 8: Compute Geometry - CPU Matrix Tiling\n";
    std::cout << "Matrix size: " << N << " x " << N << '\n';
    std::cout << "Repeats per implementation: "
              << REPEATS << "\n\n";

    std::cout << "Running naive implementation...\n";
    const double naive_ms = benchmark([&]() {
        multiply_naive(A, B, C_naive);
    });

    std::cout << "Running tiled 16x16 implementation...\n";
    const double tile16_ms = benchmark([&]() {
        multiply_tiled(A, B, C16, 16);
    });

    std::cout << "Running tiled 64x64 implementation...\n";
    const double tile64_ms = benchmark([&]() {
        multiply_tiled(A, B, C64, 64);
    });

    const double naive_checksum = checksum(C_naive);
    const double tile16_checksum = checksum(C16);
    const double tile64_checksum = checksum(C64);

    std::cout << "\nResults (median of " << REPEATS << " runs)\n";
    std::cout << "Implementation | Time (ms) | Speedup | Checksum\n";

    std::cout << "Naive          | "
              << naive_ms << " | "
              << 1.0 << "x | "
              << naive_checksum << '\n';

    std::cout << "Tiled 16x16    | "
              << tile16_ms << " | "
              << naive_ms / tile16_ms << "x | "
              << tile16_checksum << '\n';

    std::cout << "Tiled 64x64    | "
              << tile64_ms << " | "
              << naive_ms / tile64_ms << "x | "
              << tile64_checksum << '\n';

    std::cout << "\nCorrectness verification\n";
    std::cout << "Max difference (naive vs 16x16): "
              << max_difference(C_naive, C16) << '\n';
    std::cout << "Max difference (naive vs 64x64): "
              << max_difference(C_naive, C64) << '\n';

    std::cout << "\nInterpretation\n";
    std::cout << "- Tiling improves data reuse by processing matrix "
                 "sub-blocks.\n";
    std::cout << "- The best tile size depends on CPU cache, compiler, "
                 "and hardware.\n";
    std::cout << "- Compare measured runtimes; do not assume a larger "
                 "tile is always faster.\n";

    return 0;
}
