// allreduce_bench.cpp -- correctness and microbenchmark for the three
// hand-rolled all-reduce algorithms.
//
// Usage:
//     mpirun -np 8 ./build/allreduce_bench
//     mpirun -np 8 ./build/allreduce_bench --sizes 1024,16384,262144,4194304 --reps 50
//
// For each requested size and each algorithm: verifies the result against
// MPI_Allreduce, then prints the median of `reps` timed runs (with a warm-up).

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "allreduce.h"

static std::vector<std::size_t> parse_sizes(const std::string& s) {
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back((std::size_t)std::stoull(tok));
    }
    return out;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::size_t> sizes = {
        1024, 8192, 65536, 262144, 1048576, 4194304
    };
    int  reps      = 30;
    int  warmup    = 5;
    bool skip_check = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if      (a == "--sizes")    sizes  = parse_sizes(need("--sizes"));
        else if (a == "--reps")     reps   = std::stoi(need("--reps"));
        else if (a == "--warmup")   warmup = std::stoi(need("--warmup"));
        else if (a == "--no-check") skip_check = true;
    }

    if (rank == 0) {
        std::printf("# all-reduce microbench: %d ranks\n", size);
        std::printf("# reps=%d warmup=%d\n", reps, warmup);
        std::printf("# %-10s %-8s %12s %12s %12s\n",
                    "size", "algo", "median_ms", "min_ms", "GB/s_eff");
    }

    const allreduce::Algorithm algos[] = {
        allreduce::Algorithm::TREE,
        allreduce::Algorithm::RING,
        allreduce::Algorithm::HALVING_DOUBLING,
        allreduce::Algorithm::MPI_BUILTIN,
    };

    std::mt19937_64 rng(1234 + rank);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (std::size_t N : sizes) {
        std::vector<double> reference(N);  // initial input (same on every rank? no -- different)
        for (std::size_t i = 0; i < N; ++i) reference[i] = dist(rng);

        // Compute the ground-truth sum once via MPI_Allreduce.
        std::vector<double> truth = reference;
        MPI_Allreduce(MPI_IN_PLACE, truth.data(), (int)N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (allreduce::Algorithm alg : algos) {
            // Correctness check
            if (!skip_check) {
                std::vector<double> buf = reference;
                allreduce::run(alg, buf.data(), N, MPI_COMM_WORLD);

                double local_max_err = 0.0;
                for (std::size_t i = 0; i < N; ++i) {
                    double e = std::abs(buf[i] - truth[i]);
                    if (e > local_max_err) local_max_err = e;
                }
                double max_err = 0.0;
                MPI_Allreduce(&local_max_err, &max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if (max_err > 1e-9 * (double)size) {
                    if (rank == 0) {
                        std::fprintf(stderr,
                            "FAIL: %s at N=%zu max_err=%.3e (tol=%.3e)\n",
                            allreduce::algorithm_name(alg), N, max_err, 1e-9 * (double)size);
                    }
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
            }

            // Warm-up
            for (int w = 0; w < warmup; ++w) {
                std::vector<double> buf = reference;
                allreduce::run(alg, buf.data(), N, MPI_COMM_WORLD);
            }

            // Timed runs
            std::vector<double> times(reps);
            for (int r = 0; r < reps; ++r) {
                std::vector<double> buf = reference;
                MPI_Barrier(MPI_COMM_WORLD);
                auto t0 = std::chrono::high_resolution_clock::now();
                allreduce::run(alg, buf.data(), N, MPI_COMM_WORLD);
                auto t1 = std::chrono::high_resolution_clock::now();
                times[r] = std::chrono::duration<double>(t1 - t0).count();
            }

            // Reduce times: take max across ranks for each rep, then median.
            for (int r = 0; r < reps; ++r) {
                double t = times[r];
                MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                times[r] = t;
            }

            std::sort(times.begin(), times.end());
            double median = times[reps / 2];
            double tmin   = times.front();
            double bytes  = 2.0 * (double)(size - 1) / (double)size * (double)N * sizeof(double);
            double gbs    = bytes / median / 1.0e9;

            if (rank == 0) {
                std::printf("  %-10zu %-8s %12.4f %12.4f %12.3f\n",
                            N, allreduce::algorithm_name(alg),
                            median * 1000.0, tmin * 1000.0, gbs);
            }
        }
        if (rank == 0) std::printf("\n");
    }

    MPI_Finalize();
    return 0;
}
