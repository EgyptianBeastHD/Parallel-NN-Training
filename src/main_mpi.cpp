// main_mpi.cpp -- data-parallel SGD with selectable all-reduce algorithms.
//
// Each rank holds a full copy of the model and processes a shard of every
// global batch. After backward, gradients are summed across ranks (with our
// hand-rolled tree/ring/HD all-reduces or MPI_Allreduce as a baseline), then
// divided by world_size, then SGD steps.
//
// Run:
//     mpirun -np 4 ./build/train_mpi --algo ring --epochs 10 --global-batch 64
//
// All flags are listed in `print_usage` below.

#include <mpi.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn.h"
#include "allreduce.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Options {
    std::string algo          = "ring";   // tree | ring | hd | mpi
    int         epochs        = 10;
    int         global_batch  = 64;       // total batch across all ranks
    double      lr            = 0.01;
    double      momentum      = 0.9;
    double      lr_decay      = 1.0;
    bool        nonblocking   = false;    // pipeline per-layer iallreduce w/ backward
    bool        verify        = false;    // double-run all-reduce vs MPI_Allreduce per iter
    std::string train_csv     = "data/mnist_train.csv";
    std::string test_csv      = "data/mnist_test.csv";
    std::vector<int> sizes    = {784, 256, 128, 10};
    int         warmup_iters  = 0;        // skip first N iters in timing
    bool        verbose       = false;    // print extra per-iter info on rank 0
};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [options]\n"
        "  --algo <tree|ring|hd|mpi>      all-reduce algorithm (default: ring)\n"
        "  --epochs <int>                 number of epochs (default: 10)\n"
        "  --global-batch <int>           total batch size across ranks (default: 64)\n"
        "  --lr <float>                   learning rate (default: 0.01)\n"
        "  --momentum <float>             SGD momentum (default: 0.9)\n"
        "  --nonblocking                  pipeline per-layer iallreduce with backward\n"
        "  --verify                       per-iter verify hand-rolled all-reduce vs MPI_Allreduce\n"
        "  --warmup <int>                 skip first N iterations in timing (default: 0)\n"
        "  --train <path>                 train CSV (default: data/mnist_train.csv)\n"
        "  --test  <path>                 test  CSV (default: data/mnist_test.csv)\n"
        "  --verbose                      extra logging on rank 0\n"
        "  -h, --help                     show this message\n",
        prog);
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if      (a == "--algo")         o.algo         = need("--algo");
        else if (a == "--epochs")       o.epochs       = std::stoi(need("--epochs"));
        else if (a == "--global-batch") o.global_batch = std::stoi(need("--global-batch"));
        else if (a == "--lr")           o.lr           = std::stod(need("--lr"));
        else if (a == "--momentum")     o.momentum     = std::stod(need("--momentum"));
        else if (a == "--nonblocking")  o.nonblocking  = true;
        else if (a == "--verify")       o.verify       = true;
        else if (a == "--warmup")       o.warmup_iters = std::stoi(need("--warmup"));
        else if (a == "--train")        o.train_csv    = need("--train");
        else if (a == "--test")         o.test_csv     = need("--test");
        else if (a == "--verbose")      o.verbose      = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); std::exit(0); }
        else throw std::invalid_argument("unknown arg: " + a);
    }
    return o;
}

// ---------------------------------------------------------------------------
// Pipelined backward: interleaves per-layer MPI_Iallreduce with the backward
// pass. Equivalent to net.backward(X, Y) followed by per-layer all-reduce,
// but lets the comm of layer L run concurrently with the compute of L-1.
//
// Walks layers from L-1 down to 0. For each layer we compute dZ, then dW
// and db, then immediately fire off MPI_Iallreduce on dW and db. We never
// modify W during this loop, so dA = W_{l+1}^T * dZ_{l+1} is unaffected by
// the in-flight all-reduces (those touch dW / db, not W).
//
// All requests are returned via `handles`; the caller waits on them before
// dividing by P and calling the optimizer.
// ---------------------------------------------------------------------------
static void backward_pipelined(NeuralNet& net,
                               const MatrixXd& X,
                               const MatrixXd& Y_onehot,
                               MPI_Comm comm,
                               std::vector<allreduce::NbHandle>& handles)
{
    const int L          = (int)net.layers.size();
    const int batch_size = (int)X.cols();

    handles.clear();
    handles.resize(2 * L);  // one for dW, one for db, per layer

    MatrixXd dZ_next;  // dZ of layer l+1 we just computed, used to derive dZ_l

    for (int l = L - 1; l >= 0; --l) {
        Layer& Ll = net.layers[l];

        MatrixXd dZ;
        if (l == L - 1) {
            // softmax + cross-entropy fused gradient
            dZ = (Ll.A - Y_onehot) / batch_size;
        } else {
            // dA propagated through the next layer's weights, then ReLU mask
            MatrixXd dA = net.layers[l + 1].W.transpose() * dZ_next;
            dZ          = dA.array() * (Ll.Z.array() > 0.0).cast<double>();
        }

        const MatrixXd& A_prev = (l == 0) ? X : net.layers[l - 1].A;
        Ll.dW = dZ * A_prev.transpose();
        Ll.db = dZ.rowwise().sum();

        // Layer l's gradients are ready -- start their all-reduce now.
        allreduce::istart(Ll.dW.data(), (std::size_t)Ll.dW.size(), comm, handles[2 * l + 0]);
        allreduce::istart(Ll.db.data(), (std::size_t)Ll.db.size(), comm, handles[2 * l + 1]);

        dZ_next = std::move(dZ);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Options opts;
    try {
        opts = parse_args(argc, argv);
    } catch (const std::exception& e) {
        if (world_rank == 0) {
            std::fprintf(stderr, "arg error: %s\n", e.what());
            print_usage(argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    allreduce::Algorithm alg = allreduce::Algorithm::RING;
    try {
        alg = allreduce::parse_algorithm(opts.algo);
    } catch (const std::exception& e) {
        if (world_rank == 0) std::fprintf(stderr, "%s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (opts.global_batch % world_size != 0) {
        if (world_rank == 0) {
            std::fprintf(stderr,
                "ERROR: --global-batch (%d) must be divisible by world_size (%d)\n",
                opts.global_batch, world_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int local_batch = opts.global_batch / world_size;

    // ----- load data (every rank reads its own copy; MNIST CSV is small) -----
    if (world_rank == 0) std::printf("Loading data...\n");
    auto [X_train, y_train] = load_csv(opts.train_csv);
    auto [X_test,  y_test]  = load_csv(opts.test_csv);

    const int N_train     = (int)X_train.cols();
    const int N_test      = (int)X_test.cols();
    const int num_batches = N_train / opts.global_batch;

    // ----- build network and broadcast initial weights from rank 0 -----------
    NeuralNet net(opts.sizes);
    SGD       optimizer(net, opts.lr, opts.momentum, opts.lr_decay);

    {
        VectorXd params = net.pack_params();
        MPI_Bcast(params.data(), (int)params.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        net.unpack_params(params);
    }

    // Private communicator for the all-reduce calls. Isolates our point-to-point
    // traffic from the loss MPI_Allreduce, MPI_Bcast of test accuracy, etc.
    MPI_Comm reduce_comm = allreduce::dup_comm(MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::printf("MPI ranks:        %d\n", world_size);
        std::printf("All-reduce algo:  %s%s\n",
                    allreduce::algorithm_name(alg),
                    opts.nonblocking ? " (pipelined non-blocking)" : "");
        std::printf("Architecture:    ");
        for (size_t i = 0; i < opts.sizes.size(); ++i)
            std::printf(" %d%s", opts.sizes[i], (i + 1 == opts.sizes.size() ? "\n" : " ->"));
        std::printf("Total params:     %d\n", net.total_params());
        std::printf("Global batch:     %d  (local batch: %d)\n",
                    opts.global_batch, local_batch);
        std::printf("train: %d  test: %d  batches/epoch: %d\n\n",
                    N_train, N_test, num_batches);
    }

    // ----- training loop ------------------------------------------------------
    // All ranks shuffle with the same seed so they agree on which samples form
    // each global batch. Rank r takes samples [r*local_batch, (r+1)*local_batch)
    // within each global batch.
    std::mt19937 rng(0);
    std::vector<int> idx(N_train);
    std::iota(idx.begin(), idx.end(), 0);

    // Buffers used inside the iteration loop. Allocated once.
    MatrixXd X_local(opts.sizes.front(), local_batch);
    VectorXi y_local(local_batch);

    int total_iters = 0;

    // Per-iteration timers (rank-0 view; every rank participates in MPI_Barrier).
    double t_compute_total = 0.0;
    double t_comm_total    = 0.0;
    double t_iter_total    = 0.0;

    for (int epoch = 0; epoch < opts.epochs; ++epoch) {
        std::shuffle(idx.begin(), idx.end(), rng);

        double epoch_loss = 0.0;
        auto   t_epoch0   = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < num_batches; ++b) {
            const int global_start = b * opts.global_batch;
            const int my_start     = global_start + world_rank * local_batch;

            // Build this rank's slice of the global batch.
            for (int i = 0; i < local_batch; ++i) {
                X_local.col(i) = X_train.col(idx[my_start + i]);
                y_local(i)     = y_train(idx[my_start + i]);
            }
            MatrixXd Y_local = one_hot(y_local, opts.sizes.back());

            const bool time_this = (total_iters >= opts.warmup_iters);

            MPI_Barrier(MPI_COMM_WORLD);
            auto t_iter0 = std::chrono::high_resolution_clock::now();

            // ---- forward ----
            MatrixXd probs = net.forward(X_local);

            double local_loss = 0.0;
            for (int i = 0; i < local_batch; ++i)
                local_loss -= std::log(probs(y_local(i), i) + 1e-9);

            // ---- backward + all-reduce ----
            std::chrono::high_resolution_clock::time_point t_compute_done;
            std::chrono::high_resolution_clock::time_point t_comm_done;

            if (opts.nonblocking) {
                // Pipelined: backward fires off iallreduce per layer as it goes.
                std::vector<allreduce::NbHandle> handles;
                backward_pipelined(net, X_local, Y_local, reduce_comm, handles);
                t_compute_done = std::chrono::high_resolution_clock::now();
                allreduce::iwait_all(handles);
                t_comm_done    = std::chrono::high_resolution_clock::now();

                // Gradients are now SUMs; convert to averages.
                for (auto& Ll : net.layers) {
                    Ll.dW.array() /= (double)world_size;
                    Ll.db.array() /= (double)world_size;
                }
            } else {
                // Blocking: standard backward, then a single hand-rolled all-reduce
                // on the flattened gradient buffer.
                net.backward(X_local, Y_local);
                t_compute_done = std::chrono::high_resolution_clock::now();

                VectorXd flat = net.pack_gradients();

                // Optional: verify our hand-rolled all-reduce against MPI_Allreduce
                // on this exact buffer. Aborts if they differ. Slow, only for debug.
                VectorXd ref_copy;
                if (opts.verify) {
                    ref_copy = flat;
                    MPI_Allreduce(MPI_IN_PLACE, ref_copy.data(),
                                  (int)ref_copy.size(), MPI_DOUBLE, MPI_SUM,
                                  MPI_COMM_WORLD);
                }

                allreduce::run(alg, flat.data(), (std::size_t)flat.size(), reduce_comm);

                if (opts.verify) {
                    double local_max = (flat - ref_copy).cwiseAbs().maxCoeff();
                    double max_err   = 0.0;
                    MPI_Allreduce(&local_max, &max_err, 1, MPI_DOUBLE, MPI_MAX,
                                  MPI_COMM_WORLD);
                    if (max_err > 1e-9 * world_size) {
                        if (world_rank == 0) {
                            std::fprintf(stderr,
                                "VERIFY FAIL: algo=%s iter=%d max_err=%.3e\n",
                                allreduce::algorithm_name(alg), total_iters, max_err);
                        }
                        MPI_Abort(MPI_COMM_WORLD, 3);
                    }
                }

                flat.array() /= (double)world_size;
                net.unpack_gradients(flat);

                t_comm_done = std::chrono::high_resolution_clock::now();
            }

            // ---- optimizer step ----
            optimizer.update(net, epoch);

            auto t_iter1 = std::chrono::high_resolution_clock::now();

            if (time_this) {
                t_compute_total += std::chrono::duration<double>(t_compute_done - t_iter0).count();
                t_comm_total    += std::chrono::duration<double>(t_comm_done    - t_compute_done).count();
                t_iter_total    += std::chrono::duration<double>(t_iter1        - t_iter0).count();
            }

            // Average the per-iteration loss across ranks for reporting.
            double global_loss = 0.0;
            MPI_Allreduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            epoch_loss += global_loss / (double)opts.global_batch;

            ++total_iters;
        }

        auto   t_epoch1 = std::chrono::high_resolution_clock::now();
        double elapsed  = std::chrono::duration<double>(t_epoch1 - t_epoch0).count();

        // Test accuracy on rank 0 only.
        double test_acc = 0.0;
        if (world_rank == 0) {
            VectorXi pred = net.predict(X_test);
            test_acc      = accuracy(pred, y_test);
        }
        MPI_Bcast(&test_acc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            double samples_per_sec = (double)(num_batches * opts.global_batch) / elapsed;
            std::printf(
                "Epoch %2d/%d | Loss: %.4f | Test Acc: %5.2f%% | Time: %.2fs | %.0f samples/s\n",
                epoch + 1, opts.epochs,
                epoch_loss / num_batches,
                test_acc * 100.0,
                elapsed,
                samples_per_sec);
        }
    }

    // ----- summary ------------------------------------------------------------
    int   timed_iters = std::max(0, total_iters - opts.warmup_iters);
    if (world_rank == 0 && timed_iters > 0) {
        double comp_ms = 1000.0 * t_compute_total / timed_iters;
        double comm_ms = 1000.0 * t_comm_total    / timed_iters;
        double iter_ms = 1000.0 * t_iter_total    / timed_iters;
        std::printf("\n--- timing summary (%d iters, warmup %d skipped) ---\n",
                    timed_iters, opts.warmup_iters);
        std::printf("avg compute / iter:   %7.3f ms\n", comp_ms);
        std::printf("avg all-reduce/iter:  %7.3f ms  (%.1f%% of iter)\n",
                    comm_ms, 100.0 * comm_ms / iter_ms);
        std::printf("avg total / iter:     %7.3f ms\n", iter_ms);
        std::printf("gradient buffer:      %d doubles (%.2f MB)\n",
                    net.total_params(),
                    net.total_params() * sizeof(double) / 1.0e6);
    }

    MPI_Comm_free(&reduce_comm);
    MPI_Finalize();
    return 0;
}
