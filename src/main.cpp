#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include "nn.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main()
{
#ifdef _OPENMP
    std::cout << "OpenMP enabled — threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP disabled — single-threaded build\n";
#endif

    std::cout << "Loading data...\n";
    auto [X_train, y_train] = load_csv("data/mnist_train.csv");
    auto [X_test, y_test] = load_csv("data/mnist_test.csv");

    int N_train = static_cast<int>(X_train.cols());
    int N_test = static_cast<int>(X_test.cols());
    std::cout << "train: " << N_train << "  test: " << N_test << "\n\n";

    NeuralNet net({784, 256, 128, 10});
    SGD optimizer(net, /*lr=*/0.01, /*momentum=*/0.9, /*lr_decay=*/1.0);

    std::cout << "total params: " << net.total_params() << "\n\n";

    const int epochs = 10;
    const int batch_size = 64;
    const int num_batches = N_train / batch_size;

    std::mt19937 rng(0);
    std::vector<int> idx(N_train);
    std::iota(idx.begin(), idx.end(), 0);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::shuffle(idx.begin(), idx.end(), rng);

        double epoch_loss = 0.0;

        for (int b = 0; b < num_batches; b++)
        {
            int start = b * batch_size;

            Eigen::MatrixXd X_batch(784, batch_size);
            Eigen::VectorXi y_batch(batch_size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
            for (int i = 0; i < batch_size; i++)
            {
                X_batch.col(i) = X_train.col(idx[start + i]);
                y_batch(i) = y_train(idx[start + i]);
            }
            Eigen::MatrixXd Y_batch = one_hot(y_batch, 10);

            Eigen::MatrixXd probs = net.forward(X_batch);

            double loss = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : loss) schedule(static) num_threads(omp_get_max_threads())
#endif
            for (int i = 0; i < batch_size; i++)
                loss -= std::log(probs(y_batch(i), i) + 1e-9);
            epoch_loss += loss / batch_size;

            net.backward(X_batch, Y_batch);
            optimizer.update(net, epoch);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        Eigen::VectorXi test_pred = net.predict(X_test);
        double test_acc = accuracy(test_pred, y_test);

        std::printf("Epoch %2d/%d | Loss: %.4f | Test Acc: %5.2f%% | Time: %.2fs\n",
                    epoch + 1, epochs,
                    epoch_loss / num_batches,
                    test_acc * 100.0,
                    elapsed);
    }

    return 0;
}
