#include "nn.h"
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

static MatrixXd relu(const MatrixXd &Z)
{
    return Z.array().max(0.0);
}

static MatrixXd softmax(const MatrixXd &Z)
{
    // subtract column max before exp to avoid overflow
    MatrixXd shifted = Z.rowwise() - Z.colwise().maxCoeff();
    MatrixXd exps = shifted.array().exp();
    return exps.array().rowwise() / exps.colwise().sum().array();
}

NeuralNet::NeuralNet(const std::vector<int> &sizes) : sizes(sizes)
{
    std::mt19937_64 rng(42);

    for (int l = 0; l < static_cast<int>(sizes.size()) - 1; l++)
    {
        int in = sizes[l];
        int out = sizes[l + 1];
        Layer layer;

        // He init: N(0, sqrt(2/fan_in))
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / in));
        layer.W = MatrixXd(out, in).unaryExpr([&](double)
                                              { return dist(rng); });
        layer.b = VectorXd::Zero(out);
        layer.dW = MatrixXd::Zero(out, in);
        layer.db = VectorXd::Zero(out);
        layer.Z = MatrixXd::Zero(out, 1);
        layer.A = MatrixXd::Zero(out, 1);

        layers.push_back(std::move(layer));
    }
}

MatrixXd NeuralNet::forward(const MatrixXd &X)
{
    const MatrixXd *input = &X;
    for (int l = 0; l < static_cast<int>(layers.size()); l++)
    {
        layers[l].Z = (layers[l].W * (*input)).colwise() + layers[l].b;
        bool is_output = (l == static_cast<int>(layers.size()) - 1);
        layers[l].A = is_output ? softmax(layers[l].Z) : relu(layers[l].Z);
        input = &layers[l].A;
    }
    return layers.back().A;
}

void NeuralNet::backward(const MatrixXd &X, const MatrixXd &Y_onehot)
{
    int L = layers.size();
    int batch_size = X.cols();

    std::vector<MatrixXd> dZ(L);

    // combined softmax + cross-entropy: dZ = (probs - one_hot) / N
    dZ[L - 1] = (layers[L - 1].A - Y_onehot) / batch_size;

    for (int l = L - 2; l >= 0; l--)
    {
        MatrixXd dA = layers[l + 1].W.transpose() * dZ[l + 1];
        dZ[l] = dA.array() * (layers[l].Z.array() > 0.0).cast<double>();
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(L)
#endif
    for (int l = 0; l < L; l++)
    {
        const MatrixXd &A_prev = (l == 0) ? X : layers[l - 1].A;
        layers[l].dW = dZ[l] * A_prev.transpose();
        layers[l].db = dZ[l].rowwise().sum();
    }
}

VectorXi NeuralNet::predict(const MatrixXd &X)
{
    MatrixXd probs = forward(X);
    VectorXi pred(X.cols());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
    for (int i = 0; i < X.cols(); i++)
    {
        Eigen::Index idx;
        probs.col(i).maxCoeff(&idx);
        pred(i) = static_cast<int>(idx);
    }
    return pred;
}

int NeuralNet::total_params() const
{
    int total = 0;
    for (const auto &layer : layers)
        total += static_cast<int>(layer.W.size()) + static_cast<int>(layer.b.size());
    return total;
}

VectorXd NeuralNet::pack_gradients() const
{
    VectorXd flat(total_params());
    int offset = 0;
    for (const auto &layer : layers)
    {
        int nW = static_cast<int>(layer.dW.size());
        int nb = static_cast<int>(layer.db.size());
        flat.segment(offset, nW) = Eigen::Map<const VectorXd>(layer.dW.data(), nW);
        offset += nW;
        flat.segment(offset, nb) = layer.db;
        offset += nb;
    }
    return flat;
}

void NeuralNet::unpack_gradients(const VectorXd &flat)
{
    int offset = 0;
    for (auto &layer : layers)
    {
        int nW = static_cast<int>(layer.dW.size());
        int nb = static_cast<int>(layer.db.size());
        Eigen::Map<VectorXd>(layer.dW.data(), nW) = flat.segment(offset, nW);
        offset += nW;
        layer.db = flat.segment(offset, nb);
        offset += nb;
    }
}

VectorXd NeuralNet::pack_params() const
{
    VectorXd flat(total_params());
    int offset = 0;
    for (const auto &layer : layers)
    {
        int nW = static_cast<int>(layer.W.size());
        int nb = static_cast<int>(layer.b.size());
        flat.segment(offset, nW) = Eigen::Map<const VectorXd>(layer.W.data(), nW);
        offset += nW;
        flat.segment(offset, nb) = layer.b;
        offset += nb;
    }
    return flat;
}

void NeuralNet::unpack_params(const VectorXd &flat)
{
    int offset = 0;
    for (auto &layer : layers)
    {
        int nW = static_cast<int>(layer.W.size());
        int nb = static_cast<int>(layer.b.size());
        Eigen::Map<VectorXd>(layer.W.data(), nW) = flat.segment(offset, nW);
        offset += nW;
        layer.b = flat.segment(offset, nb);
        offset += nb;
    }
}

SGD::SGD(const NeuralNet &net, double lr, double momentum, double lr_decay)
    : lr(lr), momentum(momentum), lr_decay(lr_decay)
{
    for (const auto &layer : net.layers)
    {
        vW.push_back(MatrixXd::Zero(layer.W.rows(), layer.W.cols()));
        vb.push_back(VectorXd::Zero(layer.b.size()));
    }
}

void SGD::update(NeuralNet &net, int epoch)
{
    double current_lr = lr * std::pow(lr_decay, epoch);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(net.layers.size()))
#endif
    for (int l = 0; l < static_cast<int>(net.layers.size()); l++)
    {
        vW[l] = momentum * vW[l] - current_lr * net.layers[l].dW;
        vb[l] = momentum * vb[l] - current_lr * net.layers[l].db;
        net.layers[l].W += vW[l];
        net.layers[l].b += vb[l];
    }
}

std::pair<MatrixXd, VectorXi> load_csv(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::istringstream ss(line);
        std::vector<double> row;
        std::string token;
        while (std::getline(ss, token, ','))
            row.push_back(std::stod(token));
        rows.push_back(std::move(row));
    }

    if (rows.empty())
        throw std::runtime_error("Empty file: " + path);

    int N = static_cast<int>(rows.size());
    int D = static_cast<int>(rows[0].size()) - 1; // col 0 is label

    MatrixXd X(D, N);
    VectorXi y(N);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
#endif
    for (int i = 0; i < N; i++)
    {
        y(i) = static_cast<int>(rows[i][0]);
        for (int j = 0; j < D; j++)
            X(j, i) = rows[i][j + 1] / 255.0;
    }

    return {X, y};
}

MatrixXd one_hot(const VectorXi &labels, int num_classes)
{
    int N = labels.size();
    MatrixXd Y = MatrixXd::Zero(num_classes, N);
    for (int i = 0; i < N; i++)
        Y(labels(i), i) = 1.0;
    return Y;
}

double accuracy(const VectorXi &pred, const VectorXi &labels)
{
    return (pred.array() == labels.array()).cast<double>().mean();
}
