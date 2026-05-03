#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <utility>

struct Layer {
    Eigen::MatrixXd W, dW;
    Eigen::VectorXd b, db;
    Eigen::MatrixXd Z, A;  // Z = pre-activation, A = post-activation
};

class NeuralNet {
public:
    std::vector<Layer> layers;
    std::vector<int> sizes;

    explicit NeuralNet(const std::vector<int>& sizes);  // e.g. {784, 256, 128, 10}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    void backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y_onehot);
    Eigen::VectorXi predict(const Eigen::MatrixXd& X);

    // flat layout per layer: W (col-major) then b, repeated — pass directly to MPI_Allreduce
    Eigen::VectorXd pack_gradients() const;
    void unpack_gradients(const Eigen::VectorXd& flat);

    Eigen::VectorXd pack_params() const;
    void unpack_params(const Eigen::VectorXd& flat);

    int total_params() const;
};

class SGD {
public:
    double lr, momentum, lr_decay;
    std::vector<Eigen::MatrixXd> vW;
    std::vector<Eigen::VectorXd> vb;

    SGD(const NeuralNet& net, double lr, double momentum, double lr_decay = 1.0);
    void update(NeuralNet& net, int epoch);
};

std::pair<Eigen::MatrixXd, Eigen::VectorXi> load_csv(const std::string& path);
Eigen::MatrixXd one_hot(const Eigen::VectorXi& labels, int num_classes);
double accuracy(const Eigen::VectorXi& pred, const Eigen::VectorXi& labels);
