#pragma once

#include "tiny_dnn/tiny_dnn.h"

void construct_net(tiny_dnn::network<tiny_dnn::sequential>& nn,
    tiny_dnn::core::backend_t backend_type);

template<typename T>
std::vector<T> slice(std::vector<T> const& v, int m, int n);

void train_lenet(const std::string& data_dir_path,
    double learning_rate,
    const int n_train_epochs,
    const int n_minibatch,
    tiny_dnn::core::backend_t backend_type);
