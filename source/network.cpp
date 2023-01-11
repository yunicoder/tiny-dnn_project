#include <tiny_dnn/tiny_dnn.h>

#include "network.h"

void construct_net(tiny_dnn::network<tiny_dnn::sequential>& nn,
    tiny_dnn::core::backend_t backend_type) {
    // connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
  // clang-format off
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
    // clang-format on
#undef O
#undef X

  // construct nets
  //
  // C : convolution
  // S : sub-sampling
  // F : fully connected
  // clang-format off
    using fc = tiny_dnn::layers::fc;
    using conv = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using tanh = tiny_dnn::activation::tanh;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;

    nn << conv(32, 32, 5, 1, 6,   // C1, 1@32x32-in, 6@28x28-out
        padding::valid, true, 1, 1, 1, 1, backend_type)
        << tanh()
        << ave_pool(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
        << tanh()
        << conv(14, 14, 5, 6, 16,   // C3, 6@14x14-in, 16@10x10-out
            connection_table(tbl, 6, 16),
            padding::valid, true, 1, 1, 1, 1, backend_type)
        << tanh()
        << ave_pool(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
        << tanh()
        << conv(5, 5, 5, 16, 120,   // C5, 16@5x5-in, 120@1x1-out
            padding::valid, true, 1, 1, 1, 1, backend_type)
        << tanh()
        << fc(120, 10, true, backend_type)  // F6, 120-in, 10-out
        << tanh();
}


template<typename T>
std::vector<T> slice(std::vector<T> const& v, int m, int n)
{
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

void train_lenet(const std::string& data_dir_path,
    double learning_rate,
    const int n_train_epochs,
    const int n_minibatch,
    tiny_dnn::core::backend_t backend_type) {
    // specify loss-function and learning strategy
    tiny_dnn::network<tiny_dnn::sequential> nn;
    tiny_dnn::adagrad optimizer;

    construct_net(nn, backend_type);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<tiny_dnn::label_t> mnist_all_train_labels, mnist_all_test_labels;
    std::vector<tiny_dnn::vec_t> mnist_all_train_images, mnist_all_test_images;

    tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
        &mnist_all_train_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
        &mnist_all_train_images, -1.0, 1.0, 2, 2);
    tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
        &mnist_all_test_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
        &mnist_all_test_images, -1.0, 1.0, 2, 2);

    std::cout << "all train size:" << mnist_all_train_labels.size() << ", all test size: " << mnist_all_test_labels.size() << std::endl;

    // slice to number that require current training
    int train_num = 1000;  // ¡‰ñŽg—p‚·‚é‰æ‘œ–‡”
    std::vector<tiny_dnn::label_t> train_labels = slice(mnist_all_train_labels, 1, train_num);
    std::vector<tiny_dnn::label_t> test_labels = slice(mnist_all_test_labels, 1, train_num);
    std::vector<tiny_dnn::vec_t> train_images = slice(mnist_all_train_images, 1, train_num);
    std::vector<tiny_dnn::vec_t> test_images = slice(mnist_all_test_images, 1, train_num);
    std::cout << "train:" << train_labels.size() << ", test: " << test_labels.size() << std::endl;


    std::cout << "start training" << std::endl;

    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;

    optimizer.alpha *=
        std::min(tiny_dnn::float_t(4),
            static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

    int epoch = 1;
    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
            << t.elapsed() << "s elapsed." << std::endl;

        // loss‚ÌŒvŽZ
        std::cout << "calculate loss..." << std::endl;
        auto train_loss = nn.get_loss<tiny_dnn::mse>(train_images, train_labels);
        auto test_loss = nn.get_loss<tiny_dnn::mse>(test_images, test_labels);

        // accuracy‚ÌŒvŽZ
        std::cout << "calculate accuracy..." << std::endl;
        tiny_dnn::result train_results = nn.test(train_images, train_labels);
        tiny_dnn::result test_results = nn.test(test_images, test_labels);
        float_t train_accuracy = (float_t)train_results.num_success * 100 / train_results.num_total;
        float_t test_accuracy = (float_t)test_results.num_success * 100 / test_results.num_total;

        std::cout << "train loss: " << train_loss << " test loss: " << test_loss << std::endl;
        std::cout << "train accuracy: " << train_accuracy << "% test accuracy: " << test_accuracy << "%" << std::endl;


        ++epoch;
        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

    // training
    nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
        n_train_epochs, on_enumerate_minibatch,
        on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);
    // save network model & trained weights
    nn.save("models/LeNet-model");
}
