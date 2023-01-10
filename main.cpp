#include <iostream>
#include <tiny_dnn/tiny_dnn.h>

#include "utils.h"
#include "network.h"

void validate_args(
    int argc, char** argv,
    double& learning_rate,
    int& epochs,
    std::string& data_path,
    int& minibatch_size,
    tiny_dnn::core::backend_t& backend_type)
{
    for (int count = 1; count + 1 < argc; count += 2) {
        std::string argname(argv[count]);
        if (argname == "--learning_rate") {
            learning_rate = atof(argv[count + 1]);
        }
        else if (argname == "--epochs") {
            epochs = atoi(argv[count + 1]);
        }
        else if (argname == "--minibatch_size") {
            minibatch_size = atoi(argv[count + 1]);
        }
        else if (argname == "--backend_type") {
            backend_type = parse_backend_name(argv[count + 1]);
        }
        else if (argname == "--data_path") {
            data_path = std::string(argv[count + 1]);
        }
        else {
            std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
            usage(argv[0]);
            exit(-1);
        }
    }
    if (data_path == "") {
        std::cerr << "Data path not specified." << std::endl;
        usage(argv[0]);
        exit(-1);
    }
    if (learning_rate <= 0) {
        std::cerr
            << "Invalid learning rate. The learning rate must be greater than 0."
            << std::endl;
        exit(-1);
    }
    if (epochs <= 0) {
        std::cerr << "Invalid number of epochs. The number of epochs must be "
            "greater than 0."
            << std::endl;
        exit(-1);
    }
    if (minibatch_size <= 0 || minibatch_size > 60000) {
        std::cerr
            << "Invalid minibatch size. The minibatch size must be greater than 0"
            " and less than dataset size (60000)."
            << std::endl;
        exit(-1);
    }
}

int main(int argc, char** argv) {
    double learning_rate = 0.1;
    int epochs = 1;
    std::string data_path = "C:\\Users\\tkmpa\\workdir\\Github\\tiny-dnn\\data";  // datapath‚ðŽ©•ª‚ÅŽw’è
    int minibatch_size = 32;
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

    if (argc == 2) {
        std::string argname(argv[1]);
        if (argname == "--help" || argname == "-h") {
            usage(argv[0]);
            return 0;
        }
    }

    validate_args(argc, argv, learning_rate, epochs, data_path, minibatch_size, backend_type);
    
    std::cout << "Running with the following parameters:" << std::endl
        << "Data path: " << data_path << std::endl
        << "Learning rate: " << learning_rate << std::endl
        << "Minibatch size: " << minibatch_size << std::endl
        << "Number of epochs: " << epochs << std::endl
        << "Backend type: " << backend_type << std::endl
        << std::endl;
    try {
        train_lenet(data_path, learning_rate, epochs, minibatch_size, backend_type);
    }
    catch (tiny_dnn::nn_error& err) {
        std::cerr << "Exception: " << err.what() << std::endl;
    }
    return 0;
}