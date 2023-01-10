#include <tiny_dnn/tiny_dnn.h>

#include "utils.h"

tiny_dnn::core::backend_t parse_backend_name(const std::string& name) {
    const std::array<const std::string, 5> names = { {
      "internal", "nnpack", "libdnn", "avx", "opencl",
    } };
    for (size_t i = 0; i < names.size(); ++i) {
        if (name.compare(names[i]) == 0) {
            return static_cast<tiny_dnn::core::backend_t>(i);
        }
    }
    return tiny_dnn::core::default_engine();
}

void usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
        << " --learning_rate 1"
        << " --epochs 30"
        << " --minibatch_size 16"
        << " --backend_type internal" << std::endl;
}
