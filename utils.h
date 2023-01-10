#pragma once

#include <tiny_dnn/tiny_dnn.h>

tiny_dnn::core::backend_t parse_backend_name(const std::string& name);

void usage(const char* argv0);
