#include <iostream>
#include <random>
#include <vector>
#include <tiny_dnn/tiny_dnn.h>


int main(int argc, char** argv) {
    std::cout << "[DEBUG] start project" << std::endl;

    // ���������p
    std::random_device rd;
    std::mt19937_64 mt(rd());

    // �w�K�p�f�[�^
    std::vector<tiny_dnn::vec_t> input_train;
    std::vector<tiny_dnn::vec_t> teach_train;
    std::vector<tiny_dnn::label_t> label_train;
    std::vector<tiny_dnn::vec_t> input_test;
    std::vector<tiny_dnn::vec_t> teach_test;
    std::vector<tiny_dnn::label_t> label_test;

    // �w�K�p�f�[�^����
    int max_record = 1000;
    for (int i_record = 0; i_record < max_record; ++i_record)
    {
        auto makeData = [&rd](std::vector<tiny_dnn::vec_t>& input, std::vector<tiny_dnn::vec_t>& teach, std::vector<tiny_dnn::label_t>& label)
        {
            auto dist = std::uniform_int_distribution<int>(0, 3);

            input.emplace_back();
            for (int i_param = 0; i_param < 10; ++i_param)
            {
                input.back().emplace_back(dist(rd));
            }
            int sum = 0;
            for (auto itm : input.back()) { sum += (int)itm; }

            teach.emplace_back(31, 0.0f);
            teach.back().at(sum) = 1.0f;

            label.emplace_back(sum);
        };

        if (i_record < (int)((float)max_record * 0.7f))
        {
            makeData(input_train, teach_train, label_train);
        }
        else
        {
            makeData(input_test, teach_test, label_test);
        }
    }

    //�l�b�g���[�N�̍\�z
    tiny_dnn::network<tiny_dnn::sequential> net;
    {
        net << tiny_dnn::fully_connected_layer(10, 50);
        net << tiny_dnn::tanh_layer();

        net << tiny_dnn::fully_connected_layer(50, 31);
        net << tiny_dnn::softmax_layer();
    }

    // �I�v�e�B�}�C�U�̌���
    tiny_dnn::adam optimizer;

    // �w�K���s
    net.train<tiny_dnn::cross_entropy_multiclass>(optimizer, input_train, label_train, 64, 1);

    // �덷�̌v��
    auto loss_train = net.get_loss<tiny_dnn::mse>(input_train, teach_train) / (int)((float)max_record * 0.7f);
    auto loss_test = net.get_loss<tiny_dnn::mse>(input_test, teach_test) / (int)((float)max_record * 0.3f);

    // �o��
    std::cout << "train loss:" << loss_train << "\t test loss:" << loss_test << std::endl;

    // �������~�߂�
    //while (true) { std::this_thread::yield(); }

    std::cout << "[DEBUG] end project" << std::endl;
    return 0;
}