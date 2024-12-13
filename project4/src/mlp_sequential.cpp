#include "mlp_network.hpp"

void nn_epoch_cpp(const float* input_array, const unsigned char* label_array, float* Weight1, float* Weight2, float* bias1, float* bias2, size_t input_num, size_t input_dim, size_t hidden_num, size_t class_num, float lr, size_t batch_num)
{
    float* label_array_batch = new float[batch_num * class_num];
    float* fc_1_mid = new float[batch_num * hidden_num];
    float* fc_1_out = new float[batch_num * hidden_num];
    float* relu_1_out = new float[batch_num * hidden_num];
    float* fc_2_mid = new float[batch_num * class_num];
    float* fc_2_out = new float[batch_num * class_num];
    float* softmax_out = new float[batch_num * class_num];
    float* output_grad = new float[batch_num * class_num];
    float* hidden_grad = new float[batch_num * hidden_num];
    float* loss = new float[batch_num];

    for (size_t offset = 0; offset < input_num; offset += batch_num)
    {
        const float* array_batch = input_array + offset * input_dim;
        const unsigned char* label_batch = label_array + offset;
        size_t m_b = input_num - offset > batch_num ? batch_num : input_num - offset;

        // First FC Layer
        gemm(array_batch, Weight1, fc_1_mid, m_b, input_dim, hidden_num);
        add_bias(fc_1_mid, fc_1_out, bias1, m_b, hidden_num);

        // ReLU Activation
        Relu(fc_1_out, relu_1_out, m_b * hidden_num);

        // Second FC Layer
        gemm(relu_1_out, Weight2, fc_2_mid, m_b, hidden_num, class_num);
        add_bias(fc_2_mid, fc_2_out, bias2, m_b, class_num);

        // Softmax
        Softmax(fc_2_out, softmax_out, m_b, class_num);

        // Loss
        vector_to_one_hot_matrix(label_batch, label_array_batch, m_b, class_num);
        cross_entropy_loss(softmax_out, label_array_batch, loss, m_b, class_num);

        // Backward
        cross_entropy_loss_grad(softmax_out, label_array_batch, output_grad, m_b, class_num);
        update_bias(bias2, output_grad, m_b, lr, class_num);
        update_weight(Weight2, output_grad, relu_1_out, m_b, lr, hidden_num, class_num);
        relu_grad(fc_1_out, hidden_grad, m_b, hidden_num);
        update_bias(bias1, hidden_grad, m_b, lr, hidden_num);
        update_weight(Weight1, hidden_grad, array_batch, m_b, lr, input_dim, hidden_num);
    }

    delete[] label_array_batch;
    delete[] fc_1_mid;
    delete[] fc_1_out;
    delete[] relu_1_out;
    delete[] fc_2_mid;
    delete[] fc_2_out;
    delete[] softmax_out;
    delete[] output_grad;
    delete[] hidden_grad;
    delete[] loss;
}

void train_nn(const DataSet* train_data, const DataSet* test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_weight1 = train_data->input_dim * hidden_dim;
    size_t size_bias1 = hidden_dim;
    size_t size_weight2 = hidden_dim * num_classes;
    size_t size_bias2 = num_classes;
    float* W1 = new float[size_weight1];
    float* b1 = new float[size_bias1];
    float* W2 = new float[size_weight2];
    float* b2 = new float[size_bias2];
    std::mt19937 rng;
    rng.seed(0);
    float stddev = std::sqrt(2.0f / train_data->input_dim);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (size_t i = 0; i < size_weight1; i++) {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_weight2; i++) {
        W2[i] = dist(rng);
    }
    for (size_t i = 0; i < size_bias1; i++) {
        b1[i] = 0;
    }
    for (size_t i = 0; i < size_bias2; i++) {
        b2[i] = 0;
    }

    unsigned char* test_result_class = new unsigned char[test_data->images_num];  // corrected
    float* test_result = new float[test_data->images_num * num_classes];
    float* fc1_temp = new float[test_data->images_num * hidden_dim];  // corrected: declare and allocate memory

    float test_err;  // declaration should be here to be used later

    std::cout << "| Epoch |  Acc Rate  |  Training Time" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        auto time_1 = std::chrono::high_resolution_clock::now();
        nn_epoch_cpp(train_data->images_matrix, train_data->labels_array, W1, W2, b1, b2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, lr, batch);
        auto time_2 = std::chrono::high_resolution_clock::now();

        // Test
        size_t test_input_num = test_data->images_num;
        for (size_t offset = 0; offset < test_input_num; offset += batch) {
            const float* array_batch = test_data->images_matrix + offset * test_data->input_dim;
            size_t m_b = test_input_num - offset > batch ? batch : test_input_num - offset;
            gemm(array_batch, W1, fc1_temp, m_b, test_data->input_dim, hidden_dim);
            add_bias(fc1_temp, fc1_temp, b1, m_b, hidden_dim);
            Relu(fc1_temp, fc1_temp, m_b * hidden_dim);
            gemm(fc1_temp, W2, test_result + offset * num_classes, m_b, hidden_dim, num_classes);
            add_bias(test_result + offset * num_classes, test_result + offset * num_classes, b2, m_b, num_classes);
        }

        argmax(test_result, test_result_class, num_classes, test_data->images_num);

        test_err = mean_acc(test_result_class, test_data->labels_array, test_data->images_num, num_classes);

        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1);
        std::cout << "|  " << std::setw(4) << std::right << epoch+1 << " |   "
                  << std::fixed << std::setprecision(3) << test_err << "%  |   "
                  << elapsed_time.count() << " ms" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Execution Time: " << total_elapsed_time.count() << " milliseconds\n";

    delete[] W1;
    delete[] W2;
    delete[] b1;
    delete[] b2;
    delete[] test_result;
    delete[] test_result_class;
    delete[] fc1_temp;  // ensure to delete this to avoid memory leak
}
