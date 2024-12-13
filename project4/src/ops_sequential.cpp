#include "ops.hpp"
#include <algorithm>

const float epsilon = 1e-20;

void gemm(const float* A, const float* B, float* Out, size_t batch, size_t mn, size_t k)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < k; ++i) {
            Out[b * k + i] = 0;
            for (size_t j = 0; j < mn; ++j) {
                Out[b * k + i] += A[b * mn + j] * B[j * k + i];
            }
        }
    }

    // END YOUR CODE HERE <-
}

void add_bias(float* A, float* B, const float* bias, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < out_dim; ++j) {
            B[i * out_dim + j] = A[i * out_dim + j] + bias[j];
        }
    }
    
    // END YOUR CODE HERE <-
}

void Relu(float* A, float* B, size_t size)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < size; ++i) {
        B[i] = std::max(0.0f, A[i]);
    }
    
    // END YOUR CODE HERE <-
}

void Softmax(float* A, float* B, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t b = 0; b < batch; ++b) {
        float max_val = *std::max_element(A + b * out_dim, A + (b + 1) * out_dim);
        float sum_exp = 0;
        for (size_t i = 0; i < out_dim; ++i) {
            B[b * out_dim + i] = exp(A[b * out_dim + i] - max_val);
            sum_exp += B[b * out_dim + i];
        }
        for (size_t i = 0; i < out_dim; ++i) {
            B[b * out_dim + i] /= sum_exp;
        }
    }
    
    // END YOUR CODE HERE <-
}

void vector_to_one_hot_matrix(const unsigned char* A, float* B, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    std::fill(B, B + batch * out_dim, 0);
    for (size_t i = 0; i < batch; ++i) {
        B[i * out_dim + A[i]] = 1.0f;
    }
    
    // END YOUR CODE HERE <-
}

void cross_entropy_loss(const float* A, const float* B, float* Loss, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
        // Optional for your debug
    for (size_t i = 0; i < batch; ++i) {
        Loss[i] = 0;
        for (size_t j = 0; j < out_dim; ++j) {
            size_t idx = i * out_dim + j;
            Loss[i] += -B[idx] * log(A[idx] + epsilon);
        }
    }    
    // END YOUR CODE HERE <-
}

void cross_entropy_loss_grad(const float* A, const float* B, float* Grad, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < batch * out_dim; ++i) {
        Grad[i] = (A[i] - B[i]) / (A[i] * (1 - A[i]) + epsilon);
    }
    
    // END YOUR CODE HERE <-
}

void update_bias(float* Bias, const float* Output_Grad, size_t batch, float lr, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t j = 0; j < out_dim; ++j) {
        float grad_sum = 0;
        for (size_t i = 0; i < batch; ++i) {
            grad_sum += Output_Grad[i * out_dim + j];
        }
        Bias[j] -= lr * grad_sum / batch;
    }
    
    // END YOUR CODE HERE <-
}

void input_grad(const float* Weight, const float* Output_Grad, float* Input, float* Input_Grad, size_t batch, size_t in_dim, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    std::fill(Input_Grad, Input_Grad + batch * in_dim, 0);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < in_dim; ++j) {
            for (size_t k = 0; k < out_dim; ++k) {
                Input_Grad[i * in_dim + j] += Weight[j * out_dim + k] * Output_Grad[i * out_dim + k];
            }
        }
    }
    
    // END YOUR CODE HERE <-
}

void update_weight(float* Weight, const float* Output_Grad, const float* Input, size_t batch, float lr, size_t in_dim, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t j = 0; j < in_dim; ++j) {
        for (size_t k = 0; k < out_dim; ++k) {
            float grad_sum = 0;
            for (size_t i = 0; i < batch; ++i) {
                grad_sum += Input[i * in_dim + j] * Output_Grad[i * out_dim + k];
            }
            Weight[j * out_dim + k] -= lr * grad_sum / batch;
        }
    }
    
    // END YOUR CODE HERE <-
}

void relu_grad(const float* A, float* Grad, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < batch * out_dim; ++i) {
        Grad[i] *= A[i] > 0 ? 1.0 : 0.0;
    }
    
    // END YOUR CODE HERE <-
}

float mean_acc(const unsigned char* result, const unsigned char* labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE HERE ->
    
    float correct = 0.0f;
    for (size_t i = 0; i < images_num; ++i) {
        if (result[i] == labels_array[i]) {
            correct += 1.0f;
        }
    }
    return (correct / images_num) * 100.0f;
    // END YOUR CODE HERE <-
}

void argmax(const float* A, unsigned char* B, size_t num_classes, size_t images_num)
{
    // BEGIN YOUR CODE HERE ->
    for (size_t i = 0; i < images_num; ++i) {
        size_t idx = i * num_classes;
        float max_val = A[idx];
        size_t max_idx = 0;
        for (size_t j = 1; j < num_classes; ++j) {
            if (A[idx + j] > max_val) {
                max_val = A[idx + j];
                max_idx = j;
            }
        }
        B[i] = max_idx;
    }
    
    // END YOUR CODE HERE <-
}
