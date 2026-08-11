#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "opacus/privacy_engine.h"
#include "opacus/per_sample_grad.h"

int main() {
    std::cout << "=== opacus-cpp High-Performance DP-SGD Benchmark ===" << std::endl;

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "[+] CUDA acceleration detected!" << std::endl;
        device = torch::Device(torch::kCUDA, 0);
    } else {
        std::cout << "[-] CUDA unavailable. Using CPU." << std::endl;
    }

    // 1. Build Model
    auto linear_layer = std::make_shared<torch::nn::Linear>(1024, 512);
    auto model = std::make_shared<torch::nn::Sequential>(
        linear_layer,
        torch::nn::ReLU(),
        torch::nn::Linear(512, 10)
    );
    model->to(device);

    // 2. Optimizer
    auto base_optimizer = std::make_shared<torch::optim::SGD>(
        model->parameters(), torch::optim::SGDOptions(0.01)
    );

    // 3. Privacy Engine & DP-Optimizer
    const int batch_size = 128;
    opacus::PrivacyEngine privacy_engine;
    auto [dp_model, dp_optimizer] = privacy_engine.make_private(
        model, base_optimizer, 
        /*target_delta=*/1e-5f, 
        /*max_grad_norm=*/1.0f, 
        /*noise_multiplier=*/1.1f, 
        batch_size, 
        /*sample_rate=*/0.01
    );

    // 4. Synthetic Evaluation Batch
    torch::Tensor inputs = torch::rand({batch_size, 1024}, device);
    torch::Tensor targets = torch::randint(0, 10, {batch_size}, torch::TensorOptions().dtype(torch::kLong).device(device));

    // Warmup
    std::cout << "[*] Running DP-SGD Warmup Iterations..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        dp_optimizer->zero_grad();
        torch::Tensor outputs = dp_model->forward({inputs}).toTensor();
        torch::Tensor loss = torch::nn::functional::cross_entropy(outputs, targets);
        loss.backward();
        dp_optimizer->step();
    }
    if (device.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();

    // 5. Benchmark DP-SGD Step Latency
    std::cout << "\n[*] Running 100 DP-SGD Training Steps..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < 100; ++step) {
        dp_optimizer->zero_grad();
        torch::Tensor outputs = dp_model->forward({inputs}).toTensor();
        torch::Tensor loss = torch::nn::functional::cross_entropy(outputs, targets);
        loss.backward();

        // Calculate and set per-sample gradient
        torch::Tensor per_sample_grad = opacus::PerSampleGradEngine::compute_linear_per_sample_grad(inputs, outputs);
        dp_optimizer->set_per_sample_grads(linear_layer->weight, per_sample_grad);

        // Fused DP Clipping & Noise Step
        dp_optimizer->step();
        privacy_engine.step_accountant();
    }

    if (device.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    [+] Total Time (100 DP-SGD Steps): " << total_ms << " ms" << std::endl;
    std::cout << "    [+] Average Latency Per Step: " << total_ms / 100.0 << " ms" << std::endl;
    std::cout << "    [+] Throughput: " << (100.0 * batch_size) / (total_ms / 1000.0) << " samples/sec" << std::endl;

    // Query Spent Privacy
    auto [eps, delta] = privacy_engine.get_privacy_spent();
    std::cout << "\n[+] Privacy Budget Spent: (epsilon = " << eps << ", delta = " << delta << ")" << std::endl;

    return 0;
}