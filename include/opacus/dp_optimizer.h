#ifndef OPACUS_DP_OPTIMIZER_H
#define OPACUS_DP_OPTIMIZER_H

#include <torch/torch.h>
#include <memory>

namespace opacus {

class DPOptimizer {
public:
    DPOptimizer(
        std::shared_ptr<torch::optim::Optimizer> base_optimizer,
        float max_grad_norm,
        float noise_multiplier,
        int batch_size
    );

    void zero_grad();
    void step();

    void set_per_sample_grads(const torch::Tensor& param, const torch::Tensor& per_sample_grad);

private:
    std::shared_ptr<torch::optim::Optimizer> base_optimizer_;
    float max_grad_norm_;
    float noise_multiplier_;
    int batch_size_;
    unsigned long long step_count_;

    std::unordered_map<void*, torch::Tensor> per_sample_grad_map_;
};

} // namespace opacus

#endif // OPACUS_DP_OPTIMIZER_H