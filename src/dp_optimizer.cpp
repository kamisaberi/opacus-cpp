#include "opacus/dp_optimizer.h"
#include "opacus/kernels/dp_kernels.cuh"
#include <c10/cuda/CUDAStream.h>

namespace opacus {

DPOptimizer::DPOptimizer(
    std::shared_ptr<torch::optim::Optimizer> base_optimizer,
    float max_grad_norm,
    float noise_multiplier,
    int batch_size)
    : base_optimizer_(std::move(base_optimizer)),
      max_grad_norm_(max_grad_norm),
      noise_multiplier_(noise_multiplier),
      batch_size_(batch_size),
      step_count_(0) {}

void DPOptimizer::zero_grad() {
    base_optimizer_->zero_grad();
    per_sample_grad_map_.clear();
}

void DPOptimizer::set_per_sample_grads(const torch::Tensor& param, const torch::Tensor& per_sample_grad) {
    per_sample_grad_map_[param.unsafeGetTensorImpl()] = per_sample_grad;
}

void DPOptimizer::step() {
    step_count_++;

    // Process parameters in optimizer
    for (auto& group : base_optimizer_->param_groups()) {
        for (auto& param : group.params()) {
            if (!param.grad().defined()) continue;

            auto it = per_sample_grad_map_.find(param.unsafeGetTensorImpl());
            if (it != per_sample_grad_map_.end()) {
                torch::Tensor per_sample_grad = it->second;

                if (param.is_cuda()) {
                    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
                    
                    kernels::launch_fused_dp_step(
                        param.grad().data_ptr<float>(),
                        per_sample_grad.data_ptr<float>(),
                        batch_size_,
                        param.numel(),
                        max_grad_norm_,
                        noise_multiplier_,
                        /*seed=*/1337ULL,
                        /*sequence=*/step_count_,
                        stream
                    );
                } else {
                    // CPU execution path
                    torch::Tensor norms = torch::norm(per_sample_grad.reshape({batch_size_, -1}), 2, 1);
                    torch::Tensor clip_factors = torch::clamp_max(max_grad_norm_ / (norms + 1e-6f), 1.0f);
                    torch::Tensor clipped_grads = per_sample_grad * clip_factors.reshape({batch_size_, 1, 1});

                    torch::Tensor noise = torch::randn_like(param.grad()) * noise_multiplier_ * max_grad_norm_;
                    param.grad().copy_((clipped_grads.sum(0) + noise) / batch_size_);
                }
            }
        }
    }

    // Call underlying base optimizer step (e.g. SGD/Adam update)
    base_optimizer_->step();
}

} // namespace opacus