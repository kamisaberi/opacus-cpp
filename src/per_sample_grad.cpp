#include "opacus/per_sample_grad.h"

namespace opacus {

torch::Tensor PerSampleGradEngine::compute_linear_per_sample_grad(
    const torch::Tensor& input, 
    const torch::Tensor& grad_output) 
{
    // Efficient einsum calculation for linear layer per-sample gradients:
    // input: [B, In_Features], grad_output: [B, Out_Features]
    // Result per-sample gradient: [B, Out_Features, In_Features]
    return torch::einsum("bi,bo->boi", {input, grad_output});
}

torch::Tensor PerSampleGradEngine::compute_conv2d_per_sample_grad(
    const torch::Tensor& input, 
    const torch::Tensor& grad_output) 
{
    int batch_size = input.size(0);
    // Expand dimension for batchwise conv per-sample gradient calculation
    auto grad_samples = torch::einsum("bchw,bohw->boc", {input, grad_output});
    return grad_samples;
}

} // namespace opacus