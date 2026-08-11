#ifndef OPACUS_DP_KERNELS_CUH
#define OPACUS_DP_KERNELS_CUH

#include <cuda_runtime.h>

namespace opacus {
namespace kernels {

// Fused CUDA kernel launcher: calculates per-sample norm, clips grads, adds Gaussian noise, and computes batch mean
void launch_fused_dp_step(
    float* __restrict__ param_grads,
    const float* __restrict__ per_sample_grads,
    int batch_size,
    int param_size,
    float max_grad_norm,
    float noise_multiplier,
    unsigned long long seed,
    unsigned long long sequence,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace opacus

#endif // OPACUS_DP_KERNELS_CUH