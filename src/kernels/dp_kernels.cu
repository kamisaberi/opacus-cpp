#include "opacus/kernels/dp_kernels.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

namespace opacus {
namespace kernels {

__global__ void fused_dp_clip_noise_kernel(
    float* __restrict__ param_grads,
    const float* __restrict__ per_sample_grads,
    int batch_size,
    int param_size,
    float max_grad_norm,
    float noise_multiplier,
    unsigned long long seed,
    unsigned long long sequence) 
{
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (param_idx < param_size) {
        float accumulated_grad = 0.0f;

        // Loop over each sample in the batch
        for (int b = 0; b < batch_size; ++b) {
            int element_idx = b * param_size + param_idx;
            float val = per_sample_grads[element_idx];

            // 1. Calculate sample grad L2 norm approximation/clipping scaling
            // Compute scale = min(1.0, max_grad_norm / norm_sample)
            float norm_sample = fabsf(val); // Elementwise gradient bounding
            float scale = (norm_sample > max_grad_norm) ? (max_grad_norm / norm_sample) : 1.0f;

            accumulated_grad += val * scale;
        }

        // 2. Add Gaussian noise via cuRAND if noise_multiplier > 0
        if (noise_multiplier > 0.0f) {
            curandStateState_t state;
            curand_init(seed, sequence + param_idx, 0, &state);
            float noise = curand_normal(&state) * noise_multiplier * max_grad_norm;
            accumulated_grad += noise;
        }

        // 3. Batch average & store back into parameter gradient
        param_grads[param_idx] = accumulated_grad / static_cast<float>(batch_size);
    }
}

void launch_fused_dp_step(
    float* __restrict__ param_grads,
    const float* __restrict__ per_sample_grads,
    int batch_size,
    int param_size,
    float max_grad_norm,
    float noise_multiplier,
    unsigned long long seed,
    unsigned long long sequence,
    cudaStream_t stream) 
{
    int threads_per_block = 256;
    int blocks_per_grid = (param_size + threads_per_block - 1) / threads_per_block;

    fused_dp_clip_noise_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        param_grads, per_sample_grads, batch_size, param_size, 
        max_grad_norm, noise_multiplier, seed, sequence
    );
}

} // namespace kernels
} // namespace opacus