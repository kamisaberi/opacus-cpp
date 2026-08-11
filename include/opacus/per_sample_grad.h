#ifndef OPACUS_PER_SAMPLE_GRAD_H
#define OPACUS_PER_SAMPLE_GRAD_H

#include <torch/torch.h>

namespace opacus {

class PerSampleGradEngine {
public:
    static torch::Tensor compute_linear_per_sample_grad(
        const torch::Tensor& input, 
        const torch::Tensor& grad_output
    );

    static torch::Tensor compute_conv2d_per_sample_grad(
        const torch::Tensor& input, 
        const torch::Tensor& grad_output
    );
};

} // namespace opacus

#endif // OPACUS_PER_SAMPLE_GRAD_H