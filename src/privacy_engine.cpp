#include "opacus/privacy_engine.h"

namespace opacus {

PrivacyEngine::PrivacyEngine() 
    : target_delta_(1e-5f), noise_multiplier_(1.0f), sample_rate_(0.01) {}

std::tuple<std::shared_ptr<torch::nn::Module>, std::shared_ptr<DPOptimizer>> PrivacyEngine::make_private(
    std::shared_ptr<torch::nn::Module> model,
    std::shared_ptr<torch::optim::Optimizer> optimizer,
    float target_delta,
    float max_grad_norm,
    float noise_multiplier,
    int batch_size,
    double sample_rate) 
{
    target_delta_ = target_delta;
    noise_multiplier_ = noise_multiplier;
    sample_rate_ = sample_rate;

    auto dp_optimizer = std::make_shared<DPOptimizer>(
        optimizer, max_grad_norm, noise_multiplier, batch_size
    );

    return {model, dp_optimizer};
}

void PrivacyEngine::step_accountant() {
    accountant_.step(noise_multiplier_, sample_rate_);
}

std::pair<double, double> PrivacyEngine::get_privacy_spent() const {
    return accountant_.get_privacy_spent(target_delta_);
}

} // namespace opacus