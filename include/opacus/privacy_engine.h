#ifndef OPACUS_PRIVACY_ENGINE_H
#define OPACUS_PRIVACY_ENGINE_H

#include "opacus/dp_optimizer.h"
#include "opacus/accountants/rdp_accountant.h"
#include <memory>
#include <tuple>

namespace opacus {

class PrivacyEngine {
public:
    PrivacyEngine();

    std::tuple<std::shared_ptr<torch::nn::Module>, std::shared_ptr<DPOptimizer>> make_private(
        std::shared_ptr<torch::nn::Module> model,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        float target_delta,
        float max_grad_norm,
        float noise_multiplier,
        int batch_size,
        double sample_rate
    );

    std::pair<double, double> get_privacy_spent() const;
    void step_accountant();

private:
    accountants::RDPAccountant accountant_;
    float target_delta_;
    float noise_multiplier_;
    double sample_rate_;
};

} // namespace opacus

#endif // OPACUS_PRIVACY_ENGINE_H