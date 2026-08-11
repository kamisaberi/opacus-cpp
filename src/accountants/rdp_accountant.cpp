#include "opacus/accountants/rdp_accountant.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace opacus {
namespace accountants {

RDPAccountant::RDPAccountant() {
    // Standard list of RDP orders alpha
    for (double alpha = 1.5; alpha <= 64.0; alpha += 0.5) {
        orders_.push_back(alpha);
        rdp_values_.push_back(0.0);
    }
}

double RDPAccountant::compute_rdp_step(double alpha, double sigma, double q) const {
    if (q == 0.0) return 0.0;
    if (q == 1.0) {
        return alpha / (2.0 * sigma * sigma);
    }
    // RDP upper bound approximation for subsampled Gaussian mechanism
    return (alpha * q * q) / (sigma * sigma);
}

void RDPAccountant::step(double noise_multiplier, double sample_rate) {
    for (size_t i = 0; i < orders_.size(); ++i) {
        rdp_values_[i] += compute_rdp_step(orders_[i], noise_multiplier, sample_rate);
    }
}

std::pair<double, double> RDPAccountant::get_privacy_spent(double target_delta) const {
    double min_eps = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < orders_.size(); ++i) {
        double alpha = orders_[i];
        double rdp = rdp_values_[i];

        // Optimal epsilon derivation from RDP order alpha: eps = rdp + log(1/delta) / (alpha - 1)
        double eps = rdp + (std::log(1.0 / target_delta)) / (alpha - 1.0);
        if (eps < min_eps) {
            min_eps = eps;
        }
    }

    return {min_eps, target_delta};
}

void RDPAccountant::reset() {
    std::fill(rdp_values_.begin(), rdp_values_.end(), 0.0);
}

} // namespace accountants
} // namespace opacus