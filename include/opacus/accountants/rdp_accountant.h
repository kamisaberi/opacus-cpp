#ifndef OPACUS_RDP_ACCOUNTANT_H
#define OPACUS_RDP_ACCOUNTANT_H

#include <vector>
#include <utility>

namespace opacus {
namespace accountants {

class RDPAccountant {
public:
    RDPAccountant();

    // Step accounting: log a training step with noise multiplier sigma and sampling rate q
    void step(double noise_multiplier, double sample_rate);

    // Compute (epsilon, delta) privacy spent
    std::pair<double, double> get_privacy_spent(double target_delta) const;

    void reset();

private:
    std::vector<double> orders_;
    std::vector<double> rdp_values_;

    double compute_rdp_step(double alpha, double sigma, double q) const;
};

} // namespace accountants
} // namespace opacus

#endif // OPACUS_RDP_ACCOUNTANT_H