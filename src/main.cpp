// Necessary libraries
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numbers>
#include <random>
#include <string>

// // Include header files
// #include "black_scholes.hpp"
// #include "monte_carlo.hpp"

double erfinv(double x) {
    if (x <= -1.0 || x >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    constexpr double pi = std::numbers::pi;
    constexpr double a = 0.147;

    const double sign = (x < 0.0) ? -1.0 : 1.0;
    const double logTerm = std::log(1.0 - x * x);
    const double term = 2.0 / (pi * a) + logTerm / 2.0;

    return sign * std::sqrt(-term + std::sqrt(term * term - logTerm / a));
}

double integral_function(double x) { return std::cos(2.0 * std::numbers::pi * x); }

void monte_carlo_integration(double alpha, double targetError, double lowerBound,
                             double upperBound) {

    // compute the z-score corresponding to alpha level
    const double z = std::sqrt(2.0) * erfinv(1.0 - alpha);

    // avoid stopping too early
    constexpr std::size_t minimumSamples = 1000;

    // safety limit
    constexpr std::size_t maximumSamples = 1000000000;

    // random generator engine
    std::mt19937 generator(42);

    // define the uniform distribution
    std::uniform_real_distribution<double> distribution(lowerBound, upperBound);

    // initialize number of iteration
    std::size_t n = 0;

    // estimation/statistics helper variables
    double sum = 0.0;
    double sumSquared = 0.0;
    double estimate = 0.0;
    double estimatedVariance = 0.0;
    double standardError = 0.0;
    double halfWidth = std::numeric_limits<double>::infinity();

    while ((n < minimumSamples || halfWidth > targetError) && n < maximumSamples) {

        // start generating random variable
        const double x = distribution(generator);

        // Y_i = (b-a)g(X_i) -> so E[Y_i] equals the integral
        const double y = (upperBound - lowerBound) * integral_function(x);

        // increase n
        ++n;

        sum += y;
        sumSquared += y * y;

        if (n >= 2) {
            estimate = sum / static_cast<double>(n);

            estimatedVariance = (sumSquared - static_cast<double>(n) * estimate * estimate) /
                                static_cast<double>(n - 1);

            // Prevent a tiny negative value caused by rounding
            if (estimatedVariance < 0.0) {
                estimatedVariance = 0.0;
            }

            standardError = std::sqrt(estimatedVariance / static_cast<double>(n));

            halfWidth = z * standardError;
        }
    }

    const double ciLower = estimate - halfWidth;
    const double ciUpper = estimate + halfWidth;

    std::cout << "Required number of iterations: " << n << '\n';
    // Alpha: 0.05, 0.10, 0.03
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Alpha:              " << alpha << '\n';

    // Confidence level: 95%, 90%, 97%
    std::cout << std::setprecision(0);
    std::cout << "Confidence level:   " << (1.0 - alpha) * 100.0 << "%\n";
    std::cout << std::setprecision(2);
    std::cout << "Critical value z:   " << z << '\n';

    // Statistical results: retain higher precision
    std::cout << std::setprecision(8);
    std::cout << "Estimation:         " << estimate << '\n';
    std::cout << "Empirical variance: " << estimatedVariance << '\n';
    std::cout << "Standard error:     " << standardError << '\n';
    std::cout << "Confidence interval: [" << ciLower << ", " << ciUpper << "]\n";

    if (n == maximumSamples && halfWidth > targetError) {
        std::cout << "Warning: maximum sample limit reached "
                     "before the target was satisfied.\n";
    }
}

void confidence_interval() {}

int main() {
    // EuropeanOption callOptionBS;
    // std::cout << "Call option price (Black-Scholes - no dividends): " << callOptionBS.Price()
    //           << "\n";

    // size_t number_of_simulations = 10000;

    // EuropeanOptionMonteCarlo callOptionMC;
    // std::cout << "Call option price (Monte Carlo - no dividends): "
    //           << callOptionMC.Price(number_of_simulations);

    const double pi = std::numbers::pi;

    // int tests[] = {10, 100, 1000, 10000, 100000, 1000000};

    // int alpha[] = {0.85, 0.9, 0.95, 0.99};
    // int size = sizeof(alpha) / sizeof(alpha[0]);
    // double targetError = 10e-5;
    monte_carlo_integration(0.05, 0.0001, 0.0, 1.0);

    return 0;
}