#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// ==========================================================
// Standard normal CDF
// ==========================================================
double normalCDF(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// ==========================================================
// European put: Black-Scholes
// ==========================================================
double europeanPutBlackScholes(double S0, double K, double r, double q, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

    double d2 = d1 - sigma * std::sqrt(T);

    return K * std::exp(-r * T) * normalCDF(-d2) - S0 * std::exp(-q * T) * normalCDF(-d1);
}

// ==========================================================
// American put: CRR binomial tree
// ==========================================================
double americanPutBinomial(double S0, double K, double r, double q, double sigma, double T, int N) {
    if (N <= 0)
        throw std::runtime_error("N must be positive.");

    double dt = T / static_cast<double>(N);

    // CRR up/down factors
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;

    // Risk-neutral probability
    double growth = std::exp((r - q) * dt);

    double p = (growth - d) / (u - d);

    if (p < 0.0 || p > 1.0)
        throw std::runtime_error("Invalid risk-neutral probability.");

    // One-step discount factor
    double discount = std::exp(-r * dt);

    std::vector<double> optionValue(N + 1);

    // Ratio between neighboring stock-price nodes
    double upDownRatio = u / d;

    // ======================================================
    // Step 1: terminal payoff
    // ======================================================

    double S = S0 * std::pow(d, N);

    for (int j = 0; j <= N; ++j) {
        optionValue[j] = std::max(K - S, 0.0);

        S *= upDownRatio;
    }

    // ======================================================
    // Step 2: backward induction
    // ======================================================

    for (int i = N - 1; i >= 0; --i) {
        // Lowest stock-price node at time i
        S = S0 * std::pow(d, i);

        for (int j = 0; j <= i; ++j) {
            double continuation = discount * (p * optionValue[j + 1] + (1.0 - p) * optionValue[j]);

            double exercise = std::max(K - S, 0.0);

            optionValue[j] = std::max(exercise, continuation);

            S *= upDownRatio;
        }
    }

    return optionValue[0];
}

// ==========================================================
// Main numerical experiment
// ==========================================================
int main() {
    // Contract parameters
    double S0 = 50.0;
    double K = 55.0;
    double r = 0.05;
    double q = 0.0;
    double sigma = 0.20;
    double T = 1.0;

    // Different binomial resolutions
    std::vector<int> steps = {100, 500, 1000, 1500, 10000, 15000, 100000};

    // European benchmark
    double europeanPrice = europeanPutBlackScholes(S0, K, r, q, sigma, T);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Parameters\n";
    std::cout << "----------------------------------------\n";
    std::cout << "S0    = " << S0 << '\n';
    std::cout << "K     = " << K << '\n';
    std::cout << "r     = " << r << '\n';
    std::cout << "q     = " << q << '\n';
    std::cout << "sigma = " << sigma << '\n';
    std::cout << "T     = " << T << '\n';

    std::cout << "\nEuropean Put (Black-Scholes) = " << europeanPrice << "\n\n";

    std::cout << std::setw(8) << "N" << std::setw(16) << "American" << std::setw(16) << "Change"
              << std::setw(16) << "EarlyExPrem" << std::setw(16) << "Time(ms)" << '\n';

    std::cout << std::string(72, '-') << '\n';

    double previousPrice = std::numeric_limits<double>::quiet_NaN();

    for (int N : steps) {
        auto start = std::chrono::high_resolution_clock::now();

        double americanPrice = americanPutBinomial(S0, K, r, q, sigma, T, N);

        auto end = std::chrono::high_resolution_clock::now();

        double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();

        double change = 0.0;

        if (!std::isnan(previousPrice)) {
            change = std::abs(americanPrice - previousPrice);
        }

        double earlyExercisePremium = americanPrice - europeanPrice;

        std::cout << std::setw(8) << N << std::setw(16) << americanPrice;

        if (std::isnan(previousPrice))
            std::cout << std::setw(16) << "-";
        else
            std::cout << std::setw(16) << change;

        std::cout << std::setw(16) << earlyExercisePremium << std::setw(16) << milliseconds << '\n';

        previousPrice = americanPrice;
    }

    return 0;
}