#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

struct SimulationResult {
    double price;
    double pathVariance;
    double estimatorVariance;
    double standardError;
    double ciLower;
    double ciUpper;
};

// Standard normal CDF
double normalCDF(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Black-Scholes European call price
double blackScholesCall(double S0, double K, double r, double q, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

    double d2 = d1 - sigma * std::sqrt(T);

    return S0 * std::exp(-q * T) * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
}

// Terminal stock price
double terminalStock(double S0, double r, double q, double sigma, double T, double z) {
    return S0 * std::exp((r - q - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
}

// Discounted European call payoff
double discountedCallPayoff(double S0, double K, double r, double q, double sigma, double T,
                            double z) {
    double ST = terminalStock(S0, r, q, sigma, T, z);
    return std::exp(-r * T) * std::max(ST - K, 0.0);
}

// Ordinary Monte Carlo benchmark
SimulationResult standardMonteCarlo(double S0, double K, double r, double q, double sigma, double T,
                                    int N, unsigned int seed, double zCritical) {
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sum = 0.0;
    double sumSquared = 0.0;

    for (int i = 0; i < N; ++i) {
        double z = normal(generator);
        double payoff = discountedCallPayoff(S0, K, r, q, sigma, T, z);

        sum += payoff;
        sumSquared += payoff * payoff;
    }

    double price = sum / N;
    double pathVariance = (sumSquared - N * price * price) / (N - 1);
    double estimatorVariance = pathVariance / N;
    double standardError = std::sqrt(estimatorVariance);

    double ciLower = price - zCritical * standardError;
    double ciUpper = price + zCritical * standardError;

    return {price, pathVariance, estimatorVariance, standardError, ciLower, ciUpper};
}

// Monte Carlo with Antithetic Variates
// N is the total number of payoff evaluations and must be even.
SimulationResult antitheticVariatesMonteCarlo(double S0, double K, double r, double q, double sigma,
                                              double T, int N, unsigned int seed,
                                              double zCritical) {

    int M = N / 2;

    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sumA = 0.0;
    double sumASquared = 0.0;

    for (int i = 0; i < M; ++i) {
        double z = normal(generator);

        double payoffPlus = discountedCallPayoff(S0, K, r, q, sigma, T, z);

        double payoffMinus = discountedCallPayoff(S0, K, r, q, sigma, T, -z);

        // Average of one antithetic pair
        double A = 0.5 * (payoffPlus + payoffMinus);

        sumA += A;
        sumASquared += A * A;
    }

    double price = sumA / M;

    // Variance of the independent pair averages A_i
    double pairVariance = (sumASquared - M * price * price) / (M - 1);

    // The estimator averages M independent pair averages
    double estimatorVariance = pairVariance / M;
    double standardError = std::sqrt(estimatorVariance);

    double ciLower = price - zCritical * standardError;
    double ciUpper = price + zCritical * standardError;

    return {price, pairVariance, estimatorVariance, standardError, ciLower, ciUpper};
}

int main() {
    // Option parameters
    double S0 = 90.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.00;
    double sigma = 0.20;
    double T = 1.0;

    // N must be even for antithetic pairing
    int N = 100000;

    if (N % 2 != 0) {
        std::cerr << "Error: N must be even for Antithetic Variates.\n";
        return 1;
    }

    int M = N / 2;

    // 95% confidence interval
    double confidenceLevel = 0.95;
    double zCritical = 1.959963984540054;

    // Independent seeds for the two estimators
    unsigned int standardMCSeed = 12345;
    unsigned int antitheticSeed = 54321;

    // --------------------------------------------------------
    // Option parameters: concise display
    // --------------------------------------------------------
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "European Call Option - Antithetic Variates\n";
    std::cout << "=============================================================\n";
    std::cout << "S0       = " << S0 << "\n";
    std::cout << "K        = " << K << "\n";
    std::cout << "r        = " << r << "\n";
    std::cout << "q        = " << q << "\n";
    std::cout << "sigma    = " << sigma << "\n";
    std::cout << "T        = " << T << "\n";
    std::cout << "Total payoff simulations N = " << N << "\n";
    std::cout << "Antithetic pairs M          = " << M << "\n";
    std::cout << "Confidence level = " << confidenceLevel * 100.0 << "%\n";

    SimulationResult standardResult =
        standardMonteCarlo(S0, K, r, q, sigma, T, N, standardMCSeed, zCritical);

    SimulationResult antitheticResult =
        antitheticVariatesMonteCarlo(S0, K, r, q, sigma, T, N, antitheticSeed, zCritical);

    double bsPrice = blackScholesCall(S0, K, r, q, sigma, T);

    // --------------------------------------------------------
    // Numerical results
    // --------------------------------------------------------
    std::cout << std::fixed << std::setprecision(8);

    std::cout << "\nPricing Results\n";
    std::cout << "================================================================================="
                 "=============================\n";

    std::cout << std::left << std::setw(25) << "Method" << std::right << std::setw(14) << "Price"
              << std::setw(16) << "Sample Var." << std::setw(18) << "Estimator Var."
              << std::setw(14) << "Std. Error" << std::setw(14) << "CI Lower" << std::setw(14)
              << "CI Upper"
              << "\n";

    std::cout << "---------------------------------------------------------------------------------"
                 "-----------------------------\n";

    std::cout << std::left << std::setw(25) << "Black-Scholes" << std::right << std::setw(14)
              << bsPrice << std::setw(16) << "-" << std::setw(18) << "-" << std::setw(14) << "-"
              << std::setw(14) << "-" << std::setw(14) << "-" << "\n";

    std::cout << std::left << std::setw(25) << "Standard Monte Carlo" << std::right << std::setw(14)
              << standardResult.price << std::setw(16) << standardResult.pathVariance
              << std::setw(18) << standardResult.estimatorVariance << std::setw(14)
              << standardResult.standardError << std::setw(14) << standardResult.ciLower
              << std::setw(14) << standardResult.ciUpper << "\n";

    std::cout << std::left << std::setw(25) << "Antithetic Variates" << std::right << std::setw(14)
              << antitheticResult.price << std::setw(16) << antitheticResult.pathVariance
              << std::setw(18) << antitheticResult.estimatorVariance << std::setw(14)
              << antitheticResult.standardError << std::setw(14) << antitheticResult.ciLower
              << std::setw(14) << antitheticResult.ciUpper << "\n";

    // For AV, compare estimator variances because MC averages N paths
    // while AV averages M = N/2 independent pair averages.
    double varianceRatio = antitheticResult.estimatorVariance / standardResult.estimatorVariance;

    double varianceReduction = (1.0 - varianceRatio) * 100.0;
    double varianceReductionFactor = 1.0 / varianceRatio;

    std::cout << "\nVariance Reduction\n";
    std::cout << "=============================================================\n";
    std::cout << "Variance ratio (AV / MC) = " << varianceRatio << "\n";
    std::cout << "Variance reduction       = " << varianceReduction << "%\n";

    return 0;
}