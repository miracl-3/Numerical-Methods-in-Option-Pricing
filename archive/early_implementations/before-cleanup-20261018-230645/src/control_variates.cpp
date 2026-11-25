#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

struct SimulationResult {
    double price;
    double pathVariance;
    double estimatorVariance;
    double standardError;
    double ciLower;
    double ciUpper;
};

struct ControlVariateResult {
    SimulationResult stats;
    double alphaOptimal;
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

// Ordinary Monte Carlo benchmark
SimulationResult standardMonteCarlo(double S0, double K, double r, double q, double sigma, double T,
                                    int N, unsigned int seed, double zCritical) {
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sum = 0.0;
    double sumSquared = 0.0;

    for (int i = 0; i < N; ++i) {
        double z = normal(generator);
        double ST = terminalStock(S0, r, q, sigma, T, z);
        double discountedPayoff = std::exp(-r * T) * std::max(ST - K, 0.0);

        sum += discountedPayoff;
        sumSquared += discountedPayoff * discountedPayoff;
    }

    double price = sum / N;
    double pathVariance = (sumSquared - N * price * price) / (N - 1);
    double estimatorVariance = pathVariance / N;
    double standardError = std::sqrt(estimatorVariance);

    double ciLower = price - zCritical * standardError;
    double ciUpper = price + zCritical * standardError;

    return {price, pathVariance, estimatorVariance, standardError, ciLower, ciUpper};
}

// Monte Carlo with Control Variates
ControlVariateResult controlVariatesMonteCarlo(double S0, double K, double r, double q,
                                               double sigma, double T, int N, unsigned int seed,
                                               double zCritical) {

    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> stock(N);
    std::vector<double> payoff(N);

    // Known mean of the control variable Y = S_T
    double muY = S0 * std::exp((r - q) * T);

    double sumY = 0.0;
    double sumPayoff = 0.0;

    for (int i = 0; i < N; ++i) {
        double z = normal(generator);

        stock[i] = terminalStock(S0, r, q, sigma, T, z);
        payoff[i] = std::max(stock[i] - K, 0.0); // undiscounted payoff

        sumY += stock[i];
        sumPayoff += payoff[i];
    }

    double meanY = sumY / N;
    double meanPayoff = sumPayoff / N;

    // Estimate alpha* = -Cov(Pi, Y) / Var(Y)
    double covarianceSum = 0.0;
    double varianceYSum = 0.0;

    for (int i = 0; i < N; ++i) {
        covarianceSum += (payoff[i] - meanPayoff) * (stock[i] - meanY);

        varianceYSum += (stock[i] - meanY) * (stock[i] - meanY);
    }

    double alphaOptimal = -covarianceSum / varianceYSum;

    // Control-variate estimator
    double discount = std::exp(-r * T);
    double price = discount * (meanPayoff + alphaOptimal * (meanY - muY));

    // Per-path adjusted discounted observations
    double adjustedVarianceSum = 0.0;

    for (int i = 0; i < N; ++i) {
        double adjustedValue = discount * (payoff[i] + alphaOptimal * (stock[i] - muY));

        adjustedVarianceSum += (adjustedValue - price) * (adjustedValue - price);
    }

    double pathVariance = adjustedVarianceSum / (N - 1);
    double estimatorVariance = pathVariance / N;
    double standardError = std::sqrt(estimatorVariance);

    double ciLower = price - zCritical * standardError;
    double ciUpper = price + zCritical * standardError;

    SimulationResult stats{price, pathVariance, estimatorVariance, standardError, ciLower, ciUpper};

    return {stats, alphaOptimal};
}

int main() {
    // Option parameters
    double S0 = 90.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.00;
    double sigma = 0.20;
    double T = 1.0;

    int N = 100000;

    // 95% confidence interval
    double confidenceLevel = 0.95;
    double zCritical = 1.959963984540054;

    // Independent seeds for the two final estimators
    unsigned int standardMCSeed = 12345;
    unsigned int controlVariateSeed = 54321;

    // --------------------------------------------------------
    // Option parameters: concise display
    // --------------------------------------------------------
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "European Call Option - Control Variates\n";
    std::cout << "=============================================================\n";
    std::cout << "S0       = " << S0 << "\n";
    std::cout << "K        = " << K << "\n";
    std::cout << "r        = " << r << "\n";
    std::cout << "q        = " << q << "\n";
    std::cout << "sigma    = " << sigma << "\n";
    std::cout << "T        = " << T << "\n";
    std::cout << "N        = " << N << "\n";
    std::cout << "Confidence level = " << confidenceLevel * 100.0 << "%\n";

    SimulationResult standardResult =
        standardMonteCarlo(S0, K, r, q, sigma, T, N, standardMCSeed, zCritical);

    ControlVariateResult cvResult =
        controlVariatesMonteCarlo(S0, K, r, q, sigma, T, N, controlVariateSeed, zCritical);

    double bsPrice = blackScholesCall(S0, K, r, q, sigma, T);

    // --------------------------------------------------------
    // Numerical results
    // --------------------------------------------------------
    std::cout << std::fixed << std::setprecision(8);

    std::cout << "\nEstimated optimal control coefficient\n";
    std::cout << "-------------------------------------------------------------\n";
    std::cout << "alpha* = " << cvResult.alphaOptimal << "\n";

    std::cout << "\nPricing Results\n";
    std::cout << "================================================================================="
                 "=============================\n";

    std::cout << std::left << std::setw(25) << "Method" << std::right << std::setw(14) << "Price"
              << std::setw(16) << "Path Var." << std::setw(18) << "Estimator Var." << std::setw(14)
              << "Std. Error" << std::setw(14) << "CI Lower" << std::setw(14) << "CI Upper"
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

    std::cout << std::left << std::setw(25) << "Control Variates" << std::right << std::setw(14)
              << cvResult.stats.price << std::setw(16) << cvResult.stats.pathVariance
              << std::setw(18) << cvResult.stats.estimatorVariance << std::setw(14)
              << cvResult.stats.standardError << std::setw(14) << cvResult.stats.ciLower
              << std::setw(14) << cvResult.stats.ciUpper << "\n";

    double varianceRatio = cvResult.stats.estimatorVariance / standardResult.estimatorVariance;

    double varianceReduction = (1.0 - varianceRatio) * 100.0;
    double varianceReductionFactor = 1.0 / varianceRatio;

    std::cout << "\nVariance Reduction\n";
    std::cout << "=============================================================\n";
    std::cout << "Variance ratio (CV / MC) = " << varianceRatio << "\n";
    std::cout << "Variance reduction       = " << varianceReduction << "%\n";

    return 0;
}