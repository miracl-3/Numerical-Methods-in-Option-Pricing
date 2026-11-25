// Necessary libraries
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
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

// Standard normal cumulative distribution function
double normal_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Black-Scholes price for either a call or a put
double black_scholes_price(double S0, double K, double r, double q, double sigma, double T,
                           bool isCall) {

    const double sqrtT = std::sqrt(T);

    const double d1 = (std::log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);

    const double d2 = d1 - sigma * sqrtT;

    if (isCall) {
        return S0 * std::exp(-q * T) * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
    }

    return K * std::exp(-r * T) * normal_cdf(-d2) - S0 * std::exp(-q * T) * normal_cdf(-d1);
}

// Monte Carlo price, standard error, and confidence interval
void monte_carlo_option_price(double S0, double K, double r, double q, double sigma, double T,
                              int numberOfSimulations, double alpha, bool isCall, unsigned int seed,
                              double &estimatedPrice, double &standardError, double &ciLower,
                              double &ciUpper) {
    std::mt19937 generator(seed);
    std::normal_distribution<double> normalDistribution(0.0, 1.0);

    double sum = 0.0;
    double sumSquared = 0.0;

    const double discountFactor = std::exp(-r * T);
    const double sqrtT = std::sqrt(T);

    for (int i = 0; i < numberOfSimulations; ++i) {

        // Generate Z_i ~ N(0,1)
        const double z = normalDistribution(generator);

        // Exact terminal stock price under the risk-neutral measure
        const double terminalStockPrice =
            S0 * std::exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrtT * z);

        double payoff = 0.0;

        if (isCall) {
            payoff = std::max(terminalStockPrice - K, 0.0);
        } else {
            payoff = std::max(K - terminalStockPrice, 0.0);
        }

        // Discounted payoff
        const double discountedPayoff = discountFactor * payoff;

        sum += discountedPayoff;
        sumSquared += discountedPayoff * discountedPayoff;
    }

    estimatedPrice = sum / static_cast<double>(numberOfSimulations);

    double estimatedVariance =
        (sumSquared - static_cast<double>(numberOfSimulations) * estimatedPrice * estimatedPrice) /
        static_cast<double>(numberOfSimulations - 1);

    // Prevent a tiny negative number due to floating-point rounding
    if (estimatedVariance < 0.0) {
        estimatedVariance = 0.0;
    }

    standardError = std::sqrt(estimatedVariance / static_cast<double>(numberOfSimulations));

    // For a two-sided confidence interval:
    // z_(1-alpha/2) = sqrt(2) erfinv(1-alpha)
    const double criticalValue = std::sqrt(2.0) * erfinv(1.0 - alpha);

    const double halfWidth = criticalValue * standardError;

    ciLower = estimatedPrice - halfWidth;
    ciUpper = estimatedPrice + halfWidth;
}

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

void print_option_table(bool isCall) {

    // Parameter combinations from the reference tables
    const double stockPrices[] = {90.0, 90.0, 100.0, 100.0, 110.0, 110.0};

    const double maturities[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0};

    const int numberOfCases = sizeof(stockPrices) / sizeof(stockPrices[0]);

    const double K = 100.0;
    const double r = 0.05;
    const double q = 0.0;
    const double sigma = 0.20;

    const int numberOfSimulations = 10000;
    const double alpha = 0.05;

    if (isCall) {
        std::cout << "\nCALL OPTION\n";
    } else {
        std::cout << "\nPUT OPTION\n";
    }

    std::cout << "Assumptions: r = 5%, sigma = 20%, q = 0, "
              << "N = " << numberOfSimulations << "\n\n";

    std::cout << std::setw(8) << "S(0)" << std::setw(8) << "K" << std::setw(8) << "T"
              << std::setw(14) << "BS Price" << std::setw(14) << "MC Price" << std::setw(14)
              << "MC S.E." << '\n';

    std::cout << std::string(66, '-') << '\n';

    for (int i = 0; i < numberOfCases; ++i) {

        const double S0 = stockPrices[i];
        const double T = maturities[i];

        const double bsPrice = black_scholes_price(S0, K, r, q, sigma, T, isCall);

        double mcPrice = 0.0;
        double standardError = 0.0;
        double ciLower = 0.0;
        double ciUpper = 0.0;

        monte_carlo_option_price(S0, K, r, q, sigma, T, numberOfSimulations, alpha, isCall, 42 + i,
                                 mcPrice, standardError, ciLower, ciUpper);

        std::cout << std::fixed << std::setprecision(2);

        std::cout << std::setw(8) << S0 << std::setw(8) << K << std::setw(8) << T << std::setw(14)
                  << bsPrice << std::setw(14) << mcPrice << std::setw(14) << standardError << '\n';
    }
}

void show_confidence_interval_convergence() {

    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double q = 0.0;
    const double sigma = 0.20;
    const double T = 1.0;

    const double alpha = 0.05;
    const bool isCall = true;

    const int simulationLevels[] = {100, 1000, 10000, 100000, 1000000};

    const int numberOfLevels = sizeof(simulationLevels) / sizeof(simulationLevels[0]);

    const int maximumSimulations = simulationLevels[numberOfLevels - 1];

    std::mt19937 generator(42);
    std::normal_distribution<double> normalDistribution(0.0, 1.0);

    const double discountFactor = std::exp(-r * T);
    const double sqrtT = std::sqrt(T);

    const double criticalValue = std::sqrt(2.0) * erfinv(1.0 - alpha);

    const double bsPrice = black_scholes_price(S0, K, r, q, sigma, T, isCall);

    double sum = 0.0;
    double sumSquared = 0.0;

    int currentLevel = 0;

    std::cout << "\nCONFIDENCE INTERVAL CONVERGENCE\n";
    std::cout << "European call: S(0) = 100, K = 100, T = 1\n";
    std::cout << "Black-Scholes benchmark: " << std::fixed << std::setprecision(6) << bsPrice
              << "\n\n";

    std::cout << std::setw(12) << "N" << std::setw(14) << "MC Price" << std::setw(14)
              << "Std. Error" << std::setw(14) << "CI Lower" << std::setw(14) << "CI Upper"
              << std::setw(14) << "CI Width" << '\n';

    std::cout << std::string(82, '-') << '\n';

    for (int n = 1; n <= maximumSimulations; ++n) {

        const double z = normalDistribution(generator);

        const double terminalStockPrice =
            S0 * std::exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrtT * z);

        const double payoff = std::max(terminalStockPrice - K, 0.0);

        const double discountedPayoff = discountFactor * payoff;

        sum += discountedPayoff;
        sumSquared += discountedPayoff * discountedPayoff;

        if (n == simulationLevels[currentLevel]) {

            const double estimatedPrice = sum / static_cast<double>(n);

            double estimatedVariance =
                (sumSquared - static_cast<double>(n) * estimatedPrice * estimatedPrice) /
                static_cast<double>(n - 1);

            if (estimatedVariance < 0.0) {
                estimatedVariance = 0.0;
            }

            const double standardError = std::sqrt(estimatedVariance / static_cast<double>(n));

            const double halfWidth = criticalValue * standardError;

            const double ciLower = estimatedPrice - halfWidth;

            const double ciUpper = estimatedPrice + halfWidth;

            const double ciWidth = ciUpper - ciLower;

            std::cout << std::fixed << std::setprecision(6);

            std::cout << std::setw(12) << n << std::setw(14) << estimatedPrice << std::setw(14)
                      << standardError << std::setw(14) << ciLower << std::setw(14) << ciUpper
                      << std::setw(14) << ciWidth << '\n';

            ++currentLevel;

            if (currentLevel == numberOfLevels) {
                break;
            }
        }
    }
}

// =========================================================
// Importance Sampling for European Option
// =========================================================

double importance_sampling_option_price(double S0, double K, double r, double q, double sigma,
                                        double T, int numberOfSimulations, double shift,
                                        bool isCall, unsigned int seed) {

    std::mt19937 generator(seed);
    std::normal_distribution<double> normalDistribution(0.0, 1.0);

    const double discountFactor = std::exp(-r * T);
    const double sqrtT = std::sqrt(T);

    double sum = 0.0;

    for (int i = 0; i < numberOfSimulations; ++i) {

        // Generate Y ~ N(0,1)
        const double y = normalDistribution(generator);

        // Shift to Z ~ N(shift,1)
        const double z = y + shift;

        // Terminal stock price
        const double terminalStockPrice =
            S0 * std::exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrtT * z);

        double payoff = 0.0;

        if (isCall) {
            payoff = std::max(terminalStockPrice - K, 0.0);
        } else {
            payoff = std::max(K - terminalStockPrice, 0.0);
        }

        // Likelihood ratio f(z) / q(z)
        const double likelihoodRatio = std::exp(-shift * y - 0.5 * shift * shift);

        // Weighted discounted payoff
        const double weightedPayoff = discountFactor * payoff * likelihoodRatio;

        sum += weightedPayoff;
    }

    return sum / static_cast<double>(numberOfSimulations);
}

// =========================================================
// Batch Comparison:
// Standard Monte Carlo vs Importance Sampling
// =========================================================

void compare_batch_mc_is() {

    // -----------------------------------------------------
    // Option parameters
    // -----------------------------------------------------

    const double S0 = 100.0;
    const double K = 150.0;
    const double r = 0.05;
    const double q = 0.0;
    const double sigma = 0.20;
    const double T = 1.0;

    const bool isCall = true;

    // -----------------------------------------------------
    // Simulation parameters
    // -----------------------------------------------------

    const int numberOfSimulations = 10000;
    const int numberOfBatches = 100;

    // Proposal distribution: N(2,1)
    const double shift = -4.0;

    const double alpha = 0.05;

    // -----------------------------------------------------
    // Black-Scholes benchmark
    // -----------------------------------------------------

    const double bsPrice = black_scholes_price(S0, K, r, q, sigma, T, isCall);

    // -----------------------------------------------------
    // Accumulated batch statistics
    // -----------------------------------------------------

    double mcSum = 0.0;
    double mcSumSquared = 0.0;

    double isSum = 0.0;
    double isSumSquared = 0.0;

    // =====================================================
    // Initial information
    // =====================================================

    std::cout << '\n';
    std::cout << std::string(94, '=') << '\n';
    std::cout << "                 BATCH MONTE CARLO VS IMPORTANCE SAMPLING\n";
    std::cout << std::string(94, '=') << '\n';

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Option           : European Call\n";
    std::cout << "S0               : " << S0 << '\n';
    std::cout << "K                : " << K << '\n';
    std::cout << "T                : " << T << '\n';
    std::cout << "r                : " << r * 100.0 << "%\n";
    std::cout << "sigma            : " << sigma * 100.0 << "%\n";
    std::cout << "Simulations N    : " << numberOfSimulations << " per batch\n";
    std::cout << "Batches M        : " << numberOfBatches << '\n';
    std::cout << "IS shift mu      : " << shift << '\n';

    std::cout << std::setprecision(8);
    std::cout << "Black-Scholes    : " << bsPrice << '\n';

    // =====================================================
    // Batch-by-batch output
    // =====================================================

    std::cout << "\nBATCH-BY-BATCH RESULTS\n";
    std::cout << std::string(94, '-') << '\n';

    std::cout << std::setw(7) << "Batch" << std::setw(15) << "MC Price" << std::setw(15)
              << "IS Price" << std::setw(18) << "MC Run. Var." << std::setw(18) << "IS Run. Var."
              << std::setw(17) << "IS/MC Var." << '\n';

    std::cout << std::string(94, '-') << '\n';

    // =====================================================
    // Run independent batches
    // =====================================================

    for (int batch = 0; batch < numberOfBatches; ++batch) {

        const unsigned int seed = 42 + batch;

        // -------------------------------------------------
        // Standard Monte Carlo
        // -------------------------------------------------

        double mcPrice = 0.0;
        double mcSE = 0.0;
        double mcLower = 0.0;
        double mcUpper = 0.0;

        monte_carlo_option_price(S0, K, r, q, sigma, T, numberOfSimulations, alpha, isCall, seed,
                                 mcPrice, mcSE, mcLower, mcUpper);

        // -------------------------------------------------
        // Importance Sampling
        // -------------------------------------------------

        const double isPrice = importance_sampling_option_price(
            S0, K, r, q, sigma, T, numberOfSimulations, shift, isCall, seed);

        // -------------------------------------------------
        // Update accumulated statistics
        // -------------------------------------------------

        mcSum += mcPrice;
        mcSumSquared += mcPrice * mcPrice;

        isSum += isPrice;
        isSumSquared += isPrice * isPrice;

        const int completedBatches = batch + 1;

        const double runningMcMean = mcSum / static_cast<double>(completedBatches);

        const double runningIsMean = isSum / static_cast<double>(completedBatches);

        // -------------------------------------------------
        // Running variance requires at least two batches
        // -------------------------------------------------

        if (completedBatches >= 2) {

            double runningMcVariance =
                (mcSumSquared - completedBatches * runningMcMean * runningMcMean) /
                static_cast<double>(completedBatches - 1);

            double runningIsVariance =
                (isSumSquared - completedBatches * runningIsMean * runningIsMean) /
                static_cast<double>(completedBatches - 1);

            // Prevent small negative values from rounding
            if (runningMcVariance < 0.0) {
                runningMcVariance = 0.0;
            }

            if (runningIsVariance < 0.0) {
                runningIsVariance = 0.0;
            }

            const double runningVarianceRatio = runningIsVariance / runningMcVariance;

            // Print current batch
            std::cout << std::fixed << std::setprecision(8)

                      << std::setw(7) << completedBatches

                      << std::setw(15) << mcPrice

                      << std::setw(15) << isPrice

                      << std::scientific << std::setprecision(5)

                      << std::setw(18) << runningMcVariance

                      << std::setw(18) << runningIsVariance

                      << std::fixed << std::setprecision(4)

                      << std::setw(17) << runningVarianceRatio

                      << '\n'
                      << std::flush;
        }

        else {

            // First batch: variance is not defined
            std::cout << std::fixed << std::setprecision(8)

                      << std::setw(7) << completedBatches

                      << std::setw(15) << mcPrice

                      << std::setw(15) << isPrice

                      << std::setw(18) << "-"

                      << std::setw(18) << "-"

                      << std::setw(17) << "-"

                      << '\n'
                      << std::flush;
        }
    }

    // =====================================================
    // Final batch statistics
    // =====================================================

    const double mcMean = mcSum / static_cast<double>(numberOfBatches);

    const double isMean = isSum / static_cast<double>(numberOfBatches);

    const double mcVariance = (mcSumSquared - numberOfBatches * mcMean * mcMean) /
                              static_cast<double>(numberOfBatches - 1);

    const double isVariance = (isSumSquared - numberOfBatches * isMean * isMean) /
                              static_cast<double>(numberOfBatches - 1);

    const double varianceRatio = isVariance / mcVariance;

    // const double varianceReductionFactor = mcVariance / isVariance;

    // =====================================================
    // Final summary
    // =====================================================

    std::cout << std::string(94, '-') << '\n';

    std::cout << "\nFINAL SUMMARY\n";
    std::cout << std::string(80, '-') << '\n';

    std::cout << std::left << std::setw(24) << "Method" << std::right << std::setw(18)
              << "Mean Price" << std::setw(18) << "Abs. Error" << std::setw(20) << "Batch Variance"
              << '\n';

    std::cout << std::string(80, '-') << '\n';

    std::cout << std::left << std::setw(24) << "Standard MC + Batch"

              << std::right << std::fixed << std::setprecision(8)

              << std::setw(18) << mcMean

              << std::setw(18) << std::fabs(mcMean - bsPrice)

              << std::scientific << std::setprecision(6)

              << std::setw(20) << mcVariance

              << '\n';

    std::cout << std::left << std::setw(24) << "IS + Batch"

              << std::right << std::fixed << std::setprecision(8)

              << std::setw(18) << isMean

              << std::setw(18) << std::fabs(isMean - bsPrice)

              << std::scientific << std::setprecision(6)

              << std::setw(20) << isVariance

              << '\n';

    std::cout << std::string(80, '-') << '\n';

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Variance ratio Var(IS) / Var(MC) : " << varianceRatio << '\n';

    // std::cout << "Variance reduction factor        : " << varianceReductionFactor << "x\n";

    std::cout << std::string(80, '=') << '\n';
}

int main() {

    // ---------------------------------------------------------
    // Previous Monte Carlo integration experiment
    // ---------------------------------------------------------

    // monte_carlo_integration(0.05, 0.0001, 0.0, 1.0);

    // ---------------------------------------------------------
    // Experiment 1: multiple option parameter combinations
    // ---------------------------------------------------------

    // print_option_table(true);
    // print_option_table(false);

    // ---------------------------------------------------------
    // Experiment 2: confidence interval convergence
    // ---------------------------------------------------------

    // show_confidence_interval_convergence();

    // ---------------------------------------------------------
    // Experiment 3:
    // Batch Standard MC vs Batch Importance Sampling
    // ---------------------------------------------------------

    compare_batch_mc_is();

    return 0;
}