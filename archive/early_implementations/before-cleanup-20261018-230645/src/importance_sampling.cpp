#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// ============================================================
// Result container
// ============================================================

struct SimulationResult {
    double price;
    double payoffVariance;
    double estimatorVariance;
    double standardError;
    double ciLower;
    double ciUpper;
};

// ============================================================
// Basic Black-Scholes functions
// ============================================================

double normalCDF(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double blackScholesCall(double S0, double K, double r, double q, double sigma, double T) {

    double d1 = (std::log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

    double d2 = d1 - sigma * std::sqrt(T);

    return S0 * std::exp(-q * T) * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
}

double terminalStock(double S0, double r, double q, double sigma, double T, double z) {

    return S0 * std::exp((r - q - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
}

double discountedCallPayoff(double S0, double K, double r, double q, double sigma, double T,
                            double z) {

    double ST = terminalStock(S0, r, q, sigma, T, z);
    return std::exp(-r * T) * std::max(ST - K, 0.0);
}

// ============================================================
// Sample summary
// ============================================================

SimulationResult summarizeSamples(const std::vector<double> &values, double zCritical = 1.96) {

    int N = static_cast<int>(values.size());

    double sum = 0.0;
    double sumSquared = 0.0;

    for (double x : values) {
        sum += x;
        sumSquared += x * x;
    }

    double mean = sum / N;

    double payoffVariance = (sumSquared - N * mean * mean) / (N - 1);

    double estimatorVariance = payoffVariance / N;
    double standardError = std::sqrt(estimatorVariance);

    double ciLower = mean - zCritical * standardError;
    double ciUpper = mean + zCritical * standardError;

    return {mean, payoffVariance, estimatorVariance, standardError, ciLower, ciUpper};
}

// ============================================================
// Pilot objective for optimal theta
//
// V_hat(theta)
// = (1/M) sum Pi(Z_i)^2 exp(theta^2/2 - theta Z_i)
//
// The pilot Z_i are generated once and remain fixed.
// ============================================================

double objectiveV(const std::vector<double> &zPilot, const std::vector<double> &payoffPilot,
                  double theta) {

    double sum = 0.0;

    for (std::size_t i = 0; i < zPilot.size(); ++i) {
        double exponent = 0.5 * theta * theta - theta * zPilot[i];

        sum += payoffPilot[i] * payoffPilot[i] * std::exp(exponent);
    }

    return sum / static_cast<double>(zPilot.size());
}

double objectiveVPrime(const std::vector<double> &zPilot, const std::vector<double> &payoffPilot,
                       double theta) {

    double sum = 0.0;

    for (std::size_t i = 0; i < zPilot.size(); ++i) {
        double z = zPilot[i];
        double exponent = 0.5 * theta * theta - theta * z;

        sum += payoffPilot[i] * payoffPilot[i] * (theta - z) * std::exp(exponent);
    }

    return sum / static_cast<double>(zPilot.size());
}

double objectiveVSecond(const std::vector<double> &zPilot, const std::vector<double> &payoffPilot,
                        double theta) {

    double sum = 0.0;

    for (std::size_t i = 0; i < zPilot.size(); ++i) {
        double z = zPilot[i];
        double difference = theta - z;
        double exponent = 0.5 * theta * theta - theta * z;

        sum +=
            payoffPilot[i] * payoffPilot[i] * (1.0 + difference * difference) * std::exp(exponent);
    }

    return sum / static_cast<double>(zPilot.size());
}

// ============================================================
// Newton's method for optimal theta
// ============================================================

double findOptimalTheta(const std::vector<double> &zPilot, const std::vector<double> &payoffPilot,
                        double thetaInitial, int maxIterations = 50, double tolerance = 1e-8) {

    double theta = thetaInitial;

    std::cout << "\nNewton Optimization\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << std::setw(5) << "Iter" << std::setw(15) << "Theta" << std::setw(18) << "V(theta)"
              << std::setw(18) << "V'(theta)"
              << "\n";

    for (int k = 0; k < maxIterations; ++k) {

        double V = objectiveV(zPilot, payoffPilot, theta);
        double VPrime = objectiveVPrime(zPilot, payoffPilot, theta);
        double VSecond = objectiveVSecond(zPilot, payoffPilot, theta);

        std::cout << std::setw(5) << k << std::setw(15) << theta << std::setw(18) << V
                  << std::setw(18) << VPrime << "\n";

        if (std::abs(VPrime) < tolerance) {
            break;
        }

        if (VSecond <= 0.0) {
            std::cerr << "Newton method stopped: non-positive second derivative.\n";
            break;
        }

        double step = VPrime / VSecond;

        // Basic safeguard against an excessively large Newton step.
        step = std::clamp(step, -1.0, 1.0);

        double thetaNew = theta - step;

        if (std::abs(thetaNew - theta) < tolerance) {
            theta = thetaNew;
            break;
        }

        theta = thetaNew;
    }

    return theta;
}

// ============================================================
// Standard Monte Carlo
// ============================================================

SimulationResult standardMonteCarlo(double S0, double K, double r, double q, double sigma, double T,
                                    int N, unsigned int seed) {

    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> payoffs;
    payoffs.reserve(N);

    for (int i = 0; i < N; ++i) {
        double z = normal(generator);

        payoffs.push_back(discountedCallPayoff(S0, K, r, q, sigma, T, z));
    }

    return summarizeSamples(payoffs);
}

// ============================================================
// Importance Sampling Monte Carlo
//
// Proposal: Z ~ N(theta, 1)
// Weighted payoff:
// W = Pi(Z) exp(theta^2/2 - theta Z)
// ============================================================

SimulationResult importanceSamplingMonteCarlo(double S0, double K, double r, double q, double sigma,
                                              double T, double theta, int N, unsigned int seed) {

    std::mt19937 generator(seed);
    std::normal_distribution<double> proposal(theta, 1.0);

    std::vector<double> weightedPayoffs;
    weightedPayoffs.reserve(N);

    for (int i = 0; i < N; ++i) {

        double z = proposal(generator);

        double payoff = discountedCallPayoff(S0, K, r, q, sigma, T, z);

        double likelihoodRatio = std::exp(0.5 * theta * theta - theta * z);

        weightedPayoffs.push_back(payoff * likelihoodRatio);
    }

    return summarizeSamples(weightedPayoffs);
}

// ============================================================
// Main
// ============================================================

int main() {

    // --------------------------------------------------------
    // Option parameters
    // --------------------------------------------------------

    double S0 = 90.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.00;
    double sigma = 0.20;
    double T = 1.0;

    // Pilot sample used ONLY to estimate theta.
    int M = 10000;

    // Final number of simulations for MC and IS.
    int N = 100000;

    // Separate seeds ensure the pilot, MC, and IS samples are independent.
    unsigned int pilotSeed = 12345;
    unsigned int standardMCSeed = 54321;
    unsigned int importanceSamplingSeed = 98765;

    // --------------------------------------------------------
    // Display option parameters
    // --------------------------------------------------------

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "European Call Option\n";
    std::cout << "=================================================================\n";
    std::cout << "S0       = " << S0 << "\n";
    std::cout << "K        = " << K << "\n";
    std::cout << "r        = " << r << "\n";
    std::cout << "q        = " << q << "\n";
    std::cout << "sigma    = " << sigma << "\n";
    std::cout << "T        = " << T << "\n";
    std::cout << "Pilot simulations (M) = " << M << "\n";
    std::cout << "Pricing simulations (N) = " << N << "\n";

    std::cout << std::setprecision(8);

    // --------------------------------------------------------
    // Stage 1: fixed pilot sample for theta optimization
    // --------------------------------------------------------

    std::mt19937 pilotGenerator(pilotSeed);
    std::normal_distribution<double> standardNormal(0.0, 1.0);

    std::vector<double> zPilot(M);
    std::vector<double> payoffPilot(M);

    int positivePilotPayoffs = 0;

    for (int i = 0; i < M; ++i) {
        zPilot[i] = standardNormal(pilotGenerator);

        payoffPilot[i] = discountedCallPayoff(S0, K, r, q, sigma, T, zPilot[i]);

        if (payoffPilot[i] > 0.0) {
            ++positivePilotPayoffs;
        }
    }

    if (positivePilotPayoffs == 0) {
        std::cerr << "Pilot sample contains no positive call payoffs.\n";
        return 1;
    }

    // Initial shift: standardized point where S_T = K.
    double thetaInitial =
        (std::log(K / S0) - (r - q - 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

    thetaInitial = std::max(0.0, thetaInitial);

    double thetaOptimal = findOptimalTheta(zPilot, payoffPilot, thetaInitial);

    std::cout << "\nEstimated Optimal Shift\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "theta*      = " << thetaOptimal << "\n";
    std::cout << "V(theta*)   = " << objectiveV(zPilot, payoffPilot, thetaOptimal) << "\n";

    // --------------------------------------------------------
    // Stage 2: independent pricing samples
    // --------------------------------------------------------

    SimulationResult standardResult = standardMonteCarlo(S0, K, r, q, sigma, T, N, standardMCSeed);

    SimulationResult importanceResult = importanceSamplingMonteCarlo(
        S0, K, r, q, sigma, T, thetaOptimal, N, importanceSamplingSeed);

    double bsPrice = blackScholesCall(S0, K, r, q, sigma, T);

    // --------------------------------------------------------
    // Pricing results
    // --------------------------------------------------------

    std::cout << "\nPricing Results\n";
    std::cout << "================================================================================="
                 "================================\n";

    std::cout << std::left << std::setw(25) << "Method" << std::right << std::setw(14) << "Price"
              << std::setw(17) << "Payoff Var." << std::setw(18) << "Estimator Var."
              << std::setw(15) << "Std. Error" << std::setw(16) << "95% CI Low" << std::setw(16)
              << "95% CI High"
              << "\n";

    std::cout << "---------------------------------------------------------------------------------"
                 "--------------------------------\n";

    std::cout << std::left << std::setw(25) << "Black-Scholes" << std::right << std::setw(14)
              << bsPrice << std::setw(17) << "-" << std::setw(18) << "-" << std::setw(15) << "-"
              << std::setw(16) << "-" << std::setw(16) << "-"
              << "\n";

    std::cout << std::left << std::setw(25) << "Standard Monte Carlo" << std::right << std::setw(14)
              << standardResult.price << std::setw(17) << standardResult.payoffVariance
              << std::setw(18) << standardResult.estimatorVariance << std::setw(15)
              << standardResult.standardError << std::setw(16) << standardResult.ciLower
              << std::setw(16) << standardResult.ciUpper << "\n";

    std::cout << std::left << std::setw(25) << "Importance Sampling" << std::right << std::setw(14)
              << importanceResult.price << std::setw(17) << importanceResult.payoffVariance
              << std::setw(18) << importanceResult.estimatorVariance << std::setw(15)
              << importanceResult.standardError << std::setw(16) << importanceResult.ciLower
              << std::setw(16) << importanceResult.ciUpper << "\n";

    // --------------------------------------------------------
    // Variance reduction
    // --------------------------------------------------------

    double varianceRatio = importanceResult.payoffVariance / standardResult.payoffVariance;

    double varianceReduction = (1.0 - varianceRatio) * 100.0;

    double varianceReductionFactor =
        standardResult.payoffVariance / importanceResult.payoffVariance;

    double standardErrorReductionFactor =
        standardResult.standardError / importanceResult.standardError;

    std::cout << "\nVariance Reduction\n";
    std::cout << "=================================================================\n";
    std::cout << "Variance ratio (IS / MC)     = " << varianceRatio << "\n";
    std::cout << "Variance reduction           = " << varianceReduction << "%\n";
    std::cout << "Variance reduction factor    = " << varianceReductionFactor << "x\n";
    std::cout << "Std. error reduction factor  = " << standardErrorReductionFactor << "x\n";

    return 0;
}