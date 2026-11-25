#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace std;

// ==========================================================
// Put payoff
// ==========================================================
double putPayoff(double S, double K) { return max(K - S, 0.0); }

// ==========================================================
// Standard normal CDF
// ==========================================================
double normalCDF(double x) { return 0.5 * erfc(-x / sqrt(2.0)); }

// ==========================================================
// European put: Black-Scholes
// ==========================================================
double europeanPutBlackScholes(double S0, double K, double r, double q, double sigma, double T) {
    double d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));

    double d2 = d1 - sigma * sqrt(T);

    return K * exp(-r * T) * normalCDF(-d2) - S0 * exp(-q * T) * normalCDF(-d1);
}

// ==========================================================
// CRR American put
// ==========================================================
double americanPutBinomial(double S0, double K, double r, double q, double sigma, double T, int M) {
    double dt = T / M;

    double u = exp(sigma * sqrt(dt));
    double d = 1.0 / u;

    double p = (exp((r - q) * dt) - d) / (u - d);

    double discount = exp(-r * dt);

    vector<double> value(M + 1);

    double ratio = u / d;

    // Terminal stock prices
    double S = S0 * pow(d, M);

    for (int j = 0; j <= M; ++j) {
        value[j] = putPayoff(S, K);
        S *= ratio;
    }

    // Backward induction
    for (int i = M - 1; i >= 0; --i) {
        S = S0 * pow(d, i);

        for (int j = 0; j <= i; ++j) {
            double continuation = discount * (p * value[j + 1] + (1.0 - p) * value[j]);

            double exercise = putPayoff(S, K);

            value[j] = max(exercise, continuation);

            S *= ratio;
        }
    }

    return value[0];
}

// ==========================================================
// Solve 3x3 system
// ==========================================================
array<double, 3> solve3x3(array<array<double, 4>, 3> A) {
    const int n = 3;

    for (int i = 0; i < n; ++i) {
        int pivot = i;

        for (int j = i + 1; j < n; ++j) {
            if (abs(A[j][i]) > abs(A[pivot][i]))
                pivot = j;
        }

        if (abs(A[pivot][i]) < 1e-12)
            throw runtime_error("Singular regression matrix.");

        swap(A[i], A[pivot]);

        for (int j = i + 1; j < n; ++j) {
            double factor = A[j][i] / A[i][i];

            for (int k = i; k <= n; ++k) {
                A[j][k] -= factor * A[i][k];
            }
        }
    }

    array<double, 3> beta{};

    for (int i = n - 1; i >= 0; --i) {
        double sum = A[i][3];

        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * beta[j];
        }

        beta[i] = sum / A[i][i];
    }

    return beta;
}

// ==========================================================
// Quadratic regression
//
// y = beta0 + beta1*x + beta2*x^2
// ==========================================================
array<double, 3> quadraticRegression(const vector<double> &x, const vector<double> &y) {
    if (x.size() != y.size())
        throw runtime_error("x and y size mismatch.");

    if (x.size() < 3)
        throw runtime_error("Not enough observations.");

    double n = static_cast<double>(x.size());

    double sumX = 0.0;
    double sumX2 = 0.0;
    double sumX3 = 0.0;
    double sumX4 = 0.0;

    double sumY = 0.0;
    double sumXY = 0.0;
    double sumX2Y = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        double xi = x[i];
        double yi = y[i];

        double x2 = xi * xi;
        double x3 = x2 * xi;
        double x4 = x2 * x2;

        sumX += xi;
        sumX2 += x2;
        sumX3 += x3;
        sumX4 += x4;

        sumY += yi;
        sumXY += xi * yi;
        sumX2Y += x2 * yi;
    }

    array<array<double, 4>, 3> A = {
        {{n, sumX, sumX2, sumY}, {sumX, sumX2, sumX3, sumXY}, {sumX2, sumX3, sumX4, sumX2Y}}};

    return solve3x3(A);
}

double predict(double x, const array<double, 3> &beta) {
    return beta[0] + beta[1] * x + beta[2] * x * x;
}

// ==========================================================
// Simulate GBM paths under Q
// ==========================================================
vector<vector<double>> simulatePaths(double S0, double r, double q, double sigma, double T, int M,
                                     int N, unsigned long long seed) {
    double dt = T / M;

    double drift = (r - q - 0.5 * sigma * sigma) * dt;

    double diffusion = sigma * sqrt(dt);

    mt19937_64 generator(seed);
    normal_distribution<double> normal(0.0, 1.0);

    vector<vector<double>> paths(N, vector<double>(M + 1));

    for (int n = 0; n < N; ++n) {
        paths[n][0] = S0;

        for (int i = 1; i <= M; ++i) {
            double Z = normal(generator);

            paths[n][i] = paths[n][i - 1] * exp(drift + diffusion * Z);
        }
    }

    return paths;
}

// ==========================================================
// Regression information at each exercise date
// ==========================================================
struct RegressionModel {
    array<double, 3> beta{};
    bool valid = false;
};

struct LSMPolicy {
    vector<RegressionModel> model;

    bool exerciseAtTime0 = false;

    double trainingContinuation0 = 0.0;
};

// ==========================================================
// Train LSM stopping rule
// ==========================================================
LSMPolicy trainLSM(const vector<vector<double>> &paths, double K, double r, double T) {
    int N = static_cast<int>(paths.size());
    int M = static_cast<int>(paths[0].size()) - 1;

    double dt = T / M;

    vector<double> cashflow(N);
    vector<int> exerciseTime(N, M);

    LSMPolicy policy;

    policy.model.resize(M + 1);

    // ------------------------------------------------------
    // Terminal payoff
    // ------------------------------------------------------
    for (int n = 0; n < N; ++n) {
        cashflow[n] = putPayoff(paths[n][M], K);
    }

    // ------------------------------------------------------
    // Backward regression
    // ------------------------------------------------------
    for (int i = M - 1; i >= 1; --i) {
        vector<double> x;
        vector<double> y;

        x.reserve(N);
        y.reserve(N);

        // Construct regression data
        for (int n = 0; n < N; ++n) {
            double S = paths[n][i];

            double exercise = putPayoff(S, K);

            // ITM paths only
            if (exercise > 0.0) {
                // Normalize S by K
                double state = S / K;

                // Discount currently selected future
                // cash flow back to time i
                double futureValue = cashflow[n] * exp(-r * (exerciseTime[n] - i) * dt);

                x.push_back(state);
                y.push_back(futureValue);
            }
        }

        // Need enough points for quadratic regression
        if (x.size() < 3)
            continue;

        array<double, 3> beta = quadraticRegression(x, y);

        policy.model[i].beta = beta;
        policy.model[i].valid = true;

        // --------------------------------------------------
        // Exercise / continuation decision
        // --------------------------------------------------
        for (int n = 0; n < N; ++n) {
            double S = paths[n][i];

            double exercise = putPayoff(S, K);

            if (exercise <= 0.0)
                continue;

            double state = S / K;

            double continuation = predict(state, beta);

            // Continuation values should be non-negative
            continuation = max(continuation, 0.0);

            if (exercise > continuation) {
                cashflow[n] = exercise;
                exerciseTime[n] = i;
            }
        }
    }

    // ------------------------------------------------------
    // Estimate continuation value at t = 0
    // ------------------------------------------------------
    double continuation0 = 0.0;

    for (int n = 0; n < N; ++n) {
        continuation0 += cashflow[n] * exp(-r * exerciseTime[n] * dt);
    }

    continuation0 /= N;

    policy.trainingContinuation0 = continuation0;

    double immediate0 = putPayoff(paths[0][0], K);

    policy.exerciseAtTime0 = (immediate0 > continuation0);

    return policy;
}

// ==========================================================
// LSM pricing result
// ==========================================================
struct LSMResult {
    double price = 0.0;
    double standardError = 0.0;
};

// ==========================================================
// Price learned stopping rule on independent paths
// ==========================================================
LSMResult evaluateLSM(const vector<vector<double>> &paths, const LSMPolicy &policy, double K,
                      double r, double T) {
    int N = static_cast<int>(paths.size());
    int M = static_cast<int>(paths[0].size()) - 1;

    double dt = T / M;

    vector<double> discountedPayoff(N);

    // If training determined exercise immediately
    if (policy.exerciseAtTime0) {
        double payoff = putPayoff(paths[0][0], K);

        fill(discountedPayoff.begin(), discountedPayoff.end(), payoff);
    }

    else {
        for (int n = 0; n < N; ++n) {
            bool exercised = false;

            // Move forward along test path
            for (int i = 1; i < M; ++i) {
                double S = paths[n][i];

                double exercise = putPayoff(S, K);

                if (exercise <= 0.0)
                    continue;

                if (!policy.model[i].valid)
                    continue;

                double state = S / K;

                double continuation = predict(state, policy.model[i].beta);

                continuation = max(continuation, 0.0);

                if (exercise > continuation) {
                    discountedPayoff[n] = exp(-r * i * dt) * exercise;

                    exercised = true;

                    break;
                }
            }

            // If never exercised early,
            // receive terminal payoff
            if (!exercised) {
                discountedPayoff[n] = exp(-r * T) * putPayoff(paths[n][M], K);
            }
        }
    }

    // ------------------------------------------------------
    // Mean
    // ------------------------------------------------------
    double mean = 0.0;

    for (double x : discountedPayoff)
        mean += x;

    mean /= N;

    // ------------------------------------------------------
    // Sample variance and SE
    // ------------------------------------------------------
    double variance = 0.0;

    for (double x : discountedPayoff) {
        double diff = x - mean;
        variance += diff * diff;
    }

    variance /= (N - 1);

    double SE = sqrt(variance / N);

    return {mean, SE};
}

// ==========================================================
// Main
// ==========================================================
int main() {
    // ------------------------------------------------------
    // Same contract parameters as CRR experiment
    // ------------------------------------------------------
    double S0 = 50.0;
    double K = 55.0;
    double r = 0.05;
    double q = 0.0;
    double sigma = 0.20;
    double T = 1.0;

    // Exercise dates
    int M = 50;

    // Monte Carlo paths
    int Ntrain = 50000;
    int Ntest = 100000;

    // ------------------------------------------------------
    // European benchmark
    // ------------------------------------------------------
    double european = europeanPutBlackScholes(S0, K, r, q, sigma, T);

    // ------------------------------------------------------
    // CRR benchmark with SAME exercise grid as LSM
    // ------------------------------------------------------
    double crrMatched = americanPutBinomial(S0, K, r, q, sigma, T, M);

    // Fine CRR approximation to continuous exercise
    double crrFine = americanPutBinomial(S0, K, r, q, sigma, T, 15000);

    // ------------------------------------------------------
    // Training paths
    // ------------------------------------------------------
    vector<vector<double>> trainingPaths = simulatePaths(S0, r, q, sigma, T, M, Ntrain, 42);

    LSMPolicy policy = trainLSM(trainingPaths, K, r, T);

    // Release training paths before creating test set
    trainingPaths.clear();
    trainingPaths.shrink_to_fit();

    // ------------------------------------------------------
    // Independent test paths
    // ------------------------------------------------------
    vector<vector<double>> testPaths = simulatePaths(S0, r, q, sigma, T, M, Ntest, 123456);

    LSMResult lsm = evaluateLSM(testPaths, policy, K, r, T);

    // ------------------------------------------------------
    // Results
    // ------------------------------------------------------
    double earlyExercisePremium = lsm.price - european;

    double errorVsMatchedCRR = lsm.price - crrMatched;

    cout << fixed << setprecision(6);

    cout << "Parameters\n";
    cout << "-----------------------------------\n";

    cout << "S0       = " << S0 << '\n';
    cout << "K        = " << K << '\n';
    cout << "r        = " << r << '\n';
    cout << "q        = " << q << '\n';
    cout << "sigma    = " << sigma << '\n';
    cout << "T        = " << T << '\n';
    cout << "M        = " << M << '\n';
    cout << "Ntrain   = " << Ntrain << '\n';
    cout << "Ntest    = " << Ntest << '\n';

    cout << "\nBenchmarks\n";
    cout << "-----------------------------------\n";

    cout << "European Black-Scholes = " << european << '\n';

    cout << "CRR (matched M)        = " << crrMatched << '\n';

    cout << "CRR (fine reference)   = " << crrFine << '\n';

    cout << "\nLSM\n";
    cout << "-----------------------------------\n";

    cout << "LSM price              = " << lsm.price << '\n';

    cout << "Standard error         = " << lsm.standardError << '\n';

    cout << "95% CI                 = [" << lsm.price - 1.96 * lsm.standardError << ", "
         << lsm.price + 1.96 * lsm.standardError << "]\n";

    cout << "Early exercise premium = " << earlyExercisePremium << '\n';

    cout << "LSM - matched CRR      = " << errorVsMatchedCRR << '\n';

    cout << "\nTraining continuation at t=0 = " << policy.trainingContinuation0 << '\n';

    cout << "Exercise immediately at t=0? = " << (policy.exerciseAtTime0 ? "Yes" : "No") << '\n';

    return 0;
}