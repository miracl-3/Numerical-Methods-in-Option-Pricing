#ifndef EUROPEAN_OPTION_MONTE_CARLO_HPP
#define EUROPEAN_OPTION_MONTE_CARLO_HPP

#include <cstddef>
#include <string>
#include <vector>

struct MonteCarloPricingResult
{
    double price;
    double standardError;
    double confidenceLower;
    double confidenceUpper;
};

struct TerminalDistributionSummary
{
    std::size_t simulations;
    std::size_t histogramBins;
    std::size_t terminalSampleSize;
    double histogramMin;
    double histogramMax;
    std::size_t underflowCount;
    std::size_t overflowCount;
    double mean;
    double variance;
    double standardDeviation;
    double minValue;
    double maxValue;
    double skewness;
    double excessKurtosis;
    double logMean;
    double logVariance;
    double logStandardDeviation;
    double logSkewness;
    double logExcessKurtosis;
    double theoreticalMean;
    double theoreticalVariance;
    double theoreticalStandardDeviation;
    double theoreticalMedian;
    double theoreticalLogMean;
    double theoreticalLogStandardDeviation;
    double elapsedSeconds;
};

class EuropeanOptionMonteCarlo
{
private:
    void init();
    void copy(const EuropeanOptionMonteCarlo &option);

    double CallPrice(std::size_t numberOfSimulations) const;
    double PutPrice(std::size_t numberOfSimulations) const;

    double generateStockPricePath(double z) const;

public:
    // Public data, following the textbook's beginner-friendly style
    double r;   // risk-free interest rate
    double vol; // volatility
    double K;   // strike price
    double T;   // time to maturity in years
    double S;   // current underlying stock price
    double q;   // continuous dividend yield

    std::string optType; // "C" or "P"

public:
    // Constructors
    EuropeanOptionMonteCarlo();
    EuropeanOptionMonteCarlo(const EuropeanOptionMonteCarlo &option);
    EuropeanOptionMonteCarlo(const std::string &optionType);

    // Destructor
    virtual ~EuropeanOptionMonteCarlo();

    // Assignment operator
    EuropeanOptionMonteCarlo &operator=(const EuropeanOptionMonteCarlo &option);

    // Monte Carlo pricing function
    double Price(std::size_t numberOfSimulations) const;
    MonteCarloPricingResult PriceWithStatistics(
        std::size_t numberOfSimulations,
        unsigned int seed = 42) const;

    std::vector<std::vector<double>> GeneratePricePaths(
        std::size_t numberOfPaths,
        std::size_t numberOfTimeSteps,
        unsigned int seed = 42) const;

    void ExportPricePathsCsv(
        const std::string &filePath,
        std::size_t numberOfPaths,
        std::size_t numberOfTimeSteps,
        unsigned int seed = 42) const;

    TerminalDistributionSummary ExportTerminalDistributionCsv(
        const std::string &histogramFilePath,
        const std::string &summaryFilePath,
        const std::string &terminalSampleFilePath,
        std::size_t numberOfSimulations,
        std::size_t histogramBins,
        std::size_t terminalSampleLimit = 1000000,
        unsigned int seed = 42) const;

    // Modifier
    void toggle();
};

#endif
