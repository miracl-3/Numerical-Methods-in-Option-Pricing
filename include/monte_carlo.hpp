#ifndef EUROPEAN_OPTION_MONTE_CARLO_HPP
#define EUROPEAN_OPTION_MONTE_CARLO_HPP

#include <string>
#include <cstddef>

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

    // Modifier
    void toggle();
};

#endif