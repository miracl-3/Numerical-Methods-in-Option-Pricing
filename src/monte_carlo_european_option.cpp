// Include essential libraries
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// Include header files
#include "monte_carlo.hpp"

// Initial default parameters
void EuropeanOptionMonteCarlo::init()
{
    // Initial all parameters
    r = 0.08;      // 8% risk-free interest rate
    vol = 0.30;    // standard deviation of stock returns
    K = 65.0;      // fixed strike price
    T = 0.25;      // 3 months left until maturity or 1/4 of a year
    S = 60.0;      // current stock price
    q = r;         // dividend
    optType = "C"; // compute call option firce
}

// Define the copy methods that copies other object' parameters
void EuropeanOptionMonteCarlo::copy(const EuropeanOptionMonteCarlo &option)
{
    r = option.r;
    vol = option.vol;
    K = option.K;
    T = option.T;
    S = option.S;
    q = option.q;
    optType = option.optType;
}

// Initialize constructor
EuropeanOptionMonteCarlo::EuropeanOptionMonteCarlo()
{
    init();
}

// Copy constructor
EuropeanOptionMonteCarlo::EuropeanOptionMonteCarlo(
    const EuropeanOptionMonteCarlo &option)
{
    copy(option);
}

// Option-type constructor
EuropeanOptionMonteCarlo::EuropeanOptionMonteCarlo(
    const std::string &optionType)
{
    init();
    optType = optionType;
}

// Destructor
EuropeanOptionMonteCarlo::~EuropeanOptionMonteCarlo()
{
}

// Assignment constructor
EuropeanOptionMonteCarlo &EuropeanOptionMonteCarlo::operator=(
    const EuropeanOptionMonteCarlo &option)
{
    if (this == &option)
    {
        return *this;
    }

    copy(option);
    return *this;
}

// Random process of stock price path
double EuropeanOptionMonteCarlo::generateStockPricePath(double z) const
{
    double drift = (r - q - 0.5 * vol * vol) * T;
    double diffusion = vol * std::sqrt(T) * z;

    return S * std::exp(drift + diffusion);
}

// Call the function to compute option price
double EuropeanOptionMonteCarlo::Price(std::size_t numberOfSimulations) const
{
    if (numberOfSimulations == 0)
    {
        throw std::invalid_argument("Number of simulations must be positive.");
    }

    if (optType == "C")
    {
        return CallPrice(numberOfSimulations);
    }
    else
    {
        return PutPrice(numberOfSimulations);
    }
}

// Compute the call option price
double EuropeanOptionMonteCarlo::CallPrice(std::size_t numberOfSimulations) const
{
    std::mt19937 generator(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    double payoffSum = 0.0;

    for (std::size_t i = 0; i < numberOfSimulations; ++i)
    {
        double z = normal(generator);
        double ST = generateStockPricePath(z);

        double payoff = std::max(ST - K, 0.0);
        payoffSum += payoff;
    }

    double averagePayoff = payoffSum / static_cast<double>(numberOfSimulations);

    return std::exp(-r * T) * averagePayoff;
}

// Compute the put option price
double EuropeanOptionMonteCarlo::PutPrice(std::size_t numberOfSimulations) const
{
    std::mt19937 generator(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    double payoffSum = 0.0;

    for (std::size_t i = 0; i < numberOfSimulations; ++i)
    {
        double z = normal(generator);
        double ST = generateStockPricePath(z);

        double payoff = std::max(K - ST, 0.0);
        payoffSum += payoff;
    }

    double averagePayoff = payoffSum / static_cast<double>(numberOfSimulations);

    return std::exp(-r * T) * averagePayoff;
}

void EuropeanOptionMonteCarlo::toggle()
{
    if (optType == "C")
    {
        optType = "P";
    }
    else
    {
        optType = "C";
    }
}