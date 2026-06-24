// Include essential libraries
#include <math.h>
#include <string>

// Include header files
#include "black_scholes.hpp"

// Declare the normal distribution function
double normalCDF(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// Class body or code file
double EuropeanOption::CallPrice() const
{
    double tmp = vol * sqrt(T); // denominator = vol * sqrt(T)

    double d1 = ((log(S / K) + (r - q + (vol * vol) * 0.5)) * T) / tmp;
    double d2 = d1 - vol * sqrt(T);

    return S * exp(-q * T) * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2);
}

double EuropeanOption::PutPrice() const
{
    double tmp = vol * sqrt(T); // denominator = vol * sqrt(T)

    double d1 = ((log(S / K) + (r - q + (vol * vol) * 0.5)) * T) / tmp;
    double d2 = d1 - vol * sqrt(T);

    return K * exp(-r * T) * normalCDF(-d2) - S * exp(-q * T) * normalCDF(-d1);
}

void EuropeanOption::init()
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

void EuropeanOption::copy(const EuropeanOption &o2)
{
    r = o2.r;
    vol = o2.vol;
    K = o2.K;
    T = o2.T;
    S = o2.S;
    q = o2.q;
    optType = o2.optType;
}

EuropeanOption::EuropeanOption()
{
    // Default call option
    init();
}

EuropeanOption::EuropeanOption(const EuropeanOption &o2)
{ // Copy constructor
    copy(o2);
}

EuropeanOption::EuropeanOption(const std::string &optionType)
{ // Create option type
    init();
    optType = optionType;
    if (optType == "c")
    {
        optType = "C";
    }
}

EuropeanOption::~EuropeanOption()
{ // Destructor
}

EuropeanOption &EuropeanOption::operator=(const EuropeanOption &opt2)
{ // Assignment operator (deep copy)
    if (this == &opt2)
        return *this;
    copy(opt2);
    return *this;
}

// Functions that calculate option price and sensitivities
double EuropeanOption::Price() const
{
    if (optType == "C")
    {
        return CallPrice();
    }
    else
    {
        return PutPrice();
    }
}

// Modifier functions
void EuropeanOption::toggle()
{ // Change option type (C/P, P/C)
    if (optType == "C")
        optType = "P";
    else
        optType = "C";
}
