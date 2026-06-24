// Black-Scholes models for European Options
#ifndef BLACK_SCHOLES_HPP
#define BLACK_SCHOLES_HPP

class EuropeanOption
{
private:
    void init();                             // Initialize all default values
    void copy(const EuropeanOption &option); // Copy all values from another option
    double CallPrice() const;
    double PutPrice() const;

public:
    // Public data
    double r;   // interest rate
    double vol; // sigma = volatility
    double K;   // strike price
    double T;   // maturity (years), i.e. how many times left to maturity, for example 0.5 ~ 6 months left
    double S;   // underlying stock price
    double q;   // dividend

public:
    // Constructors
    EuropeanOption();                              // Default call option
    EuropeanOption(const EuropeanOption &option2); // Copy of constructor

    // Destructors
    virtual ~EuropeanOption();

    // Assignment operator
    EuropeanOption &operator=(const EuropeanOption &option2);

    // Function that calculate option price and (possible) sensitivities
    double Price() const;
    // double Delta() const;

    // Modifier functions
    void toggle(); // Change options type (Call or Put
};

#endif
