// Necessary libraries
#include <iostream>
#include <math.h>
#include <algorithm>

using namespace std;

// Include header files
#include "black_scholes.hpp"
#include "monte_carlo.hpp"

int main()
{
    EuropeanOption callOptionBS;
    cout << "Call option price (Black-Scholes - no dividends): "
         << callOptionBS.Price()
         << endl;

    size_t number_of_simulations = 1000000000;

    EuropeanOptionMonteCarlo callOptionMC;
    cout << "Call option price (Monte Carlo - no dividends): " << callOptionMC.Price(number_of_simulations) << endl;

    return 0;
}