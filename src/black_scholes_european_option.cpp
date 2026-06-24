// Include essential libraries
#include <math.h>

// Include header files
#include "black_scholes.hpp"

// Class body or code file
double EuropeanOption::CallPrice() const
{

    double tmp = vol * sqrt(T);
    double d1 = ((log(S / K) + (r - q + (vol * vol) * 0.5)) * T) / tmp;
};
