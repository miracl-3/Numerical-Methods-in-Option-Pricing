#include <iostream>

using namespace std;

int getValueFromUser() {
    int val{};
    cout << "Enter a value: ";
    cin >> val;
    return val;
}

int doubleNumber(int input) { return input * 2; }

// Example of forward declaration
int doMath(int first, int second, int third, int fourth);

int main() {

    // int num{getValueFromUser()};

    // cout << "You entered " << num << "\n";
    cout << "The value of such operation is: " << doMath(2, 3, 4, 5);
    return 0;
}

int doMath(int first, int second, int third, int fourth) { return first + second * third / fourth; }