#include <iostream>

using namespace std;

// First exercise: print out input, its double and triple value
void firstExercise()
{
    cout << "Enter an integer: ";

    int input{};

    cin >> input;

    // Return double of the input
    cout << "Double " << input << " is " << input * 2 << '\n';

    // Return triple of the input
    cout << "Triple " << input << " is " << input * 3 << '\n';
}

// Second exercise: print out the subtraction and addition of two inputs
void secondExercise()
{
    // Output from the console to type in input
    cout << "Enter an integer: ";

    // Value-initialization
    int firstInput{};

    // Input from keyboard the value
    cin >> firstInput;

    cout << "Enter another integer: ";

    // Value-initialization
    int secondInput{};

    // Input from keyboard the value
    cin >> secondInput;

    // Output is subtraction
    cout << firstInput << " + " << secondInput << " is " << firstInput + secondInput << '\n';
    // Output is subtraction
    cout << firstInput << " - " << secondInput << " is " << firstInput - secondInput << '\n';
}

int main()
{
    // firstExercise();

    secondExercise();

    return 0;
}