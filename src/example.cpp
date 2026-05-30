#include <iostream>
#include <vector>
#include <string>

// ── Phase 0 smoke test ───────────────────────────────────────────────────────
// This file has one job: confirm that:
//   1. g++ can find this file and compile it with -std=c++17
//   2. The build task (Ctrl+Shift+B) writes the .exe to build/
//   3. The debugger can attach and hit a breakpoint on main()
//
// Delete or replace this file once Phase 1 begins.

int main() {
    // C++17 feature: structured bindings — if this compiles, your standard is set correctly
    std::vector<std::pair<std::string, double>> params = {
        {"kappa",  2.0},
        {"theta",  0.04},
        {"sigma",  0.3},
        {"rho",   -0.7},
        {"v0",     0.04}
    };

    std::cout << "=== Heston MC — environment check ===\n";
    for (auto& [name, value] : params) {   // structured binding: C++17
        std::cout << "  " << name << " = " << value << "\n";
    }
    std::cout << "Compiler: OK | C++17: OK | Build task: OK\n";

    return 0;
}