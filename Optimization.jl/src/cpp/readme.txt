WHAT
^^^^

Multithreading implementation of Restarted Nesterov Momentum subgradient method
specialized to QMCFBProblems.
The instance is imported from files in DIMACS format generated
with Netgen+Qfcgen+Pargen.



HOWTO
^^^^^

BUILD:
	g++ -std=c++17 nano.cpp -o bin/test -ltbb -Wall -Wextra -pedantic -O3

USE:
	./test problem_name n_stages n_iters_per_stage alpha beta [alpha_div=2.0]

Calculate the lower bound of the optimal value via the Restarted Nesterov Momentum
