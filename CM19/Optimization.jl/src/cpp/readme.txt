HOWTO

BUILD:
	g++ -std=c++17 nano.cpp -o bin/test -ltbb -Wall -Wextra -pedantic -O3
	
USE:
	./test problem_name n_stages n_iters_per_stage alpha beta
	
Calculate the lower bound of the optimal value via the Restarted Nesterov Momentum

