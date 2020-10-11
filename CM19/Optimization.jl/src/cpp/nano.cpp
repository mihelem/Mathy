#include "problem.cpp"
#include <chrono>

int main(int argc, char *argv[]) {

  if (argc > 5) {
    double alpha_div = 2.0;
    int n_stages {std::stoi(argv[2])};
    int n_iters {std::stoi(argv[3])};
    double alpha {std::stod(argv[4])};
    double beta {std::stod(argv[5])};

    std::string filename(argv[1]);
    ProblemVecs<double, int, std::vector> problem("./", filename, 0);
    std::cout << problem.m << ' ' << problem.n << std::endl;


    std::vector<double> mu(problem.m, 0.0);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto solver = solve(
      problem,
      std::make_optional(std::reference_wrapper(mu)), // μ₀ ← %
      n_stages,                                             // stages ← %
      n_iters,                                           // iters per stage ← %
      alpha,                                            // α ← %
      beta,                                          // β ← %
      alpha_div);                                           // α → α/%
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

    std::cout << "\n" << std::setprecision(16) <<
      solver.sg.data.L_best << " in " << duration << std::endl;

    std::ofstream logge("bench.csv", std::fstream::out | std::fstream::app);
    logge << filename << " "
          << n_stages << " "
          << n_iters << " "
          << alpha << " "
          << beta << " "
          << alpha_div << " "
          << std::setprecision(16) << solver.sg.data.L_best << " "
          << duration << std::endl;
    logge.close();
/*
    mu = solver.sg.data.mu;
    std::cout << "solver.sg.data.mu : "; for (auto o:mu) { std::cout << o << ' '; } std::cout << std::endl;

    std::vector<double> x(problem.n);
    x = problem.get_x(mu, x);
    std::cout << "x : "; for (auto y:x) std::cout << y << ' '; std::cout << std::endl;

    std::vector<double> dL(problem.m);
    problem.get_dL(x, dL);
    std::cout << "dL : "; for (auto l:dL) std::cout << l << ' '; std::cout << std::endl;
    std::cout << "|∂L|=" << norm(2, dL) << std::endl;
    std::vector<double> Lvec(problem.n + problem.m);
    auto L{problem.get_L(x, mu, dL, Lvec)};
    std::cout << "L=" << L << std::endl; */
  }
}
