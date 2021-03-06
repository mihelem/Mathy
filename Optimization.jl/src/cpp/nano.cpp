#include "problem.cpp"
#include <chrono>

size_t find_last(std::string s, char c) {
  size_t pos = s.size();
  for (size_t i{0}; i<s.size(); ++i) {
    if (s[i] == c) {
      pos = i;
    }
  }
  return pos;
}

std::string find_last_token(std::string s, char c) {
  return s.substr(find_last(s, c)+1, s.size());
}

int main(int argc, char *argv[]) {

  if (argc != 6 && argc != 7) {
    std::cout << argv[0]
              << " problem_name n_stages n_iters_per_stage alpha beta [alpha_div=2.0]\n"
              << "\t\tto run RNM on the problem described in problem_name.{qfc, dmx}"
              << std::endl;
    return 0;
  } else {
    int n_stages {std::stoi(argv[2])};
    int n_iters {std::stoi(argv[3])};
    double alpha {std::stod(argv[4])};
    double beta {std::stod(argv[5])};
    double alpha_div = argc==6 ? 2.0 : std::stod(argv[6]);

    std::string filename(argv[1]);
    auto get_singular = [](std::string const &filename) -> int {
        try {
          return std::stoi(find_last_token(filename, '-'));
        } catch (std::invalid_argument &ex) {
          return 0;
        }
      };
    auto singular{get_singular(filename)};
    ProblemVecs<double, int, std::vector> problem("./", filename, singular);
    std::cout << problem.m << ' ' << problem.n << std::endl;


    std::vector<double> mu(problem.m, 0.0);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto solver = solve(
      problem,
      std::make_optional(std::reference_wrapper(mu)),     // μ₀ ← %
      n_stages,                                           // stages ← %
      n_iters,                                            // iters per stage ← %
      alpha,                                              // α ← %
      beta,                                               // β ← %
      alpha_div);                                         // α → α/%
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
  }
}
