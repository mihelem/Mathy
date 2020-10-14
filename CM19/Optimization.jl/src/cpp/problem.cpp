#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>
#include <optional>
#include <limits>
#include <iomanip>
#include <cmath>

bool starts_with(std::string const &s, char c) {
  return s.size() > 0 && s[0] == c;
}

template <class Value>
Value norm(uint16_t l, std::vector<Value> const & v) {
  Value result{0};
  for ( auto u: v ) {
    Value t{1};
    for (auto i{0}; i<l; ++i)
      t *= u;
    result += abs(t);
  }
  return result;
}

template <class T>
void print(std::string name, std::vector<T> const &v) {
  std::cout << name << " : ";
  for (auto u:v) std::cout << u << ' ';
  std::cout << std::endl;
}
template <>
void print<double>(std::string name, std::vector<double> const &v) {
  std::cout << name << " : ";
  for (auto u:v) std::cout << std::fixed << std::setprecision(2) << u << ' ';
  std::cout << std::endl;
}

template <
  class Value,
  class Index,
  template <class ...> class Container,
  template <class, class, template <class...> class> class Problem>
struct QMCFBPtype_traits {
  template <class ...args>
  using container = Container<args...>;
  using value = Value;
  using index = Index;
  using problem = Problem<Value, Index, Container>;
};

template <
  class Value,
  class Index,
  template <class ...> class Container,
  template <class, class, template <class ...> class> class Problem>
struct ProblemVecsBase {
  using type = QMCFBPtype_traits<Value, Index, Container, Problem>;

  typename type::index m, n;

  typename type::template container<typename type::index> all_m;

  typename type::template container<typename type::index> all_n;
  typename type::index sing_begin;
  typename type::template container<typename type::index> nsing_n;
  typename type::template container<typename type::index> sing_n;

  typename type::template container<typename type::value> Q;
  typename type::template container<typename type::value> q_fixed;
  typename type::template container<typename type::value> invQ;
  typename type::template container<typename type::value> q;
  typename type::template container<typename type::index> e; // out - in
  typename type::template container<typename type::index> v; // out - in
  typename type::template container<typename type::index> vn;

  typename type::template container<typename type::value> l;
  typename type::template container<typename type::value> u;
  typename type::template container<typename type::value> b;
};

template <class Value, class Index, template <class ...> class Container>
struct ProblemVecs : ProblemVecsBase<Value, Index, Container, ProblemVecs> {};

template <class Value, class Index>
struct ProblemVecs<Value, Index, std::vector>
: ProblemVecsBase<Value, Index, std::vector, ProblemVecs> {
  using type = typename ProblemVecsBase<Value, Index, std::vector, ProblemVecs>::type;

  ProblemVecs(
    std::string const &path,
    std::string const &filename,
    typename type::index singular
  ) {
    std::ifstream streamQFC(path+filename+".qfc");

    streamQFC >> this->n;
    this->sing_begin = this->n-singular;
    this->all_n = std::vector<typename type::index>(this->n);
    std::iota(this->all_n.begin(), this->all_n.end(), 0);

    this->q_fixed = std::vector<typename type::value>(this->n);
    for(auto &x: this->q_fixed) {
      streamQFC >> x;
    }

    this->Q = std::vector<typename type::value>(this->n, 0);
    this->invQ = std::vector<typename type::value>(this->sing_begin);
    for (int i=0; i<this->sing_begin; ++i) {
      streamQFC >> this->Q[i];
      this->invQ[i] = 1 / this->Q[i];
    }

    streamQFC.close();
    std::string line;

    std::string tmp;
    std::ifstream streamDMX(path+filename+".dmx");
    while (std::getline(streamDMX, line)) {
      if (starts_with(line, 'p')) {
        std::stringstream ss(line);
        typename type::index n2;
        ss >> tmp >> tmp >> this->m >> n2;
        if (n2 != this->n) {
          std::cerr << "Incompatible dimensions btw QFC and DMX file";
        }
        this->n = n2;
        break;
      }
    }

    this->all_m = std::vector<typename type::index>(this->m);
    std::iota(this->all_m.begin(), this->all_m.end(), 0);
    typename type::index i;
    typename type::value bi;
    typename type::index edge_cnt{0};
    this->vn = std::vector<typename type::index>(2*this->m+1, 0);
    this->e = std::vector<typename type::index>(2*this->n);
    this->b = std::vector<typename type::value>(this->m);
    this->l = std::vector<typename type::value>(this->n);
    this->u = std::vector<typename type::value>(this->n);
    this->q = std::vector<typename type::value>(this->n);

    typename type::index j{0};
    std::vector<typename type::index> arcs(this->n, 0);
    do {
      if (starts_with(line, 'n')) {
        std::stringstream ss(line);
        ss >> tmp >> i >> bi;
        this->b[--i] = -bi;
      } else if (starts_with(line, 'a')) {
        std::stringstream ss(line);
        ss >> tmp >> this->e[2*edge_cnt] >> this->e[2*edge_cnt+1]
          >> this->l[edge_cnt] >> this->u[edge_cnt] >> this->q[edge_cnt];
        --this->e[2*edge_cnt], --this->e[2*edge_cnt+1];

        if (this->e[2*edge_cnt] >= 0 && this->e[2*edge_cnt] < this->m &&
          this->e[2*edge_cnt+1] >= 0 && this->e[2*edge_cnt+1] < this->m) {
          arcs[j] = 1;
          ++this->vn[2*this->e[2*edge_cnt]+1];
          ++this->vn[2*this->e[2*edge_cnt+1]+2];
          ++edge_cnt;
        }
        ++j;
      }
    } while (std::getline(streamDMX, line));

    streamDMX.close();
    auto nonsingular{std::reduce(arcs.begin(), arcs.begin()+this->sing_begin, 0)};
    for (typename type::index i{0}, j{0}; i<this->sing_begin; ++i) {
      if (arcs[i]) {
        this->Q[j] = this->Q[i];
        this->invQ[j] = this->invQ[i];
        ++j;
      }
    }
    for (typename type::index i{0}, j{0}; i<this->n; ++i) {
      if (arcs[i]) {
        this->q[j] = this->q[i];
        this->q_fixed[j] = this->q_fixed[i];
        ++j;
      }
    }
    this->sing_begin = nonsingular;
    singular = edge_cnt-nonsingular;
    this->sing_n = std::vector<typename type::index>(singular);
    this->nsing_n = std::vector<typename type::index>(nonsingular);
    std::iota(this->sing_n.begin(), this->sing_n.end(), this->sing_begin);
    std::iota(this->nsing_n.begin(), this->nsing_n.end(), 0);

    std::vector<typename type::index> is(2*this->m);
    for (typename type::index i{0}; i<2*this->m; ++i) {
      this->vn[i+1] += this->vn[i];
      is[i] = this->vn[i];
    }
    this->n = edge_cnt;
    this->v = std::vector<typename type::index>(2*this->n);
    for (typename type::index i{0}; i<this->n; ++i) {
      this->v[is[2*this->e[2*i]]++] = this->v[is[2*this->e[2*i+1]+1]++] = i;
    }
    this->e.resize(2*this->n);
    this->l.resize(this->n);
    this->u.resize(this->n);
    this->Q.resize(this->n);
    this->invQ.resize(this->n);
    this->q_fixed.resize(this->n);
    this->q.resize(this->n);
    this->all_n.resize(this->n);
  }

  typename type::template container<typename type::value> &get_x(
    typename type::template container<typename type::value> const &mu,
    typename type::template container<typename type::value> &x
  ) const {
    std::for_each(
      std::execution::par_unseq,
      this->sing_n.begin(),
      this->sing_n.end(),
      [&](typename type::index i) {
        x[i] = -this->q[i]+mu[this->e[2*i]]-mu[this->e[2*i+1]] >= 0 ? this->u[i] : this->l[i];
      }
    );
    std::for_each(
      std::execution::par_unseq,
      this->nsing_n.begin(),
      this->nsing_n.end(),
      [&](typename type::index i) {
        x[i] =
          std::max(
            std::min(
              this->invQ[i] * (-this->q[i] + mu[this->e[2*i]] - mu[this->e[2*i+1]]),
              this->u[i]),
            this->l[i]);
      }
    );
    return x;
  }

  typename type::template container<typename type::value> &get_dL(
    typename type::template container<typename type::value> const &x,
    typename type::template container<typename type::value> &dL
  ) const {
    for_each(
      std::execution::par_unseq,
      this->all_m.begin(),
      this->all_m.end(),
      [&](typename type::index i) {
        typename type::value tmp = 0;
        for (auto j{this->vn[2*i]}; j<this->vn[2*i+1]; ++j) {
          tmp -= x[this->v[j]];
        }
        for (auto j{this->vn[2*i+1]}; j<this->vn[2*i+2]; ++j) {
          tmp += x[this->v[j]];
        }
        dL[i] = -this->b[i] + tmp;
      }
    );
    return dL;
  }

  typename type::value get_L(
    typename type::template container<typename type::value> const &x,
    typename type::template container<typename type::value> const &mu,
    typename type::template container<typename type::value> const &dL,
    typename type::template container<typename type::value> &Lvec
  ) const {
    for_each(
      std::execution::par_unseq,
      this->nsing_n.begin(),
      this->nsing_n.end(),
      [&](typename type::index i) {
        Lvec[i] = x[i]*(0.5*x[i]*this->Q[i] + this->q[i]);
      }
    );
    for_each(
      std::execution::par_unseq,
      this->sing_n.begin(),
      this->sing_n.end(),
      [&](typename type::index i) {
        Lvec[i] = x[i]*this->q[i];
      }
    );
    for_each(
      std::execution::par_unseq,
      this->all_m.begin(),
      this->all_m.end(),
      [&](typename type::index i) {
        Lvec[this->n + i] = mu[i]*dL[i];
      }
    );

    return std::reduce(
        std::execution::par_unseq,
        begin(Lvec),
        end(Lvec)
      );
  }
};


template <
  class Value,
  class Index,
  template <class ...> class Container,
  template <class, class, template <class...> class, template <class, class, template <class ...> class> class> class Subgradient>
struct SubgradientIterationData {
  SubgradientIterationData() = delete;
};

template <
  class Value,
  class Index,
  template <class ...> class Container,
  template <class, class, template <class...> class, template <class, class, template <class ...> class> class> class Subgradient>
struct SubgradientIterationParams {
  SubgradientIterationParams() = delete;
};

template <
  class Value,
  class Index,
  template <class ...> class Container,
  template <class, class, template <class ...> class> class Problem,
  template <class, class, template <class...> class, template <class, class, template <class ...> class> class> class Subgradient>
struct SubgradientIteration {
  using type = QMCFBPtype_traits<Value, Index, Container, Problem>;
  Index n_iters;

  SubgradientIterationData<Value, Index, Container, Subgradient> data;
  SubgradientIterationParams<Value, Index, Container, Subgradient> params;
};

template <
  class Value,
  class Index,
  template <class...> class Container,
  template <class, class, template <class ...> class> class Problem>
struct NesterovMomentumIteration;

template <class Value, class Index, template <class...> class Container>
struct SubgradientIterationParams<Value, Index, Container, NesterovMomentumIteration> {
  Value alpha;
  Value beta;
};

template <class Value, class Index, template <class ...> class Container>
struct SubgradientIterationData<Value, Index, Container, NesterovMomentumIteration> {
  Container<Value> mu;
  Container<Value> x;
  Container<Value> v;
  Container<Value> g;
  Container<Value> mu2;
  Container<Value> x2;

  Container<Value> mu_best;
  Container<Value> x_best;
  Container<Value> g_best;
  Container<Value> Lvec;
  Value L_best;
  Index i_best;
};

/*
    α, β, v = M.α, M.β, M.v
    g = ∂f(μ + β*v)
    v[:] = β*v - α*g
    (μ+v, α, g, β, v)
*/

template <
  class Value,
  class Index,
  template <class...> class Container,
  template <class, class, template <class ...> class> class Problem>
struct NesterovMomentumIteration
: SubgradientIteration<Value, Index, Container, Problem, NesterovMomentumIteration>{};


template <class Value, class Index>
struct NesterovMomentumIteration<Value, Index, std::vector, ProblemVecs>
: SubgradientIteration<Value, Index, std::vector, ProblemVecs, NesterovMomentumIteration>
{
  using type = typename SubgradientIteration<Value, Index, std::vector, ProblemVecs, NesterovMomentumIteration>::type;

  NesterovMomentumIteration(
    std::optional<typename type::index> n_iters,
    std::optional<typename type::value> alpha,
    std::optional<typename type::value> beta
  ) {
    if (n_iters) this->n_iters = *n_iters;
    if (alpha) this->params.alpha = *alpha;
    if (beta) this->params.beta = *beta;
  }

  void init(
    typename type::problem const &problem,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> mu,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> x,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> v,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> g,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> mu2,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> x2,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> mu_best,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> x_best,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> g_best,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> Lvec,
    std::optional<typename type::value> L_best,
    std::optional<typename type::index> i_best
  ) {
    this->data.mu = mu ? std::move(*mu) : typename type::template container<typename type::value>(problem.m, 0);
    this->data.x = x ? std::move(*x) : typename type::template container<typename type::value>(problem.n, 0);
    this->data.v = v ? std::move(*v) : typename type::template container<typename type::value>(problem.m, 0);
    this->data.g = g ? std::move(*g) : typename type::template container<typename type::value>(problem.m, 0);
    this->data.mu2 = mu2 ? std::move(*mu2) : typename type::template container<typename type::value>(problem.m, 0);
    this->data.x2 = x2 ? std::move(*x2) : typename type::template container<typename type::value>(problem.n, 0);
    this->data.mu_best = mu_best ? std::move(*mu_best) : typename type::template container<typename type::value>(problem.m, 0);
    this->data.x_best = x_best ? std::move(*x_best) : typename type::template container<typename type::value>(problem.n, 0);
    this->data.g_best = g_best ? std::move(*g_best) : typename type::template container<typename type::value>(problem.m, 0);
    this->data.Lvec = Lvec ? std::move(*Lvec) : typename type::template container<typename type::value>(problem.m+problem.n, 0);
    this->data.L_best = L_best ? *L_best : std::numeric_limits<typename type::value>::lowest()/2;
    this->data.i_best = i_best ? *i_best : 0;
  }

  void run(
    typename type::problem const &problem,
    typename type::template container<typename type::value> &mu
  ) {

    for (auto i{0}; i<this->n_iters; ++i) {
      for_each(
        std::execution::par_unseq,
        problem.all_m.begin(),
        problem.all_m.end(),
        [&](typename type::index i) {
          this->data.mu2[i] = mu[i] + this->params.beta*this->data.v[i];
        }
      );
      problem.get_x(this->data.mu2, this->data.x2);
      problem.get_dL(this->data.x2, this->data.g);
      for_each(
        std::execution::par_unseq,
        problem.all_m.begin(),
        problem.all_m.end(),
        [&](typename type::index i) {
          this->data.v[i] =
            this->params.beta*this->data.v[i] + this->params.alpha*this->data.g[i];
          mu[i] += this->data.v[i];
        }
      );

      problem.get_x(this->data.mu, this->data.x);
      problem.get_dL(this->data.x, this->data.g);
      auto L{problem.get_L(this->data.x, this->data.mu, this->data.g, this->data.Lvec)};
      if (L > this->data.L_best) {
        this->data.L_best = L;
        std::copy(this->data.mu.begin(), this->data.mu.end(), this->data.mu_best.begin());
        // it's not worth the cost to copy also all the best x, g, etc...
      }
    }
  }
};

template <class Value>
struct RNMUpdate {
  RNMUpdate(Value alpha_div) : alpha_div{alpha_div} {};

  Value alpha_div;

  template <class Subgradient>
  Subgradient &update(Subgradient &sg) {
    sg.params.alpha /= alpha_div;
    return sg;
  }
};

template <
  class Value,
  class Index,
  template <class ...> class Container,
  template <class, class, template <class ...> class> class Problem,
  template <class, class, template <class...> class, template <class, class, template <class ...> class> class> class Subgradient,
  template <class> class Update>
struct SolverBase {
  using type = QMCFBPtype_traits<Value, Index, Container, Problem>;

  Index n_iters;
  Index n_stages;

  Subgradient<Value, Index, Container, Problem> sg;
  Update<Value> sg_update;
  SolverBase(
    typename type::index n_iters,
    typename type::index n_stages,
    typename type::value alpha,
    typename type::value beta,
    typename type::value alpha_div)
  : n_iters{n_iters}, n_stages{n_stages}, sg(n_iters, alpha, beta), sg_update(alpha_div) {}


  auto &get_subgradient() { return this->sg; }
  auto &get_update() { return this->sg_update; }
};

template <
  class Value,
  class Index,
  template <class ...> class Container,
  template <class, class, template <class ...> class> class Problem>
struct RNMSolver
: SolverBase<
    Value,
    Index,
    Container,
    Problem,
    NesterovMomentumIteration,
    RNMUpdate> {
  using type = typename SolverBase<Value, Index, Container, Problem, NesterovMomentumIteration, RNMUpdate>::type;

  RNMSolver(
    typename type::index n_iters,
    typename type::index n_stages,
    typename type::value alpha,
    typename type::value beta,
    typename type::value alpha_div)
  : SolverBase<
      Value,
      Index,
      Container,
      Problem,
      NesterovMomentumIteration,
      RNMUpdate>(n_iters, n_stages, alpha, beta, alpha_div) {}

  template <class ...Args>
  void solve(
    typename type::problem const &problem,
    std::optional<std::reference_wrapper<typename type::template container<typename type::value>>> mu,
    Args ...args
  ) {
    this->sg.init(problem, mu, args...);
    for (typename type::index i{0}; i<this->n_stages; ++i) {
      this->sg.run(problem, this->sg.data.mu);
      std::copy(
        this->sg.data.mu_best.begin(),
        this->sg.data.mu_best.end(),
        this->sg.data.mu.begin());
      this->sg_update.update(this->sg);
    }
  }
};

template <class Value, class Index>
RNMSolver<Value, Index, std::vector, ProblemVecs> solve(
  ProblemVecs<Value, Index, std::vector> const &problem,
  std::optional<std::reference_wrapper<std::vector<Value>>> optmu,
  Index n_stages,
  Index n_iters,
  Value alpha,
  Value beta,
  Value alpha_div
) {
  std::vector<Value> mu = optmu ? std::move(*optmu) : std::vector<Value>(problem.m, 0);
  RNMSolver<Value, Index, std::vector, ProblemVecs> solver(n_iters, n_stages, alpha, beta, alpha_div);
  solver.solve(
    problem,
    mu,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt
  );

  return solver;
}
