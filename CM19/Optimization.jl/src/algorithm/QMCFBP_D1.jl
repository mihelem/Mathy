# @return : x corresponding to min-norm ϵ-subgradient
function primal_from_dual(problem::QMCFBProblem, μ;
    ϵ=1e-12,
    ε=1e-8,
    max_iter=1000,
    max_iter_local=100)

    @unpack Q, q, l, u, E, b = problem
    m, n = size(E)
    Q╲ = view(Q, [CartesianIndex(i, i) for i in 1:size(Q, 1)])
    kerny = Q╲ .== 0.0
    Qx̃ = -E'μ-q
    x = max.(min.(Qx̃./Q╲, u), l)
    nanny = zeros(Bool, length(kerny))
    nanny[kerny] = abs.(Qx̃[kerny]) .≤ ϵ
    # argmin || E[:, 𝔫]*x[𝔫] + E[:, .~𝔫]*x[.~𝔫] - b ||
    # ≡ argmin || E₁*x₁ + E₀*x₀ - b ||
    # ≡ argmin ½x₁'E₁'E₁x₁ + (E₀*x₀-b)'E₁*x₁
    problem₁ = MinQuadratic.MQBProblem(
        E[:, nanny]'E[:, nanny],
        E[:, nanny]'*(E[:, .~nanny]*x[.~nanny]-b),
        l[nanny],
        u[nanny])
    instance = OptimizationInstance{MinQuadratic.MQBProblem}()
    algorithm = MinQuadratic.MQBPAlgorithmPG1(
        localization=MinQuadratic.QuadraticBoxPCGDescent(),
        verbosity=-1,
        max_iter=max_iter,
        max_iter_local=max_iter_local,
        ε=ε/√n,
        ϵ₀=ϵ)
    Optimization.set!(instance,
        problem=problem₁,
        algorithm=algorithm,
        options=MinQuadratic.MQBPSolverOptions(),
        solver=OptimizationSolver{MinQuadratic.MQBProblem}())
    Optimization.run!(instance)
    x[nanny] = instance.result.result["x"]
    @show count(.~(l .≤ x .≤ u))
    x
end
"""
Example
```julia
using Parameters
using Optimization

subgradient = Subgradient.Adagrad(α=0.1)
algorithm = QMCFBPAlgorithmD1SG(
                 localization=subgradient,
                 verbosity=1,
                 max_iter=10000,
                 ε=1e-6,
                 ϵ=1e-12);

test = get_test(algorithm, m=100, n=200, singular=60, active=30);
test.solver.options.memoranda = Set(["norm∂L_best", "L_best","i_best","L", "L"]);
for i in 1:20
           run!(test)
           set!(algorithm, test.result)
           subgradient.α /= 4.0
       end

L_lb = test.result.result["L_best"]
μ = test.result.result["μ_best"]
problem = test.problem
@unpack Q, q, l, u, E, b = 𝔓
x = Optimization.MinCostFlow.primal_from_dual(problem, μ)
heu = BFSHeuristic(𝔓, x; ϵ=1e-8)
init!(heu)
x′, ∂L′ = run!(heu)
L_ub = 0.5*x′⋅Q*x′+q'x′
```
"""
