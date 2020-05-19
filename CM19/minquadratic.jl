# Prova risolutiva per
# minₓ q'x     with
# Ex = b   and   l .≤ x .≤ u 
# L = q'x + μ'(E*x-b)
using Parameters
using LinearAlgebra
using SparseArrays

include("utils.jl")
include("optimization.jl")
include("descent.jl")

# ---------------- (Convex) Quadratic Boxed Problem ------------------ #
# minₓ ½ x'Qx + q'x         where
#     l .≤ x .≤ u
struct MQBProblem <: OptimizationProblem
    Q
    q
    l
    u
end

# ---------------------------- Solver Options --------------------------- #
mutable struct MQBPSolverOptions <: OptimizationSolverOptions{MQBProblem}
    memoranda::Set{String}  # set of variables that we want to track

    MQBPSolverOptions() = new(Set{String}([]))
    MQBPSolverOptions(memoranda::Set{String}) = new(memoranda)
end

# ----------------------------- Solver runner --------------------------- #
function run!(solver::OptimizationSolver{MQBProblem}, problem::MQBProblem)
    run!(solver.algorithm, problem, memoranda=solver.options.memoranda)
end

mutable struct MQBPAlgorithmPG1 <: OptimizationAlgorithm{MQBProblem}
    descent::DescentMethod
    verba
    max_iter
    ε
    ϵ₀
    x₀

    memorabilia
    MQBPAlgorithmPG1(;
        descent=nothing,
        verbosity=nothing,
        my_verba=nothing,
        max_iter=nothing,
        ε=nothing,
        ϵ₀=nothing,
        x₀=nothing) = begin

        algorithm = new()
        algorithm.memorabilia = Set(["normΠ∇f", "Π∇f", "x", "f"])
        set!(algorithm, descent=descent, verbosity=verbosity, my_verba=my_verba, max_iter=max_iter, ε=ε, ϵ₀=ϵ₀, x₀=x₀)
    end
end
function set!(algorithm::MQBPAlgorithmPG1;
    descent=nothing,
    verbosity=nothing,
    my_verba=nothing,
    max_iter=nothing,
    ε=nothing,
    ϵ₀=nothing,
    x₀=nothing)

    @some algorithm.descent = descent
    if verbosity !== nothing
        algorithm.verba = ((level, message) -> verba(verbosity, level, message))
    end
    @some algorithm.verba = my_verba
    @some algorithm.max_iter = max_iter
    @some algorithm.ε = ε
    @some algorithm.ϵ₀ = ϵ₀
    algorithm.x₀ = x₀

    algorithm
end
function run!(algorithm::MQBPAlgorithmPG1, 𝔓::MQBProblem; memoranda=Set([]))
    @unpack Q, q, l, u = 𝔓
    @unpack descent, max_iter, verba, ε, ϵ₀, x₀ = algorithm
    @init_memoria memoranda

    x = x₀ === nothing ? 0.5*(l+u) : x₀
    a::AbstractFloat ⪝ b::AbstractFloat = a ≤ b + ϵ₀

    get_Πx = x -> min.(max.(x, l), u)
    get_f = x -> 0.5*x'Q*x + q'x
    get_Πf = get_f ∘ get_Πx
    get_∇f = x -> Q*x+q
    get_Π∇f = x -> begin
        Π∇f = get_∇f(get_Πx(x))
        𝔲, dec = u .⪝ x, Π∇f .< 0.
        𝔩, inc = x .⪝ l, Π∇f .> 0.
        Π∇f[(𝔲 .& dec) .| (𝔩 .& inc)] .= 0.
        Π∇f
    end

    init!(descent, get_Πf, get_Π∇f, x)
    @memento Π∇f = get_Π∇f(x)
    @memento normΠ∇f = norm(Π∇f, Inf)
    verba(1, "||Π∇f|| : $normΠ∇f")
    for i in 1:max_iter
        if normΠ∇f < ε
            verba(0, "\nIterations: $i\n")
            break
        end

        @memento x = get_Πx(step!(descent, get_Πf, get_Π∇f, x))
        verba(2, "x : $x")

        @memento Π∇f = get_Π∇f(x)
        verba(2, "Π∇f : $Π∇f")
        @memento normΠ∇f = norm(Π∇f, Inf)
        verba(1, "||Π∇f|| : $normΠ∇f")
    end

    @memento f = get_f(x)
    verba(0, "f = $f")
    result = @get_result x Π∇f normΠ∇f f
    OptimizationResult{MQBProblem}(memoria=@get_memoria, result=result)
end

# -------------- Quadratic Boxed Problem Generator -------------- #
# TODO: Add custom active constraints %
function generate_quadratic_boxed_problem(type, n; active=0, singular=0)
    E = rand(type, n-singular, n)
    x = rand(type, n)
    q = -E*x
    q = [q; zeros(type, singular)]
    l, u = -10.0*rand(type, n)+x, 10.0*rand(type, n)+x
    active = min(active, n)
    l[n-active+1:n] .-= 11.
    u[n-active+1:n] .-= 11.

    return MQBProblem(E'E, q, l, u)
end

# ----------- Quadratic Boxed Problem - Algorithm Tester ------------- #
function get_test(algorithm::OptimizationAlgorithm{MQBProblem};
    n::Integer,
    singular::Integer=0,
    active::Integer=0,
    𝔓::Union{Nothing, MQBProblem}=nothing,
    type::DataType=Float64)

    if 𝔓 === nothing
        𝔓 = generate_quadratic_boxed_problem(type, n, active=active, singular=singular)
    end

    instance = OptimizationInstance{MQBProblem}()
    set!(instance, 
        problem=𝔓, 
        algorithm=algorithm, 
        options=MQBPSolverOptions(),
        solver=OptimizationSolver{MQBProblem}())
    return instance
end

#   Usage example
# include("minquadratic.jl")
#   then
# algorithm = MQBPAlgorithmPG1(descent=AdagradDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=0.)
# test = get_test(algorithm, n=10)
# test.solver.options.memoranda = Set(["normΠ∇f"])
#   (or, oneliner)
# algorithm = MQBPAlgorithmPG1(descent=AdagradDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=0.); test = get_test(algorithm, n=10); test.solver.options.memoranda = Set(["normΠ∇f"])