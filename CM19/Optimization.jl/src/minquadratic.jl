module MinQuadratic

using Parameters
using LinearAlgebra
using SparseArrays
using DataStructures

using ..Optimization
using ..Optimization.Utils
import ..Optimization.run!      # Necessary since we extend here the multiple dispatch
import ..Optimization.set!      # idem
using ..Optimization.Descent

"""
```julia
MQBProblem <: OptimizationProblem
```

**Convex Quadratic Boxed Problem**

`minₓ ½ x'Qx + q'x` where  `l .≤ x .≤ u` and `Q ≽ 0`

**Members**
* `Q`
* `q`
* `l`
* `u`

"""
struct MQBProblem <: OptimizationProblem
    Q
    q
    l
    u
end

"""
```julia
MQBPSolverOptions <: OptimizationSolverOptions{MQBProblem}
```

Options for a convex quadratic boxed problem solver.

**Members**
* `memoranda :: Set{String}` : variables that we want to track with the `@memento` macro

"""
mutable struct MQBPSolverOptions <: OptimizationSolverOptions{MQBProblem}
    memoranda::Set{String}  # set of variables that we want to track

    MQBPSolverOptions() = new(Set{String}([]))
    MQBPSolverOptions(memoranda::Set{String}) = new(memoranda)
end

"""
```julia
run!(solver, problem)
```

**Arguments**
* `solver :: OptimizationSolver{MQBProblem}`
* `problem :: MQBProblem`

"""
function run!(solver::OptimizationSolver{MQBProblem}, problem::MQBProblem)
    run!(solver.algorithm, problem, memoranda=solver.options.memoranda)
end

struct Oᾱ <: Base.Order.Ordering
    simeq
end
import Base.Order.lt
"""
```julia
lt(o::Oᾱ, a::Tuple{CartesianIndex{2}, AbstractFloat}, b::Tuple{CartesianIndex{2}, AbstractFloat})
```

Implements an ordering which should partially obviate to the problems of floating point number errors
occurring while sorting the crossing of a line search with a set of hyperplanes

**Arguments**
* `o :: Oᾱ` : specific approximate ordering for the ᾱ
* `a :: Tuple{CartesianIndex{2}, AbstractFloat}` :
* `b :: Tuple{CartesianIndex{2}, AbstractFloat}` :

"""
lt(o::Oᾱ, a::Tuple{CartesianIndex{2}, AbstractFloat}, b::Tuple{CartesianIndex{2}, AbstractFloat}) = begin
    o.simeq(a[2] , b[2]) ?
        (a[1][2], a[1][1]) < (b[1][2], b[1][1]) :
        a[2] < b[2]
end

include("algorithm/MQBP_P_PG.jl")

# -------------- Quadratic Boxed Problem Generator -------------- #
# TODO: Add custom active constraints %
function generate_quadratic_boxed_problem(type, n; active=0, singular=0)
    E = rand(type, n-singular, n)
    x = rand(type, n)
    q = -E*x
    q = [q; zeros(type, singular)]
    l, u = -10.0*rand(type, n) + x, 10.0*rand(type, n) + x
    active = min(active, n)
    l[n-active+1:n] .-= 11.
    u[n-active+1:n] .-= 11.

    return MQBProblem(E'E, q, l, u)
end

"""
```julia
get_test(algorithm; n, singular=0, active=0, 𝔓=nothing, type=Float64)
```

Prepare a test for a Quadratic Boxed Problem solver algorithm.

**Associated Optimization Problem**

`minₓ ½ x'Qx + q'x` where  `l .≤ x .≤ u`

**Returns**
* An `OptimizationInstance` adequate to the algorithm with a randomly generated problem if no problem is specified

**Arguments**
* `algorithm :: OptimizationAlgorithm{MQBProblem}`
* `n :: Integer` : dimension of the problem (e.g. `size(Q) = [n, n]`)
* `singular :: Integer` : dimension of `ker(Q)`
* `active :: Integer` : (heuristic) count of the probable active sets of the solution
* `𝔓 :: Union{Nothing, MQBProblem}` : problem; if `nothing` is given, it is randomly generated
* `type :: DataType` : type to be used for `eltype` of the arrays of the problem

**Example**
```julia
algorithm = MQBPAlgorithmPG1(descent=AdagradDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=0.)
test = get_test(algorithm, n=10)
test.solver.options.memoranda = Set(["normΠ∇f"])
run!(test)
```

"""
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
    Optimization.set!(instance,
        problem=𝔓,
        algorithm=algorithm,
        options=MQBPSolverOptions(),
        solver=OptimizationSolver{MQBProblem}())
    return instance
end

export set!,
    run!,
    QuadraticBoxPCGDescent,
    MQBProblem,
    MQBPSolverOptions,
    MQBPAlgorithmPG1,
    generate_quadratic_boxed_problem,
    get_test
end     # end module MinQuadratic

#   Usage example
# algorithm = MQBPAlgorithmPG1(descent=AdagradDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=0.)
# test = get_test(algorithm, n=10)
# test.solver.options.memoranda = Set(["normΠ∇f"])
#   (or, oneliner)
# algorithm = MQBPAlgorithmPG1(descent=AdagradDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=0.); test = get_test(algorithm, n=10); test.solver.options.memoranda = Set(["normΠ∇f"])
