"""
Main module of the package

## Types and Structs
`OptimizationProblem` is the **abstract type** whose subtypes, say `Problem`, parametrize all the other main types and structures.

### Examples:
**Testing MinQuadratic**
```julia
using Optimization
# build an OptimizationAlgorithm{MQBProblem}, where MQBProblem is the optimization problem
# min_x ½x'Qx+qx     with box constraints    l .<= x .<= u     and    Q >= 0
algorithm =
	MQBPAlgorithmPG1(
		descent=QuadraticBoxPCGDescent(),
		verbosity=1,
		max_iter=1000,
		ε=1e-7,
		ϵ₀=1e-12)
# generate an OptimizationInstance{MQBProblem} using the specified algorithm, with Q in the problem of size 200x200
test = get_test(algorithm, n=200)
# set a specific variable to be tracked during the execution of the algorithm
# the set of variables which CAN be tracked ought to be specified (by the developer of the algorithm) in algorithm.memorabilia
test.solver.options.memoranda = Set(["normΠ∇f"])
# run the instance, so run the solver with the specified algorithm on the specified problem; the result is saved in `test`
run!(test)
```

---

`OptimizationInstance{Problem}` contains a
* `problem::Problem` : the problem
* `solver::OptimizationSolver{>: Problem}` : a solver which solves supertypes of the problem
* `result::OptimizationResult{Problem}` : the result of the solver applied to the problem

---

`OptimizationSolver{Problem} contains an
* `algorithm::OptimizationAlgorithm{>: Problem}` : an algorithm which can handle supertypes of the problem
* `options::OptimizationSolverOptions{>: Problem}` : specific options for the solver, e.g. which keys to save in `memoria`

---

`OptimizationResult{Problem}` is up to now independent from the parameter, which is used just to know which problem the result is concerned with;
contains:
* `result::Dict{String, Any}` : results, saved in a dictionary; usually generated with the macro `@get_result` in utils.jl
* `memoria::Dict{String, Any}` : with *memoria* macros in utils.jl, to save the intermediate results of the computation, e.g. for plots
* `plots::Dict{String, Plots.plot}` : saving here the plots generated from intermediate data

"""
module Optimization

using LinearAlgebra
using Parameters
import Plots
import Plots: plot, plot!

include("utils.jl")
using .Utils

export  OptimizationInstance,
        OptimizationProblem,
        OptimizationAlgorithm,
        OptimizationSolver,
        OptimizationSolverOptions,
        OptimizationResult,
        LocalizationMethod,
        Heuristic,
        set_param!,
        plot!,
        plot

abstract type OptimizationProblem end

# Solver, Algorithms, Options, Results are parametrized by the OptimizationProblem
abstract type OptimizationSolverOptions{Problem <: OptimizationProblem} end
abstract type OptimizationAlgorithm{Problem <: OptimizationProblem} end
mutable struct OptimizationSolver{Problem <: OptimizationProblem}
    algorithm::OptimizationAlgorithm{>: Problem}
    options::OptimizationSolverOptions{>: Problem}

    OptimizationSolver{Problem}() where {Problem <: OptimizationProblem} = new()
end
"""
```julia
set!(solver; algorithm = nothing, options = nothing)
```

**Arguments**
* `solver :: OptimizationSolver{P}`
* `algorithm :: Union{Nothing, OptimizationAlgorithm{>: P}}`
* `options :: Union{Nothing, OptimizationSolverOptions{>: P}}`

"""
function set!(solver::OptimizationSolver{P};
    algorithm::Union{Nothing, OptimizationAlgorithm{>: P}} = nothing,
    options::Union{Nothing, OptimizationSolverOptions{>: P}} = nothing) where {P <: OptimizationProblem}

    @some solver.algorithm = algorithm
    @some solver.options = options
end

# ------------------------ Result of Computation ------------------------ #
mutable struct OptimizationResult{Problem <: OptimizationProblem}
    result::Dict{String, Any}
    memoria::Dict{String, Any}
    plots::Dict{String, Plots.plot}

    OptimizationResult{Problem}(;memoria=nothing, plots=nothing, result=nothing) where Problem  <: OptimizationProblem = begin
        object = new()
        @some object.memoria = memoria
        @some object.plots = plots
        @some object.result = result
        object
    end
end
"""
```julia
plot!(cur_plot, result, meme)
```

Add to the plot `cur_plot` the intermediate data with key `meme` saved in `result.memoria`
Arguments:
* `cur_plot :: Plots.Plot`
* `result :: OptimizationResult`
* `meme :: String`

"""
function plot!(
    cur_plot::Plots.Plot,
    result::OptimizationResult,
    meme::String,
    meme_iter::Union{String, Nothing}=nothing;
    mapping=x->x)

    if haskey(result.memoria, meme) === false
        return
    end
    result.memoria[meme] |>
    data -> mapping(data) |>
    data′ -> begin
        if (meme_iter !== nothing) && (haskey(result.memoria, meme_iter) == true)
            data_iter = result.memoria[meme_iter]
            len = min(length(data_iter), length(data′))
            Plots.plot!(cur_plot, data_iter[1:len], data′[1:len])
        else
            Plots.plot!(cur_plot, 1:length(data′), data′)
        end
    end
end

"""
```julia
plot(result, meme)
```

Plot the intermediate data with key `meme` as retrieved from `result.memoria`

**Arguments:**
* `result :: OptimizationResult`
* `meme :: String`

"""
function plot(
    result::OptimizationResult,
    meme::String,
    meme_iter::Union{String, Nothing}=nothing;
    mapping=x->x)

    if haskey(result.memoria, meme) === false
        return
    end
    result.memoria[meme] |>
    data -> mapping(data) |>
    data′ -> begin
        if (meme_iter !== nothing) && (haskey(result.memoria, meme_iter) == true)
            data_iter = result.memoria[meme_iter]
            len = min(length(data_iter), length(data′))
            Plots.plot(data_iter[1:len], data′[1:len])
        else
            Plots.plot(1:length(data′), data′)
        end
    end
end

"""
```julia
set!(result, meme, cur_plot)
```

Save the plot `cur_plot` in the dictionary `result.plots` with the key `meme`

**Arguments**
* `result :: OptimizationResult`
* `meme :: String`
* `cur_plot :: Plots.Plot`

"""
function set!(
    result::OptimizationResult,
    meme::String,
    cur_plot::Plots.Plot)

    result.plots[meme] = cur_plot
    result
end

mutable struct OptimizationInstance{Problem <: OptimizationProblem}
    problem::Problem
    solver::OptimizationSolver{>: Problem}
    result::OptimizationResult{Problem}

    OptimizationInstance{Problem}() where {Problem <: OptimizationProblem} = new()
end
"""
```julia
set!(instance; problem = nothing, solver = nothing, result = nothing, algorithm = nothing, options = nothing)
```

**Arguments**
* `instance :: OptimizationInstance{P}`
* `problem :: Union{Nothing, P}`
* `solver :: Union{Nothing, OptimizationSolver{>: P}}`
* `result :: Union{Nothing, OptimizationResult{P}}`
* `algorithm :: Union{Nothing, OptimizationAlgorithm{>: P}}`
* `options :: Union{Nothing, OptimizationSolverOptions{>: P}}`

"""
function set!(
        instance::OptimizationInstance{P};
        problem::Union{Nothing, P} = nothing,
        solver::Union{Nothing, OptimizationSolver{>: P}} = nothing,
        result::Union{Nothing, OptimizationResult{P}} = nothing,
        algorithm::Union{Nothing, OptimizationAlgorithm{>: P}} = nothing,
        options::Union{Nothing, OptimizationSolverOptions{>: P}} = nothing) where {P <: OptimizationProblem}

    @some instance.problem = problem
    @some instance.solver = solver
    @some instance.result = result
    set!(instance.solver, algorithm=algorithm, options=options)
end
"""
```julia
run!(instance)
```

**Arguments**
* `instance :: OptimizationInstance`

"""
function run!(instance::OptimizationInstance)
    set!(instance, result=run!(instance.solver, instance.problem))
end

"""
```julia
set!(instance, meme, cur_plot)
```

Save the plot `cur_plot` in the dictionary `instance.result.plots` with the key `meme`

**Arguments**
* `instance::OptimizationInstance`
* `meme::String`
* `cur_plot::Plots.Plot`

"""
function set!(
        instance::OptimizationInstance,
        meme::String,
        cur_plot::Plots.Plot)

    set!(instance.result, meme, cur_plot)
    instance
end

abstract type LocalizationMethod end
function set_param!(M::LocalizationMethod, s::Symbol, v)
    setfield!(M, s, v)
end

# todo: stronger typing
abstract type Heuristic end

include("linesearch.jl")
using .LineSearch
export  bracket_minimum,
        fibonacci_search,
        fibonacci_as_power_search,
        golden_section_search,
        line_search

include("descent.jl")
using .Descent
export  ZerothOrder,
        NelderMead,
        DescentMethod,
        init!,
        step!,
        set_param!,
        GradientDescent,
        ConjugateGradientDescent,
        MomentumDescent,
        NesterovMomentumDescent,
        AdagradDescent,
        RMSPropDescent,
        AdadeltaDescent,
        AdamDescent,
        HyperGradientDescent,
        HyperNesterovMomentumDescent,
        NoisyDescent

include("subgradient.jl")
using .Subgradient
export  Subgradient,
        SubgradientMethod,
        DualSubgradientMethod,
        DeflectedSubgradientMethod,
        init!,
        step!,
        set_param!

include("minquadratic.jl")
using .MinQuadratic
export  set!,
        run!,
        QuadraticBoxPCGDescent,
        MQBProblem,
        MQBPSolverOptions,
        MQBPAlgorithmPG1,
        generate_quadratic_boxed_problem,
        get_test

include("mincostflow.jl")
using .MinCostFlow
export  run!,
        set!,
        init!,
        QMCFBProblem,
        get_test,
        get_reduced,
        get_graph_components,
        generate_quadratic_min_cost_flow_boxed_problem,
        QMCFBPAlgorithmD3D,
        QMCFBPAlgorithmD2D,
        QMCFBPAlgorithmD1D,
        QMCFBPAlgorithmD1SG,
        BFSHeuristic,
        QMCFBPAlgorithmPD1D,
        QMCFBPSolverOptions,
        MinCostFlowProblem

include("numerical.jl")
using .Numerical
export  bidiagonal_decomposition_handmade2,
        bidiagonal_decomposition_handmade,
        GMRES_naive,
        Arnoldi_naive,
        Arnoldi_iterations,
        rayleigh_inverse_iteration,
        rayleigh_iteration,
        hessenberg_via_householder,
        choleski_factorisation,
        gaussian_elimination_row_pivot,
        gaussian_elimination,
        hessenberg_gram_schmidt,
        QR_gram_schmidt

include("hyper.jl")
using .Hyper
export WithParameterSearch,
    set!,
    run!,
    plot,
    plot!


"""
### ♂ TO BE ERASED (EXPERIMENTS)
Experiments in optimization

Naive Implementation

**Assumptions**:
Everything invertible, Q positive definite
"""
function solve_reduced_KKT(Q, q, A, b, x, ϵ)
    Q˜ = inv(Q)
    M = A*Q˜*A'
    μ̅ = -inv(M)*(b + A*Q˜*q)
    x̅ = -Q˜*(q+A'*μ̅)

    return (x̅, μ̅)
end
function active_set_method_quadratic(Q, q, A, b, x, ϵ)
    B = zeros(Bool, size(A, 1))
    while true
        x̅, μ̅  = solve_reduced_KKT(Q, q, A[B, :], b[B], x[:], ϵ)
        C = .!B .& (A*x̅ .> b)
        println(B)
        if any(C)
            d = x̅-x
            α = minimum( (b[C]-A[C, :]*x) ./ (A[C, :]*d) )
            x = x + α*d
            B = A*x .== b
        else
            μ̅ .< 0
            O = findall(x -> x, B)[μ̅ .< 0]
            if isempty(O)
                return x̅
            else
                B[O[1]] = false
            end
        end
    end
end

end # end module Optimization

# examples
# using Revise
# includet("optimization.jl")
# using .Optimization
# algorithm = MQBPAlgorithmPG1(descent=QuadraticBoxPCGDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=1e-8)
# test = get_test(algorithm, n=10)
# test.solver.options.memoranda = Set(["normΠ∇f"])
# include("optimization.jl"); using .Optimization; algorithm = MQBPAlgorithmPG1(descent=QuadraticBoxPCGDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=1e-8); test = get_test(algorithm, n=10); test.solver.options.memoranda = Set(["normΠ∇f"])

# How to save Plots:
"""
```julia
#First, generate your plot using Plots.jl:
using Plots
hdf5() #Select HDF5-Plots "backend"
p = plot(...) #Construct plot as usual

#Then, write to .hdf5 file:
Plots.hdf5plot_write(p, "plotsave.hdf5")

#After you re-open a new Julia session, you can re-read the .hdf5 plot:
using Plots
pyplot() #Must first select some backend
pread = Plots.hdf5plot_read("plotsave.hdf5")
display(pread)
```
"""
