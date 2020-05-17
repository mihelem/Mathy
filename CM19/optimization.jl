using LinearAlgebra

# TODO: here is just for a single assignment, 
# need to be extended to handle general expressions
macro some(arg)
    if typeof(arg) === Expr
        if arg.head === :(=)
            quote
                if $(arg.args[2]) !== nothing
                    $(arg.args[1]) = $(arg.args[2])
                end
            end |> esc
        end
    end
end

abstract type OptimizationProblem end

# Solver, Algorithms, Options, Results are parametrized by the OptimizationProblem
abstract type OptimizationSolverOptions{Problem <: OptimizationProblem} end
abstract type OptimizationAlgorithm{Problem <: OptimizationProblem} end
mutable struct OptimizationSolver{Problem <: OptimizationProblem}
    algorithm::OptimizationAlgorithm{>: Problem}
    options::OptimizationSolverOptions{>: Problem}

    OptimizationSolver{Problem}() where {Problem <: OptimizationProblem} = new() 
end
function set!(solver::OptimizationSolver{P}; 
    algorithm::Union{Nothing, OptimizationAlgorithm{>: P}} = nothing, 
    options::Union{Nothing, OptimizationSolverOptions{>: P}} = nothing) where {P <: OptimizationProblem}
    
    @some solver.algorithm = algorithm
    @some solver.options = options
end

abstract type OptimizationResult{Problem <: OptimizationProblem} end

mutable struct OptimizationInstance{Problem <: OptimizationProblem}
    problem::Problem
    solver::OptimizationSolver{>: Problem}
    result::OptimizationResult{Problem}

    OptimizationInstance{Problem}() where {Problem <: OptimizationProblem} = new()
end
function set!(instance::OptimizationInstance{P};
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
function run!(instance::OptimizationInstance{P}) where {P <: OptimizationProblem}
    set!(instance, result=run!(instance.solver, instance.problem))
end



# ♂ TO BE ERASED (EXPERIMENTS)
# Experiments in optimization

# Naive Implementation
# Assumptions:
#  Everything invertible, Q positive definite

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