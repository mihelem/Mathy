"""
    MinCostFlow submodule of Optimization


**Examples**

__Quadratic MinCostFlow Boxed__
```julia
algorithm = QMCFBPAlgorithmD1(descent=GradientDescent(), verbosity=0, max_iter=20, ϵₘ=1e-10, ε=1e-5, cure_singularity=false)
test = get_test(algorithm, m=5, n=8, singular=0)
run!(test)
θ = test.result.memoria["θ"]    # fictitious θ
do_plot = i -> Plots.plot([j for j in 1:length(θ[i])], θ[i])
```
"""
module MinCostFlow

using LinearAlgebra
using SparseArrays
using Parameters
using DataStructures: PriorityQueue, peek, dequeue!, enqueue!, Queue

using ..Optimization
using ..Optimization.MinQuadratic
using ..Optimization.Utils

import ..Optimization: run!, set!
import ..Optimization.MinQuadratic: get_test
import ..Optimization.Descent: init!


"""
    MinCostFlowProblem
Algorithms deployed in this submodule describes solutions for subproblems of the kind MinCostFlow
"""
abstract type MinCostFlowProblem <: OptimizationProblem end

"""
    QMCFBProblem <: MinCostFlowProblem
## (Convex) Quadratic Min Cost Flow Boxed Problem
`minₓ { ½xᵀQx + qᵀx  with  x s.t.  Ex = b  &  l ≤ x ≤ u }`
with
* `Q ∈ { diag ≥ 0 }`
* `E` : node-arc incidence matrix of directed graph, `rank ≡ nodes - connected components` if unreduced
* `reduced` : boolean indicating if the problem has been reduced, in which case E is full rank and represent the incidence matrix of a connected graph minus 1 vertex
"""
struct QMCFBProblem <: MinCostFlowProblem
    Q
    q
    l
    u
    E
    b
    reduced::Bool
end

# ---------------------------- Solver Options --------------------------- #
mutable struct QMCFBPSolverOptions <: OptimizationSolverOptions{QMCFBProblem}
    memoranda::Set{String}  # set of variables that we want to track

    QMCFBPSolverOptions() = new(Set{String}([]))
    QMCFBPSolverOptions(memoranda::Set{String}) = new(memoranda)
end

# ----------------------------- Solver runner --------------------------- #
"""
```julia
run!(solver, problem)
```

**Arguments**
* `solver :: OptimizationSolver{QMCFBProblem}`
* `problem :: QMCFBProblem`

"""
function run!(solver::OptimizationSolver{QMCFBProblem}, problem::QMCFBProblem)
    run!(solver.algorithm, problem, memoranda=solver.options.memoranda)
end

# (P)D1 : Saddle Point with Descents Methods
include("algorithm/QMCFBP_PD1_D.jl")
# D1 : Subgradient Methods - WIP
include("algorithm/QMCFBP_D1_SG.jl")
# D1 : Descent Methods, also with specialised exact line search - WIP
include("algorithm/QMCFBP_D1_D.jl")
# D1 : Commons
include("algorithm/QMCFBP_D1.jl")
# D2 : Descent Methods, also with specialised exact line search -WIP
include("algorithm/QMCFBP_D2_D.jl")
# D3 : Descent Methods - WIP
include("algorithm/QMCFBP_D3_D.jl")

# --------------------------- Heuristic -----------------------------------
# About heuristic: Projecting in the feasible space could be done with the
# subgradient method alone exploiting the alternating projection algorithm
# A network specific solution should be more effective.
include("heuristic/QMCFBP_D1_BFS.jl")

# -------------- Quadratic Min Cost Flow Boxed Problem Generator ----------
function generate_quadratic_min_cost_flow_boxed_problem(
    type,
    m,
    n;
    singular=0,
    active=0)

    Q = spdiagm(
        0 => [sort(rand(type, n-singular), rev=true); zeros(type, singular)])
    E = spzeros(Int8, m, n)
    for i=1:n
        u, v = (rand(1:m), rand(1:m-1))
        v = u==v ? m : v
        E[u, i] = 1
        E[v, i] = -1
    end
    x = rand(eltype(Q), n)
    l = -10*rand(eltype(Q), n)+x
    u = 10*rand(eltype(Q), n)+x
    Qu, Ql = Q*u, Q*l

    # prepare q such that the optimal point of the free    ½xᵀQx+qᵀx
    # has minimum outside of active side
    P = [1:n;]
    lu = CartesianIndex{2}[]
    for i in 1:active
        k = rand(1:2)
        j = rand(i:n)
        P[j], P[i] = P[i], P[j]
        push!(lu, CartesianIndex{2}(P[i], k))
    end
    𝔭 = zeros(Bool, n, 2)
    𝔭[lu] .= true
    q = rand(n) |>
        r -> -Qu + (Qu-Ql).*(𝔭[:, 1] + r.*(1 .- (𝔭[:, 2] - 𝔭[:, 1]))) + r.*(𝔭[:,1]-𝔭[:,2])
    q[end-singular+1:end] +=
        (rand(singular) .* (𝔭[end-singular+1:end, 1] - 𝔭[end-singular+1:end, 2]))

    # choose b such that there is an x :  l ≤ x ≤ u  ∩  Ex-b = 0
    b = E*x
    return QMCFBProblem(Q, q, l, u, E, b, false)
end

function noNaN(V)
    return (x -> isnan(x) ? 0. : x).(V)
end

"""
# ----------- Quadratic Min Cost Flow Boxed Problem - Algorithm Tester -------------
"""
function get_test(algorithm::OptimizationAlgorithm{QMCFBProblem};
    m::Integer=0, n::Integer=0,
    singular::Integer=0,
    active::Integer=0,
    𝔓::Union{Nothing, QMCFBProblem}=nothing,
    should_reduce::Bool=false,
    type::DataType=Float64,)

    if 𝔓 === nothing
        𝔓 = generate_quadratic_min_cost_flow_boxed_problem(type, m, n, singular=singular, active=active)
        if should_reduce == true
            𝔓 = get_reduced(𝔓)[1]
        end
    end

    instance = OptimizationInstance{QMCFBProblem}()
    set!(instance,
        problem=𝔓,
        algorithm=algorithm,
        options=QMCFBPSolverOptions(),
        solver=OptimizationSolver{QMCFBProblem}())
    return instance
end

# --------------------------- Incidence Matrix Utils --------------------- #
"""
    get_graph_components(E)
Connected components matrix-wise
**Arguments**
* `E` : node-arc incidence matrix
"""
function get_graph_components(E)
    # m : number of nodes
    # n : number of arcs
    m, n = size(E)
    M = E .≠ 0
    B = zeros(Bool, m)
    P = zeros(Bool, m, 0)
    P_C = zeros(Bool, 0, n)
    for i in 1:m
        if B[i] == true
            continue
        end

        P = cat(P, zeros(Bool, m), dims=2)
        P_C = cat(P_C, zeros(Bool, 1, n), dims=1)

        B[i] = true
        P[i, end] = true

        Vᵢ = begin
            P_C[end, :] = M[i, :]
            N = M[:, M[i, :]]
            if size(N, 2) == 0
                zeros(Bool, m)
            else
                V = (.~(B)) .& reduce((a, b) -> a .| b, [N[:, i] for i in 1:size(N, 2)])
                B .|= V
                V
            end
        end

        if any(Vᵢ) == false
            continue
        end

        P[:, end] .|= Vᵢ
        stack = findall(Vᵢ)

        j = 1
        while j ≤ size(stack, 1)
            Vⱼ = begin
                P_C[end, :] .|= M[stack[j], :]
                N = M[:, M[stack[j], :]]
                if size(N, 2) == 0
                    zeros(Bool, m)
                else
                    V = (.~(B)) .& reduce((a, b) -> a .| b, [N[:, k] for k in 1:size(N, 2)])
                    B .|= V
                    V
                end
            end
            j += 1
            if any(Vⱼ) == false
                continue
            end

            P[:, end] .|= Vⱼ
            append!(stack, findall(Vⱼ))
        end
    end

    return (P, P_C)
end

function incidence_to_adjacency(E)
    A = size(E)[1] |> m -> spzeros(Int64, m, m)
    for edge in 1:size(E, 2)
        i_s, i_t = [nzrange(E, edge);]
        if nonzeros(E)[i_s] == 1
            i_s, i_t = i_t, i_s
        end

        s, t = (i->rowvals(E)[i]).([i_s, i_t])
        A[s, t] += 1
    end
    A
end

"""
    get_reduced(𝔓::QMCFBProblem)
Return the MinCostFlow problem corresponding to the first of the connected components in which the MinCostFlow problem can be separated

**Arguments**
* `𝔓::QMCFBProblem` : Quadratic MinCostFlow Boxed Problem whose node-arc incidence matrix may represent a disconnected digraph

**Note**

After reduction, the new incidence matrix `𝔓.E` will have dimension of the left kernel equal to 1 ≡> (1,1,...,1)
"""
function get_reduced(𝔓::QMCFBProblem)
    @unpack Q, q, l, u, E, b = 𝔓
    P_row, P_col = get_graph_components(E)
    return [
        QMCFBProblem(
            Q[p_col, p_col],
            q[p_col],
            l[p_col],
            u[p_col],
            E[p_row, p_col],
            b[p_row],
            true) for (p_row, p_col) in zip(eachcol(P_row), eachrow(P_col))]
end

export  run!,
        init!,
        set!,
        QMCFBProblem,
        get_test,
        get_reduced,
        get_graph_components,
        generate_quadratic_min_cost_flow_boxed_problem,
        QMCFBPAlgorithmD3D,
        QMCFBPAlgorithmD2D,
        QMCFBPAlgorithmD1D,
        QMCFBPAlgorithmD1SG,
        QMCFBPAlgorithmPD1D,
        BFSHeuristic,
        QMCFBPSolverOptions,
        MinCostFlowProblem
end     # end of module MinCostFlow

"""
In the REPL, the following snippet may be useful to manually explore the
space of parameters. TODO -> make a macro working for every algo/subgradient etc..
```julia
using Optimization.Utils
function rupl!(;
        α=nothing,
        β=nothing,
        p=nothing,
        to_plot="norm∂L′",
        map=x->log10.(x))
   @some algorithm.subgradient.α = α
   @some algorithm.subgradient.β = β
   run!(test)
   if p === nothing
       return plot(test.result, to_plot, "i′", mapping=map)
   else
       plot!(p, test.result, to_plot, "i′", mapping=map)
   end
end
```

Then to explore the pseudospectra of the incidence matrix:
```julia
test = get_test(algorithm, m=20, n=40, singular=8, active=100);
𝔓 = test.problem;
Q, q, l, u, E, b = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b);
spectralportrait(Array(E'E))
```

To check correlation with vertex degree
```julia
function print_degree(E)
   Ep = copy(E)
   Ep[Ep.<1] .= 0
   Em = copy(E)
   Em[Em.>-1] .= 0
   p = sum([c for c in eachcol(Ep)])
   m = sum([c for c in eachcol(Em)])
   p, m
end

println.([zip(print_degree(E)...)...]);
```
"""
