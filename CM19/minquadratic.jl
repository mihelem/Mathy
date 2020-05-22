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

struct Oᾱ <: Base.Order.Ordering 
    simeq
end
import Base.Order.lt
lt(o::Oᾱ, a::Tuple{CartesianIndex{2}, AbstractFloat}, b::Tuple{CartesianIndex{2}, AbstractFloat}) = begin
    o.simeq(a[2] , b[2]) ?
        (a[1][2], a[1][1]) < (b[1][2], b[1][1]) :
        a[2] < b[2]
end
# Dummy struct, since for projected methods the step! should need a 
# signature different from the other DescentMethod s
mutable struct QuadraticBoxPCGDescent <: DescentMethod end
mutable struct MQBPAlgorithmPG1 <: OptimizationAlgorithm{MQBProblem}
    descent::DescentMethod      # 
    verba                       # verbosity utility
    max_iter                    #
    ε                           # required: norm(∇f, ?) < ε
    ϵ₀                          # abs error to which inequalities are satisfied
    x₀                          # starting point

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
        algorithm.memorabilia = Set(["normΠ∇f", "Π∇f", "x", "f", "d"])
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

    x = (x₀ === nothing) ? 0.5*(l+u) : x₀
    a::AbstractFloat ⪝ b::AbstractFloat = a ≤ b + ϵ₀
    a::AbstractFloat ≃ b::AbstractFloat = abs(a-b) ≤ ϵ₀
    to0 = (x::AbstractFloat -> x ≃ 0. ? 0. : x)

    # Box Projectors
    # Coordinate Space
    Π = (x, l, u) -> ((u .⪝ x) .| (x .⪝ l))
    Π! = (x, l, u) -> (x[:] = min.(max.(x, l), u))
    # Tanget Space
    ΠᶜT = (d, x, l, u) -> begin
        𝔲, dec = u .⪝ x, d .> 0.
        𝔩, inc = x .⪝ l, d .< 0.
        (𝔲 .& dec) .| (𝔩 .& inc)
    end
    ΠT = (d, x, l, u) -> begin
        .~ΠᶜT(d, x, l, u)
    end
    ΠT! = (d, x, l, u) -> begin
        d[ΠᶜT(d, x, l, u)] .= 0.
        d
    end

    # 
    get_Πx = (x, l, u) -> min.(max.(x, l), u)
    get_f = (Πx, Q, q) -> 0.5*Πx'Q*Πx + q'Πx
    get_Πf = (x, Q, q, l, u) -> get_f(get_Πx(x, l, u), Q, q)
    get_∇f = (Πx, Q, q) -> Q*Πx+q

    get_Π∇f = (x, Q, q, l, u) -> begin
        Πx = get_Πx(x, l, u)
        ∇f = get_∇f(Πx, Q, q)
        -ΠT!(-∇f, x, l, u)
    end


    function on_box_side(x)
        𝔅 = [x .⪝ l   u .⪝ x]
    end
    on_u = 𝔅 -> 𝔅[:, 2]
    on_l = 𝔅 -> 𝔅[:, 1]
    # ᾱ is an α corresponding to the line crossing a side of the box
    # assuming a valid  l .≤ x .≤ u
    function get_ᾱs(x, d, l, u)
        # 1 : getting inside
        # 2 : going outside
        ᾱs = zeros(eltype(d), length(d), 2) .- Inf

        𝔩 = [d .> 0  d .< 0]        
        ᾱs[𝔩] = ([l l][𝔩] - [x x][𝔩]) ./ [d d][𝔩]

        𝔲 = [d .< 0  d .> 0]
        ᾱs[𝔲] = ([u u][𝔲] - [x x][𝔲]) ./ [d d][𝔲]

        return (ᾱs, 𝔩, 𝔲)
    end
    function filter_ᾱs(ᾱs, min_α=-100*ϵ₀, max_α=Inf)
        F_ᾱs = findall( (ᾱs .> min_α) .& (ᾱs .< max_α) .& (.~isnan.(ᾱs)) )
    end

    # First approach: sort all ᾱs, then: 1- scan 2-binary search
    function sort_ᾱs(F_ᾱs, ᾱs)
        P_ᾱs = sort(F_ᾱs, lt = (i, j) -> ᾱs[i] ≃ ᾱs[j] ? (i[2], i[1]) < (j[2], j[1]) : ᾱs[i] < ᾱs[j])
    end
    # Second approach:  since usually we'll stop at one of the first ᾱs,
    #                   use a Priority Queue => ~ linear time
    function filter_ᾱ(p::CartesianIndex{2}, 𝔅)
        (p[2] == 1) == (𝔅[p[1], 1] | 𝔅[p[1], 2])
    end
    function priority_ᾱs(F_ᾱs, ᾱs)
        pq = PriorityQueue{CartesianIndex{2}, Tuple{CartesianIndex{2}, AbstractFloat}}(Oᾱ(≃))
        for i in F_ᾱs
            pq[i] = (i, ᾱs[i])
        end
        pq
    end
    
    function get_x(x, d, α, ᾱs)
        x + d .* mid.(α, ᾱs[:, 1], ᾱs[:, 2])
    end
    function get_x(x, αd, 𝔅)
        .~(𝔅[:, 1] .| 𝔅[:, 2]) |> 
            𝔉 -> l.*𝔅[:, 1] + u.*𝔅[:, 2] + (x + αd).*𝔉
    end

    function line_search(pq::PriorityQueue{CartesianIndex{2}, Tuple{CartesianIndex{2}, AbstractFloat}}, x, d, Q, q, 𝔩, 𝔲, 𝔅)
        𝔉 = .~(𝔅[:, 1] .| 𝔅[:, 2])
        verba(1, "line_search : $(count(𝔉)) inactive")
        d′ = d .* 𝔉
        if count(𝔉) > 0
            Δα = (Q*d′ |> Qd -> (d′⋅q + x'Qd, Qd'd′))
            if Δα[2] == 0. 
                verba(1, "line_search : d⋅Qd = $(Δα)") 
            end
            if Δα[1] > 0.
                return x
            elseif length(pq) == 0 || -Δα[1]/Δα[2] ⪝ peek(pq)[2][2]
                return x - Δα[1] * d′ / Δα[2]
            end
        end
        while length(pq) > 0
            i, ᾱ = peek(pq)[2]
            dequeue!(pq)
            if filter_ᾱ(i, 𝔅) == false
                continue
            end

            if i[2] == 1
                𝔅[i[1], :] = [false false]
                𝔉[i[1]] = true
                d′[i[1]] = d[i[1]]
            else
                𝔅[i[1], :] = [𝔩[i]   𝔲[i]]
                𝔉[i[1]] = false
                d′[i[1]] = 0.
            end

            if (length(pq) > 0)
                i′, ᾱ′ = peek(pq)[2]
                if (filter_ᾱ(i′, 𝔅) == false) || ((i′[2] == i[2]) && (ᾱ′ ≃ ᾱ))
                    continue
                end
            end

            x′ = get_x(x, ᾱ*d, 𝔅)
            if count(𝔉) == 0
                return x′
            end
            verba(1, "line_search : $(count(𝔉)) inactive")
            Δα = (Q*d′ |> Qd -> (d′⋅q + x'Qd, Qd'd′))
            if Δα[2] == 0.
                verba(1, "line_search : d⋅Qd = $(Δα)")
            end
            if Δα[1] > 0.
                return x′
            elseif length(pq) == 0 || ᾱ-Δα[1]/Δα[2] ⪝ peek(pq)[2][2]
                return x′ - Δα[1] * d′ / Δα[2]
            end
        end
        return x
    end
    function line_search(P_ᾱs::Array{CartesianIndex{2}, 1}, ᾱs, x, d)

    end

    # Projected Conjugate Gradient with stop when crossing border
    function local_search(x, Q, q, l, u, max_iter, crossstop=true)
        x = get_Πx(x, l, u)
        g = get_Π∇f(x, Q, q, l, u)
        d = -g
        for i in 1:max_iter
            ᾱs = (get_ᾱs(x, d, l, u)[1] |> ᾱs -> ᾱs[filter_ᾱs(ᾱs)])
            ᾱ = length(ᾱs) == 0 ? Inf : minimum(ᾱs)
            Δα = (d'q + d'Q*x, d'Q*d)
            if Δα[1] > 0
                break
            end
            if Δα[2] == 0.
                verba(1, "local_search : d⋅Qd = 0.")
            end
            α = - (d'q + d'Q*x) / (d'Q*d)
            if α ⪝ 0.
                break
            end
            x[:] = get_Πx(x + min(α, ᾱ)*d, l, u)
            if α ⪝ ᾱ
                break
            end

            g′ = get_Π∇f(x, Q, q, l, u)
            β = max(0, g′⋅(g′-g) / g⋅g)
            d[:] = -g′ + β*d
            g = g′
        end
        x
    end

    function step(x, d, Q, q, l, u)
        (ᾱs, 𝔩, 𝔲) = get_ᾱs(x, d, l, u)
        if any(isnan.(ᾱs))
            verba(0, "step : ERROR: got an ᾱ=NaN")
        end
        F_ᾱs = filter_ᾱs(ᾱs)
        pq = priority_ᾱs(F_ᾱs, ᾱs)
        𝔅 = on_box_side(x)
        x = line_search(pq, x, d, Q, q, 𝔩, 𝔲, 𝔅)

        𝔉 = .~(𝔅[:, 1] .| 𝔅[:, 2])
        if any(𝔉)
            x[𝔉] = local_search(x[𝔉], Q[𝔉, 𝔉], q[𝔉] + Q[𝔉, .~𝔉]*x[.~𝔉], l[𝔉], u[𝔉], 50)
        end
        x
    end

    function solve(descent, x, Q, q, l, u)
        if typeof(descent) !== QuadraticBoxPCGDescent
            init!(descent, x -> get_Πf(x, l, u), x -> get_Π∇f(x, Q, q, l, u), x)
        end
        x[:] = get_Πx(x, l, u)
        g = get_∇f(x, Q, q)
        @memento Π∇f = -ΠT!(-g, x, l, u)
        @memento normΠ∇f = norm(Π∇f, Inf)        
        @memento d = -g
        @memento Πd = -Π∇f
        verba(1, "||Π∇f|| : $normΠ∇f")
        for i in 1:max_iter
            if normΠ∇f < ε
                verba(0, "\nIterations: $i\n")
                break
            end

            if typeof(descent) !== QuadraticBoxPCGDescent
                @memento x[:] = get_Πx(step!(descent, x -> get_Πf(x, l, u), x -> get_Π∇f(x, Q, q, l, u), x), l, u)
                @memento Π∇f[:] = get_Π∇f(x, Q, q, l, u)
            else
                @memento x[:] = get_Πx(step(x, d, Q, q, l, u), l, u)
                g′ = get_∇f(x, Q, q)
                @memento Π∇f[:] = -ΠT!(-g′, x, l, u)
                # g[:] = -ΠT!(-g, x, l, u)
                @memento β = g′⋅(g′-g) / g⋅g
                β = max(0. , isnan(β) ? 0. : β)
                @memento d′[:] = -g′ + β*d′
                d[:] = d′
                @memento d[:] = ΠT!(d, x, l, u)
                g[:] = g′
            end
            verba(2, "x : $x")
            verba(2, "Π∇f : $Π∇f")
            @memento normΠ∇f = norm(Π∇f, Inf)
            verba(1, "||Π∇f|| : $normΠ∇f")
        end

        @memento f = get_f(x, Q, q)
        verba(0, "f = $f")
        result = @get_result x Π∇f normΠ∇f f
        OptimizationResult{MQBProblem}(memoria=@get_memoria, result=result)
    end

    solve(descent, x, Q, q, l, u)
end

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
    Optimization.set!(instance, 
        problem=𝔓, 
        algorithm=algorithm, 
        options=MQBPSolverOptions(),
        solver=OptimizationSolver{MQBProblem}())
    return instance
end

export set!, run!, QuadraticBoxPCGDescent, MQBProblem, MQBPSolverOptions, MQBPAlgorithmPG1, generate_quadratic_boxed_problem, get_test
end     # end module MinQuadratic

#   Usage example
# include("minquadratic.jl")   # Or using Revise for tracking changes to files while developing
#   then
# algorithm = MQBPAlgorithmPG1(descent=AdagradDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=0.)
# test = get_test(algorithm, n=10)
# test.solver.options.memoranda = Set(["normΠ∇f"])
#   (or, oneliner)
# algorithm = MQBPAlgorithmPG1(descent=AdagradDescent(), verbosity=1, max_iter=1000, ε=1e-7, ϵ₀=0.); test = get_test(algorithm, n=10); test.solver.options.memoranda = Set(["normΠ∇f"])