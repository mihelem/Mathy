using LinearAlgebra
using SparseArrays
using Parameters
import Plots

# verbosity utility
function verba(verbosity, level, message)
    if level ≤ verbosity
        println(message)
    end
end
# iteration recorder
# against macro hygiene, it is creating a dictionary called memoranda
macro init_memoria(expr)
    quote
        memoria = Dict{String, AbstractArray}()
        for meme in $expr
            memoria[meme] = []
        end
    end |> esc
end
macro memento(expr)
    if (typeof(expr) === Expr) && (expr.head === :(=))
        l_symbol = expr.args[1]
        while (typeof(l_symbol) === Expr) && (l_symbol.head === :ref)
            l_symbol = l_symbol.args[1]
        end
        quote
            $(expr)
            let meme = string($(Meta.quot(l_symbol)))
                if haskey(memoria, meme)
                    push!(memoria[meme], deepcopy($(l_symbol)))
                end
            end
        end |> esc
    end
end
macro get_memoria()
    :(memoria) |> esc
end
macro get_result(args...)
    result_expr = :(Dict{String, Any}())
    for var in args
        push!(result_expr.args, :($(string(var)) => $var) |> esc)
    end
    result_expr
end

# ----------------------------------------------------------------------- #

include("optimization.jl")
include("descent.jl")
include("numerical.jl")

abstract type MinCostFlowProblem <: OptimizationProblem end
# ------------ (Convex) Quadratic Min Cost Flow Boxed Problem ----------- #
# minₓ { ½xᵀQx + qᵀx  with  x s.t.  Ex = b  &  l ≤ x ≤ u }
# Q ∈ { diag ≥ 0 }
# E node-arc incidence matrix of directed graph
# m : number of nodes
# n : number of arcs
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
function run!(solver::OptimizationSolver{QMCFBProblem}, problem::QMCFBProblem)
    run!(solver.algorithm, problem, memoranda=solver.options.memoranda)
end

# ------------------------ Result of Computation ------------------------ #
mutable struct QMCFBPResult <: OptimizationResult{QMCFBProblem}
    result::Dict{String, Any}
    memoria::Dict{String, Any}
    plots::Dict{String, Plots.plot}

    OptimizationResult{QMCFBProblem}(;memoria=nothing, plots=nothing, result=nothing) = begin
        object = new()
        @some object.memoria = memoria
        @some object.plots = plots
        @some object.result = result
        object
    end
end
function plot!(cur_plot::Plots.Plot, result::QMCFBPResult, meme::String)
    if haskey(result.memoria, meme) === false
        return
    end
    result.memoria[meme] |> (
        data -> Plots.plot!(cur_plot, 1:size(data, 1), data))
end
function plot(result::QMCFBPResult, meme::String)
    if haskey(result.memoria, meme) === false
        return
    end
    result.memoria[meme] |> (
        data -> Plots.plot(1:size(data, 1), data))
end
function set!(result::QMCFBPResult, meme::String, cur_plot::Plots.Plot)
    result.plots[meme] = cur_plot
    result
end
function set!(instance::OptimizationInstance{QMCFBProblem}, meme::String, cur_plot::Plots.Plot)
    set!(instance.result, meme, cur_plot)
    instance
end

# TODO: any exact "descent" (not really, it's a saddle point) method
# --------------------- Primal Dual algorithm PD1 ------------------------- #
mutable struct QMCFBPAlgorithmPD1 <: OptimizationAlgorithm{QMCFBProblem}
    descent::DescentMethod
    verba       # verbosity utility
    max_iter    # max number of iterations
    ϵ₀          # error within which a point is on a boundary
    ε           # precision to which ∇L is considered null
    p₀          # starting point

    memorabilia # set of the name of variables that can be recorded during execution
    QMCFBPAlgorithmPD1(;
        descent=nothing, 
        verbosity=nothing, 
        verba=nothing, 
        max_iter=nothing, 
        ϵ₀=nothing, 
        ε=nothing, 
        p₀=nothing) = begin

        algorithm = new()
        algorithm.memorabilia = Set(["objective", "Π∇L", "∇L", "p", "normΠ∇L", "normΠ∇L_μ"])

        set!(algorithm, descent=descent, verbosity=verbosity, verba=verba, max_iter=max_iter, ϵ₀=ϵ₀, ε=ε, p₀=p₀)
    end
end
# about memorabilia
# names of the variables that can be set to be recorded during execution;
# by now it is a set; in the future it could become a dictionary, since
# to each variable in the mathematical domain we can have many different
# names in the program

function set!(algorithm::QMCFBPAlgorithmPD1; 
    descent=nothing, 
    verbosity=nothing, 
    verba=nothing, 
    max_iter=nothing, 
    ϵ₀=nothing, 
    ε=nothing, 
    p₀=nothing)

    @some algorithm.descent=descent
    if verbosity !== nothing
        algorithm.verba = ((level, message) -> verba(verbosity, level, message))
    end
    @some algorithm.verba=verba
    @some algorithm.max_iter=max_iter
    @some algorithm.ϵ₀=ϵ₀
    @some algorithm.ε=ε
    algorithm.p₀=p₀
end
function run!(algorithm::QMCFBPAlgorithmPD1, 𝔓::QMCFBProblem; memoranda=Set([]))
    @unpack Q, q, l, u, E, b = 𝔓
    @unpack descent, verba, max_iter, ϵ₀, ε, p₀ = algorithm
    @init_memoria memoranda
    
    m, n = size(E)
    p = p₀ === nothing ? [l + u .* rand(n); rand(m)] : p₀
    @views get_x, get_μ = p->p[1:n], p->p[n+1:n+m]
    x, μ = get_x(p), get_μ(p)

    a ≈ b = abs(a-b) ≤ ϵ₀
    a ⪎ b = a+ϵ₀ ≥ b
    a ⪍ b = a ≤ b+ϵ₀

    Π! = p -> (x = get_x(p); x[:] = min.(max.(x, l), u); p)
    Π∇! = (p, ∇L) -> begin
            x, ∇ₓL = get_x(p), get_x(∇L)
            𝔲, 𝔩 = (x .≥ u), (x .≤ l)
            (a -> max(0., a)).(∇ₓL[𝔲])
            (a -> min(0., a)).(∇ₓL[𝔩])
            ∇L
        end

    # using ∇L = (∇_x, -∇_μ) to have descent direction for everyone
    get_∇L = p -> (x=get_x(p); μ=get_μ(p); [Q*x+q+E'μ; -E*x+b])
    get_Π∇L = p -> Π∇!(p, get_∇L(p))
    
    init!(descent, nothing, get_Π∇L, p)
    @memento Π∇L = get_Π∇L(p)
    for i=1:max_iter
        @memento normΠ∇L = norm(Π∇L, Inf)
        verba(1, "||Π∇L|| = $(normΠ∇L)")
        if normΠ∇L < ε
            verba(0, "\n$i iterazioni\n")
            break
        end
        @memento p[:] = Π!(step!(descent, nothing, get_Π∇L, p))
        @memento Π∇L = get_Π∇L(p)
        @memento normΠ∇L_μ = norm(get_μ(Π∇L), Inf)
        @memento objective = 0.5*x'Q*x+q'x
    end 

    normΠ∇L = norm(Π∇L, 2); verba(0, "||Π∇L|| = $(normΠ∇L)")
    verba(0, "||Ex-b|| = $(norm(get_μ(Π∇L), Inf))")
    L = 0.5*x'Q*x+q'x; verba(0, "L = $L")

    # Need a deep copy?
    result = @get_result p Π∇L normΠ∇L L
    OptimizationResult{QMCFBProblem}(memoria=@get_memoria, result=result)
end

# TODO: other line searches and descent methods
# ---------------------------- Dual algorithm ----------------------------- #
# Equality Constraints dualised
mutable struct QMCFBPAlgorithmD1 <: OptimizationAlgorithm{QMCFBProblem}
    descent::DescentMethod
    verba               # verbosity utility
    max_iter            # max number of iterations
    ϵₘ                  # error within which an element is considered 0
    ε                   # precision to which eq. constraint is to be satisfied
    p₀                  # starting point
    cure_singularity    # if true, approach iteratively a singular Q

    memorabilia # set of the name of variables that can be recorded during execution
    QMCFBPAlgorithmD1(;
        descent=nothing, 
        verbosity=nothing, 
        verba=nothing, 
        max_iter=nothing, 
        ϵₘ=nothing, 
        ε=nothing, 
        p₀=nothing,
        cure_singularity=nothing) = begin

        algorithm = new()
        algorithm.memorabilia = Set(["L", "∇L", "norm∇L", "x", "μ", "λ"])

        set!(algorithm, descent=descent, verbosity=verbosity, verba=verba, max_iter=max_iter, ϵₘ=ϵₘ, ε=ε, p₀=p₀, cure_singularity=cure_singularity)
    end

end
function set!(algorithm::QMCFBPAlgorithmD1; 
    descent=nothing, 
    verbosity=nothing, 
    verba=nothing, 
    max_iter=nothing, 
    ϵₘ=nothing, 
    ε=nothing, 
    p₀=nothing,
    cure_singularity=nothing)

    @some algorithm.descent=descent
    if verbosity !== nothing
        algorithm.verba = ((level, message) -> verba(verbosity, level, message))
    end
    @some algorithm.verba=verba
    @some algorithm.max_iter=max_iter
    @some algorithm.ϵₘ=ϵₘ
    @some algorithm.ε=ε
    algorithm.p₀=p₀
    @some algorithm.cure_singularity = cure_singularity
end
function run!(algorithm::QMCFBPAlgorithmD1, 𝔓::QMCFBProblem; memoranda=Set([]))
    @unpack Q, q, l, u, E, b, reduced = 𝔓
    @unpack descent, verba, max_iter, ϵₘ, ε, μ₀, cure_singularity = algorithm
    @init_memoria memoranda

    Q_diag = view(Q, [CartesianIndex(i, i) for i in 1:size(Q, 1)])
    𝔎 = Q_diag .< ϵₘ
    λ_min = minimum(Q_diag[.~𝔎])

    μ = zeros(eltype(Q), size(E, 1)); @some μ[:] = μ₀
    # reduced == true ⟹ assume E represent a connected graph
    if reduced == true
        E, b, μ = E[1:end-1, :], b[1:end-1], μ[1:end-1]
    end
    m, n = size(E)      # m: n° nodes, n: n° arcs

    Q̃ = spzeros(eltype(Q), size(Q, 1), size(Q, 2))
    Q̃_diag = view(Q̃, [CartesianIndex(i, i) for i in 1:size(Q, 1)])
    Q̃_diag[:] = 1. ./ Q_diag

    Ql, Qu = Q*l, Q*u

    # 0 attractor
    to0 = x::AbstractFloat -> (abs(x) ≥ ϵₘ ? x : 0.)
    a::AbstractFloat ≈ b::AbstractFloat = 
        (1+ϵₘ*sign(a))*a ≥ (1-ϵₘ*sign(b))*b   &&   (1+ϵₘ*sign(b))*b ≥ (1-ϵₘ*sign(a))*a

    function get_L(x, μ)
        return 0.5*x'*Q*x + q'*x + μ'*(E*x-b)
    end
    # x̃ = argminₓ L(x, μ) without box constraints
    function get_Qx̃(μ)
        verba(3, "get_Qx̃: Qx̃=$(to0.(-E'*μ-q))")
        return to0.(-E'*μ-q) #(a -> abs(a)>ϵₘ ? a : 0).(-E'*μ-q)
    end
    # ✓
    function get_Qx̃(μ̄, 𝔅)
        return to0.(-E[:, 𝔅[:, 2]]'*μ̄ -q[𝔅[:, 2]]) #   -E[:, 𝔅[:, 2]]'*μ̄ -q[𝔅[:, 2]] #
    end
    # x̅ = argminₓ L(x, μ) beholding box constraints l .<= x .<= u
    function get_x̅(μ)
        return [ maximum([min(u[i], (-μ'*E[:, i]-q[i]) / Q[i, i]), l[i]]) for i=1:n ]
    end
    # mark if x is on a side of the box constraints
    # 1 -> lower  2 -> interior  3 -> upper
    function on_box_side!(Qx̃, 𝔅)
        # add _ϵ maybe 
        𝔅[:, 1] .= (Qx̃ .≤ Ql)
        𝔅[:, 3] .= (Qx̃ .≥ Qu) .& (.~𝔅[:, 1])
        𝔅[:, 2] .= .~(𝔅[:, 1] .| 𝔅[:, 3])
        return 𝔅
    end
    function get_x̅(Qx̃, 𝔅)
        return sum([𝔅[:, 1].*l, 𝔅[:, 2].*(Q̃*Qx̃), 𝔅[:, 3].*u])
    end
    # ∇L with respecto to μ, that is the constraint E*x(μ)-b
    function get_∇L(x)
        return E*x-b
    end
    # ✓
    function get_ᾱs(Qx̃, Eᵀd)
        # 1 : getting inside
        # 2 : going outside
        ᾱs = zeros(eltype(Eᵀd), size(Eᵀd, 1), 2)

        𝔩 = [Eᵀd .< 0  Eᵀd .> 0]        
        ᾱs[𝔩] = ([Qx̃ Qx̃][𝔩] - [Ql Ql][𝔩]) ./ [Eᵀd Eᵀd][𝔩]
        𝔩[𝔩] = 𝔩[𝔩] .& (ᾱs[𝔩] .≥ -100*ϵₘ)

        𝔲 = [Eᵀd .> 0  Eᵀd .< 0]
        ᾱs[𝔲] = ([Qx̃ Qx̃][𝔲] - [Qu Qu][𝔲]) ./ [Eᵀd Eᵀd][𝔲]
        𝔲[𝔲] = 𝔲[𝔲] .& (ᾱs[𝔲] .≥ -100*ϵₘ)

        return (ᾱs, 𝔩, 𝔲)
    end
    # ✓
    function sortperm_ᾱs(ᾱs, 𝔩, 𝔲)
        P = findall(𝔩 .| 𝔲)
        return sort!(P, lt = (i, j) -> begin
            if ᾱs[i] ≈ ᾱs[j]
                (i[2], ᾱs[i], i[1]) < (j[2], ᾱs[j], j[1])
            else
                ᾱs[i] < ᾱs[j]            
            end
        end)
    end
    # ✓
    function exact_line_search!(x, μ, d, 𝔅)
        Eᵀμ, Eᵀd, dᵀb, Qx̃ = E'*μ, E'*d, d'*b, get_Qx̃(μ)
        ᾱs, 𝔩, 𝔲 = get_ᾱs(Qx̃, Eᵀd)
        function filter_inconsistent(P)
            inside = 𝔅[:, 2]
            verba(4, "filter_inconsistent: siamo nelle regioni $(findall(in))")
            verba(4, "filter_inconsistent: unfiltered=$([(p[1], p[2]) for p in P])")
            remove = zeros(Bool, size(P, 1))
            for i in 1:size(P, 1)
                p = P[i]
                if inside[p[1]] == (p[2] == 1)
                    remove[i] = true
                    continue
                end
                inside[p[1]] = (p[2] == 1)
            end
            verba(4, "filter_inconsistent: filtered=$([(p[1], p[2]) for p in P[.~remove]])")
            return P[.~remove]
        end
        P_ᾱs = filter_inconsistent(sortperm_ᾱs(ᾱs, 𝔩, 𝔲))
        verba(3, "exact_line_search: αs=$(ᾱs[P_ᾱs])")

        # x(μ) is NaN when it is not a function, so pick the best representative
        # TODO: find x minimising norm(∇L)
        function resolve_nan!(x)
            𝔫 = isnan.(x)
            if any(𝔫)
                verba(2, "resolve_nan: resolving NaN in x=$x")
                Inc = Eᵀd[𝔫] .> 0
                Dec = Eᵀd[𝔫] .< 0
                Nul = Eᵀd[𝔫] .== 0
                L̂, Û = Inc.*l[𝔫] + Dec.*u[𝔫], Inc.*u[𝔫] + Dec.*l[𝔫]
                S = dᵀb - Eᵀd[.~𝔫]'*x[.~𝔫]
                λ = (S - Eᵀd[𝔫]'*L̂) / (Eᵀd[𝔫]'*(Û-L̂))
                if 0 ≤ λ ≤ 1
                    @memento x[𝔫] = L̂ + λ*(Û - L̂) + Nul.*(l[𝔫]+u[𝔫]) / 2
                    verba(2, "resolve_nan: resolved x=$x")
                    return true
                else
                    @memento x[𝔫] = Nul.*(l[𝔫]+u[𝔫]) / 2 + ((λ > 1) ? Û : L̂)
                    verba(2, "resolve_nan: UNresolved x=$x")
                    return false
                end
            end
            return nothing
        end

        function find_α!(μ, x, α₀, α₁)
            if any(𝔅[:, 2])
                verba(3, "find_α: siamo nelle regioni $(findall(𝔅[:, 2]))")
                Δα = (Eᵀd'*x - dᵀb) / (Eᵀd[𝔅[:, 2]]' * Q̃[𝔅[:, 2], 𝔅[:, 2]] * Eᵀd[𝔅[:, 2]])
                verba(3, "find_α: Δα = $(Δα)")
                if isnan(Δα)
                    Δα = 0
                end
                if 0 ≤ Δα ≤ α₁-α₀
                    @memento μ[:] = μ + (α₀+Δα)*d
                    @memento x[:] = get_x̅(get_Qx̃(μ), 𝔅)
                    verba(3, "find_α: μ=$μ \nfind_α: Qx̃=$(get_Qx̃(μ)) \nfind_α: x=$x")
                    return true
                end
                verba(3, "find_α: Δα is outside of this region")
            end
            return false
        end

        ᾱ, μ̄  = 0., copy(μ)
        j = 1
        last_j = size(P_ᾱs, 1)
        while j ≤ last_j
            μ̄[:] = μ
            i = P_ᾱs[j]
            found_α = find_α!(μ̄, x, ᾱ, ᾱs[i])
            if found_α
                resolved_nan = resolve_nan!(x)
                if resolved_nan === true || resolved_nan === nothing
                    @memento μ[:] = μ̄
                    return
                end
            end

            # set 𝔅 for next ᾱ
            k = j
            while (k ≤ last_j) && (P_ᾱs[k][2] == P_ᾱs[j][2]) && (ᾱs[P_ᾱs[k]] ≈ ᾱs[P_ᾱs[j]])
                verba(4, "exact_line_search: cross border of region $(P_ᾱs[k])")
                verba(4, "exact_line_search: from regions $(findall(𝔅[:, 2]))")
                P_ᾱs[k] |> ii -> begin
                    𝔅[ii[1], :] = (ii[2] == 2) ? [𝔩[ii] false 𝔲[ii]] : [false true false]
                    ᾱ = ᾱs[ii]
                end
                verba(4, "exact_line_search: to regions $(findall(𝔅[:, 2]))")
                k += 1
            end
            verba(4, "exact_line_search: AT THE END OF THE GROUP $(findall(𝔅[:, 2]))")

            j = k
            μ̄[:]  = μ + ᾱ*d
            Qx̃[𝔅[:, 2]] = get_Qx̃(μ̄, 𝔅)
            @memento x[𝔅[:, 2]] = max.(min.(Q̃[𝔅[:, 2], 𝔅[:, 2]]*Qx̃[𝔅[:, 2]], u[𝔅[:, 2]]), l[𝔅[:, 2]])
        end
        μ̄[:] = μ
        found_α = find_α!(μ̄, x, ᾱ, Inf)
        if found_α
            resolved_nan = resolve_nan!(x)
            if resolved_nan === true || resolved_nan === nothing
                @memento μ[:] = μ̄
                return
            end
        end
    end

    function update_λ!(λ, λ′)
        λ = λ′
        Q_diag[𝔎] .= λ
        Q̃_diag[𝔎] .= 1. / λ
        Qu[𝔎], Ql[𝔎] = Q_diag[𝔎] .* u[𝔎], Q_diag[𝔎] .* l[𝔎]
        λ
    end

    function solve(; update_λ! = update_λ!)
        # Reach iteratively the singular Q
        λ = λ_min
        @memento λ = update_λ!(λ, λ/10.)

        Qx̃ = get_Qx̃(μ)
        𝔅 = zeros(Bool, size(E, 2), 3)
        on_box_side!(Qx̃, 𝔅)
        @memento x̅ = get_x̅(Qx̃, 𝔅)

        while any(isnan.(x̅))
            verba(2, "solve: perturbing the starting μ to avoid NaNs")
            μ[:] += ε*(rand(eltype(μ), size(μ, 1))-0.5)
            Qx̃[:] = get_Qx̃(μ)
            𝔅[:, :] = zeros(Bool, size(E, 2), 3)
            on_box_side!(Qx̃, 𝔅)
            x̅[:] = get_x̅(Qx̃, 𝔅)
        end

        @memento ∇L = get_∇L(x̅)
        @memento norm∇L = norm(∇L) 
        @memento L = get_L(x̅, μ)
        verba(1, "solve: |∇L| = $(norm∇L) \nsolve: L = $L\n")
        d = copy(∇L)
        ∇L₀ = copy(∇L)
        # reset = reset₀
        counter = 0
        while (norm∇L ≥ ε) # && (L-L₀ ≥ ε*abs(L))
            if norm∇L < λ
                @memento λ = update_λ!(λ, λ / 1.2)
            end
            exact_line_search!(x̅, μ, d, 𝔅)
            verba(2, "solve: μ=$μ\nsolve: x=$x̅")
            ∇L₀, ∇L = ∇L, get_∇L(x̅)
            @memento norm∇L = norm(∇L)
            verba(4, "solve: dᵀ∇L = $(d'∇L)")
            reset = reset-1
            d[:] = ∇L + d*(∇L'*∇L - ∇L'*∇L₀) / (∇L₀'*∇L₀)
            if d'*∇L < 0
                d[:] = ∇L
            end
            verba(3, "solve: d=$d")
            # d[:] = ∇L
            #d[:] += 1000*ε*(-0.5 .+ rand(size(d, 1)))
            @memento L = get_L(x̅, μ)
            verba(1, "solve: |∇L| = $(norm∇L) \nsolve: L = $L\n")
            counter += 1
            if counter == max_iter
                break
            end
        end

        @memento L = get_L(x̅, μ)
        verba(0, "solve: L = $L")
        verba(0, "\nsolve: $counter iterazioni\n")

        return @get_result x̅ μ L ∇L λ
    end

    return solve(update_λ! = (cure_singularity ? update_λ! : (a, b) -> a)) |>
        (result -> OptimizationResult{QMCFBProblem}(memoria=@get_memoria, result=result))
end

# WIP 
# TODO: adapt to new framework
# TODO: implement REAL projected gradient (the present one cannot work...)
# ---------------------------- Dual algorithm D2 ----------------------------- #
# Equality and Box Constraints dualised
mutable struct QMCFBPAlgorithmD2 <: OptimizationAlgorithm{QMCFBProblem}
    descent::DescentMethod
    verba               # verbosity utility
    max_iter            # max number of iterations
    ϵₘ                  # error within which an element is considered 0
    ϵ₀                  # error within which a point is on a boundary
    ε                   # precision within which eq. constraint is to be satisfied
    p₀                  # starting point
    cure_singularity    # if true, approach iteratively a singular Q 

    QMCFBPAlgorithmD2() = new()
end
function set!(algorithm::QMCFBPAlgorithmD2, 𝔓::QMCFBProblem)
end
function run!(algorithm::QMCFBPAlgorithmD2, 𝔓::QMCFBProblem)
    @unpack Q, q, l, u, E, b, reduced = 𝔓
    @unpack verba, max_iter, ϵₘ, ϵ₀, ε, p₀, cure_singularity = algorithm

    E = eltype(Q).(E)
    m, n = size(E)

    if p₀ === nothing
        p₀[:] = zeros(eltype(Q), 2n+m)
    end
    ν = copy(p₀)

    # partition subspaces corresponding to ker(Q)
    ℭ = [Q[i, i] > ϵₘ for i in 1:n]
    function partition(v)
        return (v[.~ℭ], v[ℭ])
    end
    function partition!(v)
        @views return (v[.~ℭ], v[ℭ])
    end
    n₁ = count(ℭ)
    Q₁ = Q[ℭ, ℭ]
    Q̃₁ = spdiagm(0 => [1.0/Q₁[i, i] for i in 1:n₁])
    (E₀, E₁) = E[:, .~ℭ], E[:, ℭ]
    ((q₀, q₁), (l₀, l₁), (u₀, u₁)) = partition.([q, l, u])
    @views (μ, λᵤ, λₗ) = (ν[1:m], ν[m+1:m+n], ν[m+n+1:m+2n])
    ((λᵤ₀, λᵤ₁), (λₗ₀, λₗ₁)) = partition!.([λᵤ, λₗ])

    # from the singular part of Q we get a linear problem which translates to the equation
    #     λₗ₀ = q₀ + λᵤ₀ + E₀ᵀμ
    # from which we can remove λₗ₀ from the problem, keeping the inequality constraints
    #     λᵤ₀ + E₀ᵀμ + q₀ .≥ 0
    #     λᵤ, λₗ₁ .≥ 0
    get_λₗ₀ = () -> q₀ + E₀'*μ + λᵤ₀
    λₗ₀[:] = get_λₗ₀()
    # hence we have νᵣ which is ν restricted to the free variables
    νᵣ = view(ν, [[i for i in 1:m+n]; (m+n) .+ findall(ℭ)])
    ν₁ = view(ν, [[i for i in 1:m]; m .+ findall(ℭ); (m+n) .+ findall(ℭ)])

    # I am minimizing -L(⁠ν), which is
    # ½(E₁ᵀμ + λᵤ₁ - λₗ₁)ᵀQ̃₁(E₁ᵀμ + λᵤ₁ - λₗ₁)                          ( = ½ν₁ᵀT₁ᵀQ̃₁T₁ν₁ = L₂ ) + 
    # q₁ᵀQ̃₁(E₁ᵀμ + λᵤ₁ - λₗ₁) + bᵀμ + u₁ᵀλᵤ₁ + (u₀-l₀)ᵀλᵤ₀ - l₀ᵀE₀ᵀμ - l₁ᵀλₗ₁     ( = tᵀνᵣ = L₁ ) +
    # ½q₁ᵀQ̃₁q₁ - q₀ᵀl₀                                                                   ( = L₀ )
    L₀ = 0.5q₁'*Q̃₁*q₁ - q₀'*l₀
    ∇L₁ = begin
        t_μ = E₁*Q̃₁*q₁ + b - E₀*l₀
        t_λᵤ = zeros(eltype(t_μ), n)
        t_λᵤ[ℭ] = Q̃₁*q₁ + u₁
        t_λᵤ[.~(ℭ)] = u₀ - l₀
        t_λₗ₁ = -Q̃₁*q₁ - l₁
        [t_μ; t_λᵤ; t_λₗ₁]
    end
    get_L₁ = () -> ∇L₁'*νᵣ
    T₁ = begin
        T = [E₁' spzeros(eltype(Q), n₁, n) (-I)]
        T[:, n .+ findall(ℭ)] = I(n₁)
        T
    end
    ∇∇L₂ = T₁'*Q̃₁*T₁
    get_∇L = () -> ∇L₁ + ∇∇L₂*νᵣ
    get_L₂ = () -> ( T₁*νᵣ |> (a -> 0.5*a'*Q̃₁*a) )
    get_L = () -> L₀ + get_L₁() + get_L₂()
    function get_x()
        x = spzeros(n)
        x[ℭ] = Q̃₁*(-q₁ - E₁'μ - λᵤ₁ + λₗ₁)
        if count(.~ℭ)>0
            # try? approximately active... ϵ_C ?
            λₗ₀ = get_λₗ₀()
            active_λₗ₀ = λₗ₀ .> 0
            x[.~ℭ][active_λₗ₀] .= l[.~ℭ][active_λₗ₀]
            active_λᵤ₀ = λᵤ₀ .> 0
            x[.~ℭ][active_λᵤ₀] .= u[.~ℭ][active_λᵤ₀]
            inactive_i = findall(.~ℭ) |> (P -> [P[i] for i in findall(.~(active_λᵤ₀ .| active_λₗ₀))])
            inactive = spzeros(Bool, n) |> (a -> (for i in inactive_i a[i] = true end; a))
            active = .~inactive

            # left inverse not supported for sparse vectors
            if count(inactive)>0
                x[inactive] =  E[:, inactive] \ Array(b - E[:, active]*x[active])
            end
            # TODO: check the above is satisfying the constraints
        end

        return x
    end
    
    function get_α(d)
        function get_constraints()
            # constraints: E₀ᵀμ + λᵤ₀ + q₀ .≥ 0   &&   λᵣ .≥ 0   =>
            #   α*(E₀ᵀ*d_μ + d_λᵤ₀) .≥ -(E₀ᵀμ + λᵤ₀ + q₀)
            #                α*d_λᵣ .≥ -λᵣ
            M = [E₀'d[1:m] + d[m+1:m+n][.~ℭ]   (-(E₀'μ + λᵤ₀ + q₀))]
            M = cat(M, [d[m+1:end]   (-νᵣ[m+1:end])], dims=1)

            # (𝔲, 𝔩)  : constraints defining an (upper, lower) bound for α
            𝔲, 𝔩 = (M[:, 1] .< 0), (M[:, 1] .> 0)
            C = spzeros(eltype(M), size(M, 1))
            (𝔲 .| 𝔩) |> 𝔠 -> C[𝔠] = M[𝔠, 2] ./ M[𝔠, 1]

            return (𝔩, 𝔲, C)
        end
        function apply_constraints(α, (𝔩, 𝔲, C))
            α_lb, α_ub = maximum([C[𝔩]; -Inf]), minimum([C[𝔲]; Inf])
            #if isnan(α)
                # todo: why?
            #end
            #if α + ϵ_C*abs(α) < α_lb - ϵ_C*abs(α_lb)
            #    println("ERROR: α = $α is less than $α_lb")
            #end
            α = min(max(α, α_lb), α_ub)   
            active_C = zeros(Bool, size(C, 1))
            # leaving a bit of freedom more... shall we do it? 
            α₊, α₋ = α*(1+ϵ₀*sign(α)), α*(1-ϵ₀*sign(α))
            C₊, C₋ = C .* (1 .+ ϵ₀*sign.(C)), C .* (1. .- ϵ₀*sign.(C))
            active_C[𝔲] = ((α₋ .≤ C₊[𝔲]) .& (α₊ .≥ C₋[𝔲]))

            return (α, active_C)
        end
        
        # ∂L = d'*∇∇L₂*(νᵣ + α*d) + d'*∇L₁ => α = -(d'*∇L₁ + d'*∇∇L₂*νᵣ) / (d'*∇∇L₂*d)
        # avoid multiple piping for better readability
        α = d'∇∇L₂ |> (a -> - (d'∇L₁ + a*νᵣ) / (a*d))
        𝔩, 𝔲, C = get_constraints()
        return apply_constraints(α, (𝔩, 𝔲, C))
    end

    function solve_by_proj_conj_grad()
        P∇L = -get_∇L()
        println("|∇L| = $(norm(P∇L))\tL = $(-get_L())")
        d = copy(P∇L)

        # C .≥ 0 || λₗ₀ .≥ 0 | λᵣ .≥ 0 ||
        # ------------------------------ 
        #        || E₀       |    0    ||
        #   ∇C   || [.~ℭ]I   |    I    ||
        #        || 0        |         ||
        # here I'm taking the inward normal since we have feasibility for C .≥ 0
        # (we shouldn't move along this normal)
        ∇C = -[[E₀; (I(n))[:, .~ℭ]; spzeros(eltype(Q), n₁, n-n₁)]  [spzeros(eltype(Q), m, n+n₁); I(n+n₁)]]

        function project!(M, v)
            if size(M, 2) > 0
                for c in eachcol(M)
                    vᵀc = v'c
                    if vᵀc > 0.
                        v[:] = v - c * vᵀc / (c'c)
                    end
                end
            end
        end
        
        counter = 0
        ∇L = copy(P∇L)
        ∇L₀ = copy(∇L)
        νᵣ₀ = copy(νᵣ)
        L = -get_L()
        L₀ = L
        L̄ = L
        while norm(P∇L) > ϵ
            α, active_C = get_α(d)
            νᵣ[:] += α*d

            P∇L[:] = -get_∇L()

            ∇L₀[:] = ∇L
            ∇L[:] = P∇L
            # d[:] = ∇∇L₂*d |> (Md -> P∇L - d * (P∇L'*Md) / (d'*Md))
            # d[:] = (counter & 0) != 0 ? (∇∇L₂*d |> (Md -> P∇L - d * (P∇L'*Md) / (d'*Md))) : P∇L
            d[:] = ∇L + d*(∇L'*∇L - ∇L'*∇L₀) / (∇L₀'*∇L₀)
            
            if d'∇L < 0.
                d[:] = P∇L
            end
            # d[:] = P∇L
            #d[:] = d + norm(d)*rand(eltype(d), size(d, 1))*0.2
            project!(view(∇C, :, active_C), P∇L) 
            project!(view(∇C, :, active_C), d)
            # project d onto the feasible space for νᵣ
            
            println("|P∇L| = $(norm(P∇L))\tL = $(-get_L())")

            counter += 1
            if counter > Inf
                break
            end
        end

        x, ∇L = get_x(), -get_∇L()
        P∇L = copy(∇L)
        α, active_C = get_α(d)
        project!(view(∇C, :, active_C), P∇L)
        println("\nμ = $μ\nx = $x\n∇L = $∇L\nP∇L = $P∇L\nactive_C = $active_C\n\n $counter iterazioni\n")

        λₗ₀[:] = get_λₗ₀()
        return (ν, x)
    end

    return solve_by_proj_conj_grad()
end

# WIP: really, just copy pasted from old commit!
# ---------------------------- Dual algorithm D3 ----------------------------- #
# Null Space method + Box Constraints dualised
mutable struct QMCFBPAlgorithmD3 <: OptimizationAlgorithm{QMCFBProblem}
end
function set!(algorithm::QMCFBPAlgorithmD3, 𝔓::QMCFBProblem)
end
function run!(algo::QMCFBPAlgorithmD3, 𝔓::QMCFBProblem)
    @unpack Q, q, l, u, E, b = 𝔓
    
    # Assumption : m ≤ n
    function split_eq_constraint(ϵ)
        m, n = size(E)
        A = [E b I]
        Pₕ, Pᵥ = [i for i in 1:n], [i for i in 1:m]
        n′ = n
        for i=1:m
            for i′=i:n′
                j = i
                for j′=i:m
                    if abs(A[j′, i′]) > abs(A[j, i′])
                        j = j′
                    end
                end
                if abs(A[j, i′]) > ϵ
                    Pᵥ[i], Pᵥ[j] = Pᵥ[j], Pᵥ[i]
                    A[i, i′:end], A[j, i′:end] = A[j, i′:end], A[i, i′:end]

                    Pₕ[i], Pₕ[i′] = Pₕ[i′], Pₕ[i]
                    A[:, i], A[:, i′] = A[:, i′], A[:, i]
                    A[:, i+1:i′], A[:, (n′+i+1-i′):n′] = A[:, (n′+i+1-i′):n′], A[:, i+1:i′]
                    Pₕ[i+1:i′], Pₕ[(n′+i+1-i′):n′] = Pₕ[(n′+i+1-i′):n′], Pₕ[i+1:i′]

                    n′ = n′+i-i′
                    break
                end
            end
            if abs(A[i, i]) ≤ ϵ
                break
            end

            A[i+1:end, i:end] -=  (A[i+1:end, i] / A[i, i]) .* A[i, i:end]'
        end

        dimension = m
        for i=m:-1:1
            if abs(A[i, i]) ≤ ϵ
                dimension -= 1
                continue
            end
            A[i, i:end] ./= A[i, i]
            A[1:i-1, i:end] -= A[1:i-1, i] .* A[i, i:end]'
        end

        return (dimension, Pᵥ, Pₕ, A)
    end

    dimension, Pᵥ, Pₕ, A = split_eq_constraint(ϵ)
    m, n = dimension, size(E, 2)-dimension

    @views b_B = b[Pᵥ[1:dimension]]
    @views Ẽ_Bb = A[1:dimension, size(E, 2)+1]
    @views Q_B = Q[Pₕ[1:dimension], Pₕ[1:dimension]]
    @views Q_N = Q[Pₕ[dimension+1:end], Pₕ[dimension+1:end]]
    @views Ẽ_BE_N = A[1:dimension, dimension+1:size(E, 2)]
    @views q_B, q_N = q[Pₕ[1:dimension]], q[Pₕ[dimension+1:end]]
    ∇∇L₂ = Ẽ_BE_N'Q_B*Ẽ_BE_N + Q_N
    ∇L₁ = q_N - Ẽ_BE_N' * (q_B + Q_B*Ẽ_Bb)
    L₀ = 0.5 * Ẽ_Bb'Q_B*Ẽ_Bb + q_B'Ẽ_Bb




    function test()
        return split_eq_constraint(ϵ)
    end

    return test()
end

# -------------- Quadratic Min Cost Flow Boxed Problem Generator ---------- #
# TODO: Add custom active constraints %
function generate_quadratic_min_cost_flow_boxed_problem(type, m, n; sing=0)
    Q = spdiagm(0 => [sort(rand(type, n-sing), rev=true); zeros(type, sing)])
    q = rand(eltype(Q), n)
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
    b = E*x
    return QMCFBProblem(Q, q, l, u, E, b, false)
end

function noNaN(V)
    return (x -> isnan(x) ? 0. : x).(V)
end

# ----------- Quadratic Min Cost Flow Boxed Problem - Algorithm Tester ------------- #
function get_test(algorithm::OptimizationAlgorithm{QMCFBProblem};
    singular::Integer=0,
    m::Integer, n::Integer,
    𝔓::Union{Nothing, QMCFBProblem}=nothing,
    should_reduce::Bool=false,
    type::DataType=Float64)

    if 𝔓 === nothing
        𝔓 = generate_quadratic_min_cost_flow_boxed_problem(type, m, n, sing=singular)
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
# Connected components matrix-wise
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