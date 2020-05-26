"""
```julia
QMCFBPAlgorithmD1D <: OptimizationAlgorithm{QMCFBProblem}
```

Descent algorithm applied on the dual lagrangian obtained from equality constraints dualised.
Since `Q` is not stricty positive definite, such dual lagrangian is continuous but not `C1`.
In particular it is piecewise differentiable, so the solution consists in splitting the line search on the
regions identified by edges of discontinuity of the derivative

**TODO**
* Other line searches and descent methods

"""
mutable struct QMCFBPAlgorithmD1D <: OptimizationAlgorithm{QMCFBProblem}
    descent::DescentMethod
    verba               # verbosity utility
    max_iter            # max number of iterations
    ϵₘ                  # error within which an element is considered 0
    ε                   # precision to which eq. constraint is to be satisfied
    μ₀                  # starting point
    cure_singularity    # if true, approach iteratively a singular Q
    plot_steps          # how many points to be used to draw the line search

    memorabilia # set of the name of variables that can be recorded during execution
    QMCFBPAlgorithmD1D(;
        descent=nothing,
        verbosity=nothing,
        my_verba=nothing,
        max_iter=nothing,
        ϵₘ=nothing,
        ε=nothing,
        μ₀=nothing,
        cure_singularity=nothing,
        plot_steps=nothing) = begin

        algorithm = new()
        algorithm.memorabilia = Set(["L", "∇L", "norm∇L", "x", "μ", "λ", "α", "line_plot"])

        set!(algorithm, descent=descent, verbosity=verbosity, my_verba=my_verba, max_iter=max_iter,
            ϵₘ=ϵₘ, ε=ε, μ₀=μ₀, cure_singularity=cure_singularity, plot_steps=plot_steps)
    end

end
function set!(algorithm::QMCFBPAlgorithmD1D;
    descent=nothing,
    verbosity=nothing,
    my_verba=nothing,
    max_iter=nothing,
    ϵₘ=nothing,
    ε=nothing,
    μ₀=nothing,
    cure_singularity=nothing,
    plot_steps=0)

    @some algorithm.descent=descent
    if verbosity !== nothing
        algorithm.verba = ((level, message) -> verba(verbosity, level, message))
    end
    @some algorithm.verba=my_verba
    @some algorithm.max_iter=max_iter
    @some algorithm.ϵₘ=ϵₘ
    @some algorithm.ε=ε
    algorithm.μ₀=μ₀
    @some algorithm.cure_singularity = cure_singularity
    algorithm.plot_steps = plot_steps

    algorithm
end
struct Oᾱ <: Base.Order.Ordering
    simeq
end
import Base.Order.lt
"""
```julia
lt(o::Oᾱ, a::Tuple{AbstractFloat, CartesianIndex{2}}, b::Tuple{AbstractFloat, CartesianIndex{2}})
```

Implements an ordering which should partially obviate to the problems of floating point number errors
occurring while sorting the crossing points of a line search with a set of hyperplanes.
It is based on the idea that, for a compact convex body, there is at most one ingoing and one outgoing crossing point.

**Arguments**
* `o :: Oᾱ` : specific approximate ordering for the ᾱ
* `a :: Tuple{AbstractFloat, CartesianIndex{2}}` :
* `b :: Tuple{AbstractFloat, CartesianIndex{2}}` :

"""
lt(o::Oᾱ, a::Tuple{AbstractFloat, CartesianIndex{2}}, b::Tuple{AbstractFloat, CartesianIndex{2}}) = begin
    o.simeq(a[1], b[1]) ?
        a[2] < b[2] :
        a[1] < b[1]
end
lt(o::Oᾱ,
    a::Tuple{AbstractFloat, Bool, CartesianIndex{2}},
    b::Tuple{AbstractFloat, Bool, CartesianIndex{2}}) = begin
    o.simeq(a[1], b[1]) ?
        a[2:end] < b[2:end] :
        a[1] < b[1]
end
function run!(algorithm::QMCFBPAlgorithmD1D, 𝔓::QMCFBProblem; memoranda=Set([]))
"""
    @unpack Q, q, l, u, E, b, reduced = 𝔓
    @unpack descent, verba, max_iter, ϵₘ, ε, μ₀, cure_singularity, plot_steps = algorithm
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
        return to0.(E*x-b)
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
    function exact_line_search!(x, μ, d, 𝔅, plot_steps=1000)
        Eᵀμ, Eᵀd, dᵀb, Qx̃ = E'μ, to0.(E'd), d'b, get_Qx̃(μ)
        ᾱs, 𝔩, 𝔲 = get_ᾱs(Qx̃, Eᵀd)

        function filter_inconsistent(P)
            inside = 𝔅[:, 2]
            verba(4, "filter_inconsistent: in the regions $(findall(inside))")
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

        ret = nothing
        if plot_steps > 0
            last_α = ᾱs[P_ᾱs[end]]
            step_α = 1.01*last_α/plot_steps
            ret = []
            for α in (-0.01*last_α):step_α:last_α
                push!(ret, get_L(get_x̅(μ+α*d), μ+α*d))
            end
        end

        # x(μ) is NaN when it is not a function, so pick the best representative
        # TODO: find x minimising norm(∇L)
        function resolve_nan!(x)
            𝔫 = isnan.(x)
            if any(𝔫)
                verba(2, "resolve_nan: resolving NaN in x=$x")
                Inc = Eᵀd[𝔫] .> 0.0
                Dec = Eᵀd[𝔫] .< 0.0
                Nul = Eᵀd[𝔫] .== 0.0
                L̂, Û = Inc.*l[𝔫] + Dec.*u[𝔫], Inc.*u[𝔫] + Dec.*l[𝔫]
                S = dᵀb - Eᵀd[.~𝔫]'*x[.~𝔫]
                λ = (S - Eᵀd[𝔫]'*L̂) / (Eᵀd[𝔫]'*(Û-L̂))
                if 0.0 ≤ λ ≤ 1.0
                    if count(𝔫) == 1
                        @memento x[𝔫] = L̂ + λ*(Û - L̂) + Nul.*(l[𝔫]+u[𝔫]) / 2.0
                    else
                        # argmin || E[:, 𝔫]*x[𝔫] + E[:, .~𝔫]*x[.~𝔫]-b ||
                        # ≡ argmin || E₁*x₁ + E₀*x₀ - b ||
                        # ≡ argmin ½x₁'E₁'E₁x₁ + (E₀*x₀-b)'E₁*x₁
                        𝔓₁ = MinQuadratic.MQBProblem(E[:, 𝔫]'E[:, 𝔫], E[:, 𝔫]'*(E[:, .~𝔫]*x[.~𝔫]-b), l[𝔫], u[𝔫])
                        instance = OptimizationInstance{MinQuadratic.MQBProblem}()
                        algorithm = MinQuadratic.MQBPAlgorithmPG1(descent=MinQuadratic.QuadraticBoxPCGDescent(), verbosity=1, max_iter=1000, ε=ε, ϵ₀=1e-12)
                        Optimization.set!(instance,
                            problem=𝔓₁,
                            algorithm=algorithm,
                            options=MinQuadratic.MQBPSolverOptions(),
                            solver=OptimizationSolver{MinQuadratic.MQBProblem}())
                        Optimization.run!(instance)
                        @memento x[𝔫] = instance.result.result["x"]
                    end
                    verba(2, "resolve_nan: resolved x=$x")
                    return true
                else
                    @memento x[𝔫] = Nul.*(l[𝔫]+u[𝔫]) / 2.0 + ((λ > 1.0) ? Û : L̂)
                    verba(2, "resolve_nan: UNresolved x=$x")
                    return false
                end
            end
            return nothing
        end

        function find_α!(μ, x, α₀, α₁)
            if any(𝔅[:, 2])
                verba(3, "find_α: in the regions $(findall(𝔅[:, 2]))")
                Δα = (Eᵀd'x - dᵀb) / (Eᵀd[𝔅[:, 2]]' * Q̃[𝔅[:, 2], 𝔅[:, 2]] * Eᵀd[𝔅[:, 2]])
                verba(1, "find_α: Δα = $(Δα)")
                if isnan(Δα)
                    Δα = 0.
                end
                if to0(Δα) == 0.
                    @memento α = α₀
                    @memento μ[:] = μ + α*d
                    @memento x[:] = x
                    verba(3, "find_α: μ=$μ \nfind_α: Qx̃=$(get_Qx̃(μ)) \nfind_α: x=$x")
                    return true
                elseif 0 ≤ Δα ≤ α₁-α₀
                    @memento α = α₀+Δα
                    @memento μ[:] = μ + α*d
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
                    return ret
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
                return ret
            end
        end
    end

    function simeq(a::AbstractFloat, b::AbstractFloat, ϵ=0.0)
        abs(a-b) ≤ ϵ
    end
    function simless(a::AbstractFloat, b::AbstractFloat, ϵ=0.0)
        a < b + ϵ
    end
    function simleq(a::AbstractFloat, b::AbstractFloat, ϵ=0.0)
        a ≤ b + ϵ
    end

    """
    ```julia
    get_Δ∇(x, μ, Eᵀd, Q╲, q, E, l, u) → Δ∇
    ```
    """
    **Result**
    * `Δ∇[:, 1]` : ingoing crossing point of the line search
    * `Δ∇[:, 2]` : outgoing crossing point of the line search
    """
    """
    function get_Δ∇(Eᵀμ, Eᵀd, Q╲, q, l, u)
        Δ∇ = ((- q - Eᵀμ) |> (v -> [v v]))
        (Eᵀd .> 0.0) |> (inc -> Δ∇[inc inc] -= [Q╲[inc].*u[inc]  Q╲[inc].*l[inc]])
        (Eᵀd .< 0.0) |> (dec -> Δ∇[dec dec] -= [Q╲[dec].*l[dec]  Q╲[dec].*u[dec]])
        Δ∇
    end
    """
    """
    **Return**
    * `pq₊` : priority queue with ᾱ met in the increasing part of the line search
    * `pq₋` : priority queue with ᾱ met in the decreasing part of the line search
    * `ΔQx̃` :


    **Logic**
    Line search is split in an forward (ℙ₊) and backward search (ℙ₋), where the
    backward search may be needed only because of approximations.
    * `Qx̃` : would be the optimal `Qx(μ)` with no box constraints
    * inward bitmap: ⋈ = [𝔅[:, 1]  𝔅[:, 3]]
    * `Eᵀd > 0` ⟹ `Qx̃(α)` decreasing ⟹ `u ∈ ℙ₊` if inward, `l ∈ ℙ₊` if outward
    *
    **Assumptions**

    Eᵀd .== 0.0 has been filtered out ⟹ (Eᵀd .< 0.0) == .~(Eᵀd .> 0.0)
    """
    """
    function get_priority_ΔQx̃(Eᵀμ, Eᵀd, q, 𝔅, Ql, Qu; ϵ=ϵₘ)
        Qx̃ = -Eᵀμ-q
        ΔQx̃ = [Qx̃-Ql  Qx̃-Qu]

        @views inward = 𝔅[:, [1, 3]]
        ℙ₊ = (Eᵀd .< 0.0) |> inc -> ([inc .~inc] .== inward)
        ᾱs = ΔQx̃ ./ [Eᵀd Eᵀd]
        make_pq =
            P -> PriorityQueue(
                zip(P, zip(ᾱs[P], .~inward[P], P)),
                Oᾱ((a, b) -> simeq(a, b, ϵ)))
        pq₊ = findall(ℙ₊) |> make_pq
        pq₋ = findall(.~ℙ) |> make_pq
        (pq₊, pq₋, ΔQx̃)
    end
    function in_box(Qx, Ql, Qu; ϵ=ϵₘ)
        (simleq.(-Qx, -Ql, ϵ), simleq.(Qx, Qu, ϵ)) |>
            ((𝔏, 𝔘) -> [𝔏  .~(𝔏 .| 𝔘)  𝔘])
    end
    function in_box(Eᵀμ, Ql, Qu, q; ϵ=ϵₘ)
        (-q-Eᵀμ) |>
            (Qx -> in_box(Qx, Ql, Qu; ϵ))
    end
    function filter_ᾱ(p::CartesianIndex{2}, 𝔅)
        return (p[2] == 1) == 𝔅[p[1]]
    end
    function best_∂()

    end
    function locate_null_∂!(x, l, u, Eᵀd, nanny)
        Inc = Eᵀd[nanny] .> 0.0
        Dec = Eᵀd[nanny] .< 0.0
        Nul = Eᵀd[nanny] .== 0.0
        L̂, Û = Inc.*l[nanny] + Dec.*u[nanny], Inc.*u[nanny] + Dec.*l[nanny]
        S = dᵀb - Eᵀd[.~nanny]'*x[.~nanny]
        λ = (S - Eᵀd[nanny]'*L̂) / (Eᵀd[nanny]'*(Û-L̂))
        if 0.0 ≤ λ ≤ 1.0
            @memento x[nanny] = L̂ + λ*(Û - L̂) + Nul.*(l[nanny]+u[nanny]) / 2.0
            return true
        else
            @memento x[nanny] = Nul.*(l[nanny]+u[nanny]) / 2.0 + ((λ > 1.0) ? Û : L̂)
            return false
        end
    end
    function line_search′(dᵀ∇L, pq, x, x′, μ, Q̃╲, Eᵀd, 𝔅, kerny)
        @views 𝔏, 𝕴, 𝔘 = 𝔅[:, 1], 𝔅[:, 2], 𝔅[:, 3]
        while length(pq) > 0
            Δα = [dᵀ∇L, Eᵀd[𝔅[:, 2]]'Q̃╲*Eᵀd]
            # TODO: check the first ᾱ which could be < 0.0
            if Δα[1] < 0.0
                return (x′, μ)
            end
            if any(𝕴 .& kerny)

            end

        end
    end
    function priority_ᾱs(ᾱs, ϵ)
        ℙ₊ = (ᾱs .≥ 0.0)
        ℙ₋ = .~ℙ₊
        inf_p₊ = (len(ᾱs)+1 |> n -> CartesianIndex(n, 2))
        inf_p₋ =
        ᾱ₋ = maximum[ᾱs[.~ℙ₊]; -Inf]
        P = [p for p in findall(ᾱs) if simless(0.0, ᾱs[p], ϵ)]

        [P; last_p] |>
        (P′ -> PriorityQueue(
            zip(P′, zip([ᾱs[P]; Inf], P′)),
            Oᾱ((a, b) -> simeq(a, b, ϵ)))) |>
        pq -> (ᾱ₋, pq)
    end
    function step′(d, x, μ, Q╲, q, E, b, kerny; ϵ=ϵₘ, ϵₘ=ϵₘ)
        Eᵀμ = E'μ
        𝔅 = in_box(Eᵀμ, Ql, Qu, q)
        𝔐μ = .~simeq.(d / norm(d, Inf), 0.0, ϵ)
        Eᵀd  = E[𝔐μ, :]'d[𝔐μ]
        𝔐x = .~simeq.(Eᵀd / norm(Eᵀd, Inf), 0.0, ϵ)

        x′, Eᵀμ′, Eᵀd′, Q╲′, q′, E′, l′, u′, b′, 𝔅′, kerny′ =
            x[𝔐x], Eᵀμ[𝔐x], Eᵀd[𝔐x], Q╲[𝔐x], q[𝔐x], E[:, 𝔐x], l[𝔐x], u[𝔐x],
            b[𝔐μ], 𝔅[𝔐x, :], kerny[𝔐x]
        Δ∇′ = get_Δ∇(Eᵀμ′, Eᵀd′, Q╲′, q′, l′, u′)
        ᾱs = Δ∇′ ./ Eᵀd′
        pq = priority_ᾱs(ᾱs, ϵₘ)
        dᵀ∇L = Eᵀd′'*x′ - d′'b′
        x′, μ′, 𝔅′ = line_search′(dᵀ∇L, pq, x, x′, μ′, Q╲′, Eᵀd′, 𝔅′, kerny′)
        if any(𝔅′[:, 2] .& kerny)

        end
    end
    # TODO: ϵ₀
    function solve′(μ, Q╲, Q̃╲, q, E, b; max_iter=max_iter, ε=ε, ϵ₀=ϵₘ*ϵₘ, ϵₘ=ϵₘ)
        Ql, Qu = Q╲.*l, Q╲.*u
        kerny = simeq.(Q╲, 0.0, ϵ₀)

        function get_Qx(μ)
            Qx = min.(max.(-E'μ-q, Ql), Qu)
            nanny = zeros(Bool, length(kerny))
            nanny[kerny] = simeq.(Qx[kerny], 0.0, ϵₘ)
            (Qx, nanny)
        end

        x = Q̃╲ .* Qx
        x[nanny] = 0.5*(l[nanny]+u[nanny])

        ∇L = E*x-b
        for i in 1:max_iter
            if norm(∇L, Inf) ≤ ε
                break
            end

            x, μ = step′(...)
            x′ = get_Qx(μ)
            if norm(x′[.~nanny]-x[.~nanny], Inf) > ε
                println("ATTENZIONE")       # TODO
                break
            else
                x[.~nanny] = x′[.~nanny]
            end
            ∇L = E*x-b
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
        counter = 0
        while (norm∇L ≥ ε) # && (L-L₀ ≥ ε*abs(L))
            if norm∇L < λ
                @memento λ = update_λ!(λ, λ / 1.2)
            end
            @memento line_plot = exact_line_search!(x̅, μ, d, 𝔅)
            verba(2, "solve: μ=$μ\nsolve: x=$x̅")
            ∇L₀, ∇L = ∇L, get_∇L(x̅)
            @memento norm∇L = norm(∇L)
            verba(4, "solve: dᵀ∇L = $(d'∇L)")
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
        verba(0, "\nsolve: $counter iterations\n")

        return @get_result x̅ μ L ∇L λ
    end

    return solve(update_λ! = (cure_singularity ? update_λ! : (a, b) -> a)) |>
        (result -> OptimizationResult{QMCFBProblem}(memoria=@get_memoria, result=result))
"""
end
