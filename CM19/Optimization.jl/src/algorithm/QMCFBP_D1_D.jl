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
        algorithm.memorabilia = Set(["L", "∂L", "norm∂L", "x", "μ", "α"])

        set!(algorithm,
            descent=descent,
            verbosity=verbosity,
            my_verba=my_verba,
            max_iter=max_iter,
            ϵₘ=ϵₘ,
            ε=ε,
            μ₀=μ₀,
            cure_singularity=cure_singularity,
            plot_steps=plot_steps)
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
    less
    Oᾱ(simeq) = new(simeq, (a, b) -> a < b)
    Oᾱ(simeq, less) = new(simeq, less)
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
        o.less(a[1], b[1])
end
lt(o::Oᾱ,
    a::Tuple{AbstractFloat, Bool, CartesianIndex{2}},
    b::Tuple{AbstractFloat, Bool, CartesianIndex{2}}) = begin
    o.simeq(a[1], b[1]) ?
        a[2:end] < b[2:end] :
        o.less(a[1], b[1])
end
function run!(algorithm::QMCFBPAlgorithmD1D, 𝔓::QMCFBProblem; memoranda=Set([]))
    @unpack Q, q, l, u, E, b, reduced = 𝔓
    @unpack descent, verba, max_iter, ϵₘ, ε, μ₀, cure_singularity, plot_steps = algorithm
    @init_memoria memoranda

    Q╲ = view(Q, [CartesianIndex(i, i) for i in 1:size(Q, 1)])
    μ = zeros(eltype(Q), size(E, 1)); @some μ[:] = μ₀
    # reduced == true ⟹ assume E represent a connected graph
    if reduced == true
        E, b, μ = E[1:end-1, :], b[1:end-1], μ[1:end-1]
    end
    Ql, Qu = Q*l, Q*u

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
    **Return**
    * `pq₊` : priority queue with ᾱ met in the increasing part of the line search
    * `pq₋` : priority queue with ᾱ met in the decreasing part of the line search
    * `ΔQx̃` :


    **Logic**
    Line search is split in an forward (ℙ₊) and backward search (ℙ₋), where the
    backward search may be needed only because of approximations.
    * `Qx̃` : would be the optimal `Qx(μ)` with no box constraints
    * inward bitmap: inward = [𝔅[:, 1]  𝔅[:, 3]]
    * `Eᵀd > 0` ⟹ `Qx̃(α)` decreasing ⟹ `u ∈ ℙ₊` if inward, `l ∈ ℙ₊` if outward
    *
    **Assumptions**

    Eᵀd .== 0.0 has been filtered out ⟹ (Eᵀd .< 0.0) == .~(Eᵀd .> 0.0)
    """
    function get_priority_ΔQx̃(Eᵀμ, Eᵀd, q, 𝔅, Ql, Qu; ϵ=ϵₘ)
        Qx̃ = -Eᵀμ-q
        ΔQx̃ = [Qx̃-Ql  Qx̃-Qu]

        ΔQx̃[:, 1] .< 0.0
        @views inward = 𝔅[:, [1, 3]]
        ℙ₊ = (Eᵀd .< 0.0) |> inc -> ([inc .~inc] .== inward)
        ᾱs = ΔQx̃ ./ [Eᵀd Eᵀd]
        make_pq =
            Ps -> begin
                (P, s) = Ps
                PriorityQueue(
                    zip([P; CartesianIndex(0, 0)],
                        zip(
                            [ᾱs[P]; s*Inf],
                            [.~inward[P]; false],
                            [P; CartesianIndex(0, 0)])),
                    Oᾱ((a, b) -> simeq(a, b, ϵ), (a, b) -> s*a < s*b))
            end
        pq₊ = (findall(ℙ₊), 1) |> make_pq
        pq₋ = (findall(.~ℙ₊), -1) |> make_pq
        (pq₊, pq₋, ΔQx̃)
    end
    function in_box(Qx, Ql, Qu; ϵ=ϵₘ)
        (simleq.(Qx, Ql, ϵ), simleq.(-Qx, -Qu, ϵ)) |>
            LU -> ((𝔏, 𝔘) = LU; [𝔏  .~(𝔏 .| 𝔘)  𝔘])
    end
    function in_box(Eᵀμ, Ql, Qu, q; ϵ=ϵₘ)
        (-q-Eᵀμ) |>
            (Qx -> in_box(Qx, Ql, Qu, ϵ=ϵ))
    end
    function filter_ᾱ(p::CartesianIndex{2}, outward, 𝔅)
        (p[1] == 0) || (𝔅[p[1], [2, [1, 3][p[2]]]] == [outward, !outward])
    end
    function best_primal_∂!(x, nanny, E, b, l, u)
        # argmin || E[:, 𝔫]*x[𝔫] + E[:, .~𝔫]*x[.~𝔫] - b ||
        # ≡ argmin || E₁*x₁ + E₀*x₀ - b ||
        # ≡ argmin ½x₁'E₁'E₁x₁ + (E₀*x₀-b)'E₁*x₁
        𝔓₁ = MinQuadratic.MQBProblem(
            E[:, nanny]'E[:, nanny],
            E[:, nanny]'*(E[:, .~nanny]*x[.~nanny]-b),
            l[nanny],
            u[nanny])
        instance = OptimizationInstance{MinQuadratic.MQBProblem}()
        algorithm = MinQuadratic.MQBPAlgorithmPG1(
            descent=MinQuadratic.QuadraticBoxPCGDescent(),
            verbosity=1,
            max_iter=1000,
            ε=ε,
            ϵ₀=1e-12)
        Optimization.set!(instance,
            problem=𝔓₁,
            algorithm=algorithm,
            options=MinQuadratic.MQBPSolverOptions(),
            solver=OptimizationSolver{MinQuadratic.MQBProblem}())
        Optimization.run!(instance)
        x[nanny] = instance.result.result["x"]
        x
    end
    function is_primal_null_∂(l, u, Eᵀd, dᵀ∇L)
        Inc, Nul, Dec = (Eᵀd .> 0.0, Eᵀd .== 0.0, Eᵀd .< 0.0)
        L̂, Û = Inc.*l + Dec.*u, Inc.*u + Dec.*l
        S = -dᵀ∇L
        return Eᵀd'L̂ ≤ S ≤ Eᵀd'Û
    end
    function line_search′(pq₊, pq₋, μ, 𝔅, Q╲, kerny, q, Eᵀd, bᵀd, Eᵀμ, l, u)
        # bitmap  :  𝔏 ⟹ x is l, 𝕴 ⟹ x is in the box, 𝔘 ⟹ x is u
        @views 𝔏, 𝕴, 𝔘 = 𝔅[:, 1], 𝔅[:, 2], 𝔅[:, 3]
        Qx̃₀ = -q - Eᵀμ

        # x̃₀[kerny] set to 0 so that it is easier to set in locate_primal_null_∂
        x̃₀ = (Qx̃₀ ./ Q╲) |> x̃ -> (x̃[kerny] .= 0.0; x̃)
        # the next one is costly but stablier than summing each Δ at each ᾱ
        get_dᵀ∇L₀ = () -> Eᵀd'*(𝔏.*l + 𝕴.*x̃₀ + 𝔘.*u) - bᵀd
        get_α_frac = () -> [get_dᵀ∇L₀(), Eᵀd[𝕴]'*(Eᵀd[𝕴]./Q╲[𝕴])]

        α_frac = get_α_frac()

        s = sign(α_frac[1])
        if (s == 0)
            return (0.0, 𝔅)
        end
        pq = s>0 ? pq₊ : pq₋
        ᾱ, outward, p = 0.0, false, [0,0]
        while length(pq) > 0
            next_ᾱ, next_outward, next_p = peek(pq)[2];
            verba(1, "\nnext_ᾱ = $next_ᾱ")
            if filter_ᾱ(next_p, next_outward, 𝔅) == false
                println("ATTENZIONE: filtrato ᾱ")
                continue
            end
            if !(pq.o.simeq(ᾱ, next_ᾱ) && (next_outward == outward))
                nanny = 𝕴 .& kerny
                if any(nanny)
                    if is_primal_null_∂(l[nanny], u[nanny], Eᵀd[nanny], get_dᵀ∇L₀())
                        return (ᾱ, 𝔅)
                    end
                else
                    α_frac[:] = get_α_frac()
                    print("α_frac = $(α_frac)")
                    α = α_frac[1] / α_frac[2]
                    println(" ::  would like α = $α")
                    if (s*(α-ᾱ) |> a -> (a ≤ 0.0 || isnan(a)))
                        return (ᾱ, 𝔅)
                    end
                    if s*(next_ᾱ-α) ≥ 0.0
                        return (α, 𝔅)
                    end
                end
            end

            ᾱ, outward, p = next_ᾱ, next_outward, next_p
            i, lu = p[1], p[2]
            if i*lu == 0
                println("Attenzione, raggiunto ∞")
                error()
            end
            𝔅[i, [2, [1, 3][lu]]] = [!outward, outward]
            # α_frac[1] -= (2outward-1)*Eᵀd[i]*([l[i], u[i]][lu] - x̃₀[i])
            # α_frac[2] = Eᵀd[𝕴]'Q̃╲[𝕴].*Eᵀd[𝕴] # stablier than just adding summand
            dequeue!(pq)
        end
    end
    function step′(d, x, μ, Q╲, q, E, b, kerny; ϵ=ϵₘ, ϵₘ=ϵₘ)
        Eᵀμ = E'μ
        𝔅 = in_box(Eᵀμ, Ql, Qu, q, ϵ=-ϵₘ)
        println("before line search 𝔅 : $𝔅")
        𝔐μ = .~simeq.(d / norm(d, Inf), 0.0, ϵ)
        Eᵀd, bᵀd = E[𝔐μ, :]'d[𝔐μ], b[𝔐μ]'d[𝔐μ]
        𝔐x = .~simeq.(Eᵀd / norm(Eᵀd, Inf), 0.0, ϵ)

        x′, Eᵀμ′, Eᵀd′, Q╲′, q′, E′, l′, u′, 𝔅′, kerny′, Ql′, Qu′ =
            x[𝔐x], Eᵀμ[𝔐x], Eᵀd[𝔐x], Q╲[𝔐x], q[𝔐x], E[:, 𝔐x], l[𝔐x], u[𝔐x],
            𝔅[𝔐x, :], kerny[𝔐x], Ql[𝔐x], Qu[𝔐x]
        pq₊, pq₋, ΔQx̃ = get_priority_ΔQx̃(Eᵀμ′, Eᵀd′, q′, 𝔅′, Ql′, Qu′; ϵ=ϵₘ)

        dᵀ∇L = Eᵀd′'*x′ - bᵀd
        println("dᵀ∇L = $dᵀ∇L")

        α, next_𝔅′ =
            line_search′(pq₊, pq₋, μ, 𝔅′, Q╲′, kerny′, q′, Eᵀd′, bᵀd, Eᵀμ′, l′, u′)

        next_μ = μ + α*d.*𝔐μ
        next_𝔅 = copy(𝔅)
        next_𝔅[𝔐x, :] = next_𝔅′
        println("next_𝔅 : $next_𝔅")
        next_x = copy(x)
        next_x[𝔐x] = min.(max.((-Eᵀμ′-α*Eᵀd′-q′)./ Q╲′, l′), u′) 
        nanny = next_𝔅[:, 2] .& kerny
        if any(nanny)
            best_primal_∂!(next_x, nanny, E, b, l, u)
        end
        return next_x, next_μ, next_𝔅
    end
    # TODO: ϵ₀
    function solve′(μ, Q╲, q, E, b; max_iter=max_iter, ε=ε, ϵ₀=ϵₘ*ϵₘ, ϵₘ=ϵₘ)
        Ql, Qu = Q╲.*l, Q╲.*u
        kerny = simeq.(Q╲, 0.0, ϵ₀)

        function get_Qx̃(μ)
            Qx̃ = -E'μ-q
            nanny = zeros(Bool, length(Qx̃))
            nanny[kerny] = simeq.(Qx̃[kerny], 0.0, ϵₘ)
            (Qx̃, nanny)
        end

        Qx̃, nanny = get_Qx̃(μ)
        x = max.(min.(Qx̃ ./ Q╲, u), l)
        x[nanny] = 0.5*(l[nanny]+u[nanny])

        ∂L = E*x-b
        ∂L₀ = copy(∂L)
        d = copy(∂L)
        for i in 1:max_iter
            @memento L = x'*(0.5*Q╲.*x + q) + μ'*∂L
            @memento norm∂L = norm(∂L, Inf)
            verba(1, "norm∂L : $(norm∂L)")
            if norm∂L ≤ ε
                break
            end

            x[:], μ[:], 𝔅 = step′(d, x, μ, Q╲, q, E, b, kerny, ϵ=ϵₘ, ϵₘ=ϵₘ)
            println("\nx : $x")
            # TODO: better @memento
            ∂L₀[:], ∂L[:] = ∂L, E*x-b
            β = max((∂L'*∂L - ∂L'*∂L₀) / (∂L₀'*∂L₀), 0.0)
            println("β : $β")
            d[:] = ∂L + β*d
        end

        return @get_result x μ ∂L
    end

    return solve′(μ, Q╲, q, E, b) |>
        (result -> OptimizationResult{QMCFBProblem}(memoria=@get_memoria, result=result))
end
