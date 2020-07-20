"""
# ---------------------------- Dual algorithm D2 -----------------------------
# Equality and Box Constraints dualised
WIP
## TODO
* adapt to new framework
* implement REAL projected gradient (the present one is not a real projection...)
"""
mutable struct QMCFBPAlgorithmD2D <: OptimizationAlgorithm{QMCFBProblem}
    localization::DescentMethod
    verba               # verbosity utility
    max_iter            # max number of iterations
    ϵₘ                  # error within which an element is considered 0
    ϵ₀                  # error within which a point is on a boundary
    ε                   # precision within which eq. constraint is to be satisfied
    p₀                  # starting point
    cure_singularity    # if true, approach iteratively a singular Q

    QMCFBPAlgorithmD2D() = new()
end
function set!(algorithm::QMCFBPAlgorithmD2D, 𝔓::QMCFBProblem)
end
function run!(algorithm::QMCFBPAlgorithmD2D, 𝔓::QMCFBProblem)
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

        # TODO: Projection to be implemented
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
        while norm(P∇L) > ε
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
