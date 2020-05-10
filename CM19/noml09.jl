using LinearAlgebra
using SparseArrays

# Zeroth Order Methods

# First Order Methods
abstract type DescentMethod end

struct GradientDescent <: DescentMethod
    α
end
init!(M::GradientDescent, f, ∇f, x) = M
function step!(M::GradientDescent, f, ∇f, x)
    α, g = M.α, ∇f(x)
    return x - α*g
end

mutable struct ConjugateGradientDescent <: DescentMethod
    d
    g
end
function init!(M::ConjugateGradientDescent, f, ∇f, x)
    M.g = ∇f(x)
    M.d = -M.g
    return M
end
# mmh...
function step!(M::ConjugateGradientDescent, f, ∇f, x, line_search)
    d, g = M.d, M.g
    g′= ∇f(x)

    # If we knew an approximating quadratics, it would be
    # β = (g′⋅∇∇f*d) / (d⋅∇∇f*d)
    # Fletcher-Reeves
    # β = g′⋅g′ / g⋅g
    # Polak-Ribière
    β = max(0, g′⋅(g′-g) / g⋅g)
    d′ = -g′ + β*d
    x′ = line_search(f, x, d′)
    M.d, M.g = d′, g′
    return x′
end

mutable struct MomentumDescent <: DescentMethod
    α   # learning rate
    β   # momentum decay
    v   # momentum
end
function init!(M::MomentumDescent, f, ∇f, x)
    M.v = zeros(length(x))
    return M
end
function step!(M::MomentumDescent, f, ∇f, x)
    α, β, v, g = M.α, M.β, M.v, ∇f(x)
    v[:] = β*v - α*g
    return x+v
end

mutable struct NesterovMomentumDescent <: DescentMethod
    α # learning rate
    β # momentum decay
    v # momentum
end
function init!(M::NesterovMomentumDescent, f, ∇f, x)
    α, β, v = M.α, M.β, M.v
    v[:] = β*v - α*∇f(x + β*v)
    return x+v
end

mutable struct AdagradDescent <: DescentMethod
    α # learning rate
    ϵ # small value
    s # sum of squared gradient 
end
function init!(M::AdagradDescent, f, ∇f, x)
    M.s = zeros(length(x))
    return M
end
function step!(M::AdagradDescent, f, ∇f, x)
    α, ϵ, s, g = M.α, M.ϵ, M.s, ∇f(x)
    s[:] += g .* g
    return x - α*g ./ (sqrt.(s) .+ ϵ)
end

mutable struct RMSPropDescent <: DescentMethod
    α # learning rate
    γ # decay
    ϵ # small value
    s # sum of squared gradient
end
function init!(M::RMSPropDescent, f, ∇f, x)
    M.s = zeros(length(x))
    return M
end
function step!(M::RMSPropDescent, f, ∇f, x)
    α, γ, ϵ, s, g = M.α, M.γ, M.ϵ, M.s, ∇f(x)
    s[:] = γ*s + (1-γ)*(g .* g)
    return x - α*g ./ (sqrt.(s) .+ ϵ)
end

mutable struct AdadeltaDescent <: DescentMethod
    γs # gradient decay
    γx # update decay
    ϵ # small value
    s # sum of squared gradients
    u # sum od squared gradients
end
function init!(M::AdadeltaDescent, f, ∇f, x)
    M.s = zeros(length(x))
    M.u = zeros(length(x))
    return M
end
function step!(M::AdadeltaDescent, f, ∇f, x)
    γs, γx, ϵ, s, u, g = M.γs, M.γx, M.ϵ, M.s, M.u, ∇f(x)
    s[:] = γs*s + (1-γs)*g.*g
    Δx = - (sqrt.(u) .+ ϵ) ./ (sqrt.(s) .+ ϵ) .* g
    u[:] = γx*u + (1-γx)*Δx.*Δx
    return x+Δx
end

mutable struct AdamDescent <: DescentMethod
    α # learning rate
    γv # decay
    γs # decay
    ϵ # small value
    k # step counter
    v # 1st moment estimate
    s # 2nd moment estimate
end
function init!(M::AdamDescent, f, ∇f, x)
    M.k = 0
    M.v = zeros(length(x))
    M.s = zeros(length(x))
    return M
end
function step!(M::AdamDescent, f, ∇f, x)
    α, γv, γs, ϵ, k = M.α, M.γv, M.γs, M.ϵ, M.k
    s, v, g = M.s, M.v, ∇f(x)
    v[:] = γv*v + (1-γv)*g
    s[:] = γs*s + (1-γs)*g.*g
    M.k = k += 1
    v̂ = v ./ (1 - γv^k)
    ŝ = s ./ (1 - γs^k)
    return x - α*v̂ ./ (sqrt.(ŝ) .+ ϵ)
end

mutable struct HyperGradientDescent <: DescentMethod
    α₀ # initial learning rate
    µ # learning rate of the learning rate
    α # current learning rate
    g_prev # previous gradient
end
function init!(M::HyperGradientDescent, f, ∇f, x)
    M.α = M.α₀
    M.g_prev = zeros(length(x))
    return M
end
function step!(M::HyperGradientDescent, f, ∇f, x)
    α, µ, g, g_prev = M.α, M.µ, ∇f(x), M.g_prev
    α = α + µ*(g⋅g_prev)
    M.g_prev, M.α = g, α
    return x - α*g
end

mutable struct HyperNesterovMomentumDescent <: DescentMethod
    α₀ # initila learning rate
    µ # learning rate of the learning rate
    β # momentum decay
    v # momentum
    α # current learning rate
    g_prev # previous gradient
end
function init!(M::HyperNesterovMomentumDescent, f, ∇f, x)
    M.α = M.α₀
    M.v = zeros(length(x))
    M.g_prev = zeros(length(x))
    return M
end
function step!(M::HyperNesterovMomentumDescent, f, ∇f, x)
    α, β, µ = M.α, M.β, M.µ
    v, g, g_prev = M.v, ∇f(x), M.g_prev
    α = α - µ*(g⋅(-g_prev - β*v))
    v[:] = β*v + g
    M.g_prev, M.α = g, α
    return x - α*(g + β*v)
end

mutable struct NoisyDescent <: DescentMethod
    submethod
    σ
    k
end
function init!(M::NoisyDescent, f, ∇f, x)
    init!(M.submethod, f, ∇f, x)
    M.k = 1
    return M
end
function step!(M::NoisyDescent,f, ∇f, x)
    x = step!(M.submethod, f, ∇f, x)
    σ = M.σ(M.k)
    x += σ.*randn(length(x))
    M.k += 1
    return x
end


# usage:
# 𝔓 = generate_quadratic_min_cost_flow_boxed_problem(Float64, 10, 20)
# L, x, μ, ∇L = solve_quadratic_min_flow(𝔓=𝔓, μ=zeros(Float64, 10), ϵ=1e-6, ε=1e-6)

# Problem
# minₓ { ½xᵀQx + qᵀx  with  x s.t.  Ex = b  &  l ≤ x ≤ u }
# Q ∈ { diag ≥ 0 }
struct quadratic_min_cost_flow_boxed_problem
    Q
    q
    l
    u
    E
    b
end

# m : number of vertices
# n : number of edges
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
    return quadratic_min_cost_flow_boxed_problem(Q, q, l, u, E, b)
end

function noNaN(V)
    return (x -> isnan(x) ? 0. : x).(V)
end

function get_gaussian_pivoted_and_apply(E, M)
    m, n = size(E)
    @views A = [E M]


end

# WIP
# Another Dual
# Equality constrained absorbed by the nullspace method
# dualising box constriants
function solve_quadratic_min_flow_d(; 𝔓, λ, ϵ)
    (Q, q, l, u, E, b) = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b)
    
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
    ∇L₁ = q_N - Ẽ_BE_N'(q_B + Q_B*Ẽ_Bb)
    L₀ = 0.5 * Ẽ_Bb'Q_B*Ẽ_Bb + q_B'Ẽ_Bb




    function test()
        return split_eq_constraint(ϵ)
    end

    return test()
end

# WIP 
# TODO: deflected projected subgradients methods + check what's wrong (in the model)
# Dualised constraints: 
# Ex = b
# l ≤ x ≤ u
function solve_quadratic_min_flow_d2(; 𝔓, ν, ϵ, ϵ_C=ϵ*100, ϵ_Q=0.0)
    # NB: const is still not supported for local variables (er why?)
    (Q, q, l, u, E, b) = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b)
    E = eltype(Q).(E)
    (m, n) = size(E)

    # partition subspaces corresponding to ker(Q)
    ℭ = [Q[i, i] > ϵ_Q for i in 1:n]
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

    # from the singular part of Q we get a linear problem
    # which translates to the equation
    #     λₗ₀ = q₀ + λᵤ₀ + E₀ᵀμ
    # from which we can remove λₗ₀ from the problem, 
    # keeping the inequality constraints
    #     λᵤ₀ + E₀ᵀμ + q₀ .≥ 0
    #     λᵤ, λₗ₁ .≥ 0
    get_λₗ₀ = () -> q₀ + E₀'*μ + λᵤ₀
    λₗ₀[:] = get_λₗ₀()
    # hence we have νᵣ which is ν restricted to the free variables
    νᵣ = view(ν, [[i for i in 1:m+n]; (m+n) .+ findall(ℭ)])
    ν₁ = view(ν, [[i for i in 1:m]; m .+ findall(ℭ); (m+n) .+ findall(ℭ)])

    # I am minimizing -L(⁠ν), which is
    # ½(E₁ᵀμ + λᵤ₁ - λₗ₁)ᵀQ̃₁(E₁ᵀμ + λᵤ₁ - λₗ₁) ( = ½ν₁ᵀT₁ᵀQ̃₁T₁ν₁ = L₂ ) + 
    # q₁ᵀQ̃₁(E₁ᵀμ + λᵤ₁ - λₗ₁) + bᵀμ + u₁ᵀλᵤ₁ + (u₀-l₀)ᵀλᵤ₀ - l₀ᵀE₀ᵀμ - l₁ᵀλₗ₁ ( = tᵀνᵣ = L₁ ) +
    # ½q₁ᵀQ̃₁q₁ - q₀ᵀl₀ ( = L₀ )
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
            α₊, α₋ = α*(1+ϵ_C*sign(α)), α*(1-ϵ_C*sign(α))
            C₊, C₋ = C .* (1 .+ ϵ_C*sign.(C)), C .* (1. .- ϵ_C*sign.(C))
            active_C[𝔲] = ((α₋ .≤ C₊[𝔲]) .& (α₊ .≥ C₋[𝔲]))
            println("$(active_C)")
            #println("$α_lb ≤ α = $α ≤ $α_ub")
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
            
            #if d'∇L < 0.
            #    d[:] = P∇L
            #    project!(view(∇C, :, active_C), d)
            #end
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

# Assumptions:
# constrained space ≠ ∅
# If this is not guaranteed, add a check - it should return nothing by now :)
# Usage (example):  
# x̄, μ, L, ∇L̄ = solve_quadratic_min_flow(𝔓=𝔓, μ=zeros(Float64, 2), ε=1e-12, ϵ=1e-12, reset₀=Inf)
function solve_quadratic_min_flow_d1(; 𝔓, μ, ε=1e-15, ϵ=1e-15, ϵₘ=1e-15, reset₀=Inf)
    Q, q, l, u, E, b = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b)
    Q_diag = [Q[i, i] for i in 1:size(Q, 1)]
    m, n = size(E)      # m: number 
    Q̃ = zeros(eltype(Q), size(Q))
    for i=1:size(Q, 1)
        Q̃[i, i] = 1. / Q[i, i]
    end
    Ql, Qu = Q*l, Q*u
    # println.(["l = $l", "u = $u"])
    Ql_ϵ, Qu_ϵ = Ql+ϵ*abs.(Ql), Qu-ϵ*abs.(Qu)

    function get_L(x, μ)
        return 0.5*x'*Q*x + q'*x + μ'*(E*x-b)
    end
    # x̃ = argminₓ L(x, μ) without box constraints
    function get_Qx̃(μ)
        return -E'*μ-q #(a -> abs(a)>ϵₘ ? a : 0).(-E'*μ-q)
    end
    # ✓
    function get_Qx̃(μ̄, 𝔅)
        return -E[:, 𝔅[:, 2]]'*μ̄ -q[𝔅[:, 2]] #(a -> abs(a)>ϵₘ ? a : 0).(-E[:, 𝔅[:, 2]]'*μ̄ -q[𝔅[:, 2]])
    end
    # x̅ = argminₓ L(x, μ) witholding box constraints l .<= x .<= u
    function get_x̅(μ)
        return [ maximum(noNaN.([min(u[i], (-μ'*E[:, i]-q[i]) / Q[i, i]), l[i]])) for i=1:n ]
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
        return sum(noNaN.([𝔅[:, 1].*l, 𝔅[:, 2].*(Q̃*Qx̃), 𝔅[:, 3].*u]))
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
        𝔩[𝔩] = 𝔩[𝔩] .& (ᾱs[𝔩] .≥ 0)

        𝔲 = [Eᵀd .> 0  Eᵀd .< 0]
        ᾱs[𝔲] = ([Qx̃ Qx̃][𝔲] - [Qu Qu][𝔲]) ./ [Eᵀd Eᵀd][𝔲]
        𝔲[𝔲] = 𝔲[𝔲] .& (ᾱs[𝔲] .≥ 0)

        return (ᾱs, 𝔩, 𝔲)
    end
    # ✓ (todo)
    function get_ᾱs(Qx̃, Eᵀd, 𝔅)
        # 1 : getting inside
        # 2 : going outside
        ᾱs = zeros(eltype(Eᵀd), size(Eᵀd, 1), 2)

        𝔩 = [Eᵀd .< 0  Eᵀd .> 0]        
        ᾱs[𝔩] = ([Qx̃ Qx̃][𝔩] - [Ql Ql][𝔩]) ./ [Eᵀd Eᵀd][𝔩]
        𝔩[𝔩] = 𝔩[𝔩] .& (ᾱs[𝔩] .≥ 0)

        𝔲 = [Eᵀd .> 0  Eᵀd .< 0]
        ᾱs[𝔲] = ([Qx̃ Qx̃][𝔲] - [Qu Qu][𝔲]) ./ [Eᵀd Eᵀd][𝔲]
        𝔲[𝔲] = 𝔲[𝔲] .& (ᾱs[𝔲] .≥ 0)

        return (ᾱs, 𝔩, 𝔲)
    end
    # ✓
    function sortperm_ᾱs(ᾱs, 𝔩, 𝔲)
        P = findall(𝔩 .| 𝔲)
        return sort!(P, lt = (i, j) -> (ᾱs[i], i[2], i[1]) < (ᾱs[j], j[2], j[1]))
        #return (x -> P[x]).(sortperm(ᾱs[𝔩[:] .| 𝔲[:]]))
    end
    # ✓
    function line_search!(x, μ, d, 𝔅)
        Eᵀμ, Eᵀd, dᵀb, Qx̃ = E'*μ, E'*d, d'*b, get_Qx̃(μ)
        ᾱs, 𝔩, 𝔲 = get_ᾱs(Qx̃, Eᵀd)
        P_ᾱs = sortperm_ᾱs(ᾱs, 𝔩, 𝔲)
        # println.(["ᾱs = $(ᾱs)", "P_ᾱs = $(P_ᾱs)", "sorted_ᾱs = $(ᾱs[P_ᾱs])", ""])

        # x(μ) is NaN when it is not a function, so pick the best representative
        function resolve_nan!(x)
            𝔫 = isnan.(x)
            if any(𝔫)
                Inc = Eᵀd[𝔫] .> 0
                Dec = Eᵀd[𝔫] .< 0
                Nul = Eᵀd[𝔫] .== 0
                L̂, Û = Inc.*l[𝔫] + Dec.*u[𝔫], Inc.*u[𝔫] + Dec.*l[𝔫]
                S = dᵀb - Eᵀd[.~𝔫]'*x[.~𝔫]
                λ = (S - Eᵀd[𝔫]'*L̂) / (Eᵀd[𝔫]'*(Û-L̂))
                if 0 ≤ λ ≤ 1
                    x[𝔫] = L̂ + λ*(Û-L̂)
                    return true
                else
                    x[𝔫] = L̂ + (λ > 1)*(Û - L̂)
                    return false
                end
            end
            return nothing
        end

        function find_α!(μ, x, α₀, α₁)
            # ∂L = Eᵀd'*x-dᵀb
            if any(𝔅[:, 2])
                Δα = (Eᵀd'*x-dᵀb) / (Eᵀd[𝔅[:, 2]]' * Q̃[𝔅[:, 2], 𝔅[:, 2]] * Eᵀd[𝔅[:, 2]])
                #println("Δα = $(Δα)")# because \n Δα = $(Eᵀd'*x-dᵀb) / ($(Eᵀd[𝔅[:, 2]]') * $(Q̃[𝔅[:, 2], 𝔅[:, 2]]) * $(Eᵀd[𝔅[:, 2]]))")
                if isnan(Δα)
                    Δα = 0
                end
                if 0 ≤ Δα ≤ α₁-α₀
                    μ[:] = μ + (α₀+Δα)*d
                    x[:] = get_x̅(get_Qx̃(μ), 𝔅)
                    # println.(["x = $x", "μ = $μ", "𝔅 = $𝔅"])
                    # x[𝔅[:, 2]] = -Q̃[𝔅[:, 2], 𝔅[:, 2]] * (E[:, 𝔅[:, 2]]'*μ + q[𝔅[:, 2]])
                    return true
                end
            end
            return false
        end

        ᾱ, μ̄ = 0., copy(μ) 
        for i in P_ᾱs
            # println.(["x = $x", "μ̄ = $μ̄ ", "μ = $μ", "ᾱ = $ᾱ", "𝔅 = $(𝔅[:, 2])"])
            resolved_nan = resolve_nan!(x)
            if resolved_nan == true
                # println("resolved NaN -> \n\tμ = $μ \n\tx = $x\n")
                return
            elseif resolved_nan == nothing
                if find_α!(μ, x, ᾱ, ᾱs[i]) == true
                    # println("found α -> \n\tμ = $μ \n\tx = $x\n")
                    return
                end
            end

            # println("")
            # set 𝔅 for next ᾱ
            𝔅[i[1], :] = (i[2] == 2) ? [𝔩[i] false 𝔲[i]] : [false true false]
            ᾱ = ᾱs[i]
            μ̄  = μ + ᾱ*d
            Qx̃[𝔅[:, 2]] = get_Qx̃(μ̄, 𝔅)
            x[𝔅[:, 2]] = max.(min.(Q̃[𝔅[:, 2], 𝔅[:, 2]]*Qx̃[𝔅[:, 2]], u[𝔅[:, 2]]), l[𝔅[:, 2]])
            # println("x[$(𝔅[:, 2])] = $(x[𝔅[:, 2]]) = max.(min.($(Q̃[𝔅[:, 2], 𝔅[:, 2]])*$(Qx̃[𝔅[:, 2]]), $(u[𝔅[:, 2]])), $(l[𝔅[:, 2]]))")
        end
        # println.(["x = $x", "μ̄ = $μ̄ ", "μ = $μ", "ᾱ = $ᾱ", "𝔅 = $(𝔅[:, 2])", ""])
        resolved_nan = resolve_nan!(x)
        if resolved_nan == true
            # println("resolved NaN -> \n\tμ = $μ \n\tx = $x\n")
            return
        elseif resolved_nan == nothing
            if find_α!(μ, x, ᾱ, Inf) == true
                # println("found α -> \n\tμ = $μ \n\tx = $x\n")
                return
            end
        end
    end

    function solve()
        Qx̃ = get_Qx̃(μ)
        𝔅 = zeros(Bool, size(E, 2), 3)
        on_box_side!(Qx̃, 𝔅)
        x̅ = get_x̅(Qx̃, 𝔅)

        while any(isnan.(x̅))
            println("Perturbing the starting μ to avoid NaNs")
            μ += ε*(rand(eltype(μ), size(μ, 1))-0.5)
            Qx̃ = get_Qx̃(μ)
            𝔅 = zeros(Bool, size(E, 2), 3)
            on_box_side!(Qx̃, 𝔅)
            x̅ = get_x̅(Qx̃, 𝔅)
        end

        ∇L = get_∇L(x̅)
        d = copy(∇L)
        ∇L₀ = copy(∇L)
        #L₀, L = -Inf, get_L(x̅, μ)
        reset = reset₀
        counter = 0
        while (norm(∇L) ≥ ε) # && (L-L₀ ≥ ε*abs(L))
            # println("\n---------------------\n d = $d\n")
            line_search!(x̅, μ, d, 𝔅)
            # L₀, L = L, get_L(x̅, μ)
            ∇L₀, ∇L = ∇L, get_∇L(x̅)
            # d = reset == 0 ? (reset = reset₀; ∇L) : (reset = reset-1; ∇L - d*(∇L'*EQ̃Eᵀ*d)/(d'*EQ̃Eᵀ*d) )
            #if norm(∇L) > 1
            #    Qx̃ = get_Qx̃(μ)
            #    on_box_side!(Qx̃, 𝔅)
            #    x̅ = get_x̅(Qx̃, 𝔅)
            #    ∇L = get_∇L(x̅)
            #end
            #if reset == 0
            #    reset = reset₀;
            #    d[:] = ∇L 
            #else
                reset = reset-1
                d[:] = ∇L + d*(∇L'*∇L - ∇L'*∇L₀) / (∇L₀'*∇L₀)
                if d'*∇L < 0
                    d[:] = ∇L
                end
                #println("dᵀ*∇L = $(d'*∇L)")
            #end
            println.(["|∇L| = $(norm(∇L))"]) #, "L = $L", ""])
            counter += 1
        end

        L = get_L(x̅, μ)
        println("L = $L")
        println("\n$counter iterazioni\n")
        return (x̅, μ, L, ∇L)
    end

    return solve()
end

# For the specific case when E is the incidence matrix of a graph, 
# the problem is separable in the subproblems corresponding to the
# connected components, calculated hereafter as bitmasks on the nodes
# E : node-arc incidence matrix
function get_graph_components(E)
    # m : number of nodes
    # n : number of arcs
    m, n = size(E)
    M = E .≠ 0
    B = zeros(Bool, m)
    P = zeros(Bool, m, 0)
    for i in 1:m
        if B[i] == true
            continue
        end
        
        P = cat(P, zeros(Bool, m), dims=2)

        B[i] = true
        P[i, end] = true

        Vᵢ = begin
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

    return P
end