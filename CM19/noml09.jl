using LinearAlgebra
using SparseArrays

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
function generate_quadratic_min_cost_flow_boxed_problem(type, m, n)
    Q = spdiagm(0 => sort(rand(type, n), rev=true))
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

# Assumptions:
# constrained space ≠ ∅
# If this is not guaranteed, add a check - it should return nothing by now :)

function solve_quadratic_min_flow(; 𝔓, μ, ε, ϵ=1e-15)
    Q, q, l, u, E, b = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b)
    m, n = size(E)      # m: number 
    Q̃ = zeros(eltype(Q), size(Q))
    for i=1:size(Q, 1)
        Q̃[i, i] = 1. / Q[i, i]
    end
    Ql, Qu = Q*l, Q*u
    Ql_ϵ, Qu_ϵ = Ql+ϵ*abs.(Ql), Qu-ϵ*abs.(Qu)

    function get_L(x, μ)
        return 0.5*x'*Q*x + q'*x + μ'*(E*x-b)
    end
    # x̃ = argminₓ L(x, μ) without box constraints
    function get_Qx̃(μ)
        return -E'*μ-q
    end
    # x̅ = argminₓ L(x, μ) witholding box constraints l .<= x .<= u
    function get_x̅(μ)
        return [ maximum(noNaN.([min(u[i], (-μ'*E[:, i]-q[i]) / Q[i, i]), l[i]])) for i=1:n ]
    end
    # mark if x is on a side of the box constraints
    # 1 -> lower  2 -> interior  3 -> upper
    function on_box_side!(Qx̃, 𝔅)
        𝔅[:, 1] .= (Qx̃ .≤ Ql_ϵ)
        𝔅[:, 3] .= (Qx̃ .≥ Qu_ϵ) .& (.~𝔅[:, 1])
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
    # Calculate all points in line search in which an x is touching the side of the box constraint
    function get_ᾱs_μ(μ, d)
        Eᵀd = E'*d
        ᾱs = zeros(eltype(μ), size(E, 2))

        𝔩 = Eᵀd .> 0
        ᾱs[𝔩] = -(E[:, 𝔩]'*μ + q[𝔩] + Ql[𝔩]) ./ Eᵀd[𝔩]
        𝔩[𝔩] .&= (ᾱs[𝔩] .≥ 0)

        𝔲 = Eᵀd .< 0
        ᾱs[𝔲] = -(E[:, 𝔩]'*μ + q[𝔲] + Qu[𝔲]) ./ Eᵀd[𝔲]
        𝔲[𝔲] .&= (ᾱs[𝔲] .≥ 0)

        return (ᾱs, 𝔩, 𝔲)
    end
    function get_ᾱs(Qx̃, Eᵀd)
        # 1 : approaching the box side from the inside
        # 2 : getting inside from a box side
        ᾱs = zeros(eltype(Eᵀd), size(Eᵀd, 1), 2)

        𝔩 = [Eᵀd .> 0  Eᵀd .< 0]        
        ᾱs[𝔩] = ([Qx̃ Qx̃][𝔩] - [Ql Ql][𝔩]) ./ [Eᵀd Eᵀd][𝔩]
        𝔩[𝔩][:] .&= (ᾱs[𝔩] .≥ 0)

        𝔲 = [Eᵀd .< 0 Eᵀd .> 0]
        ᾱs[𝔲] = ([Qx̃ Qx̃][𝔲] - [Qu Qu][𝔲]) ./ [Eᵀd Eᵀd][𝔲]
        𝔲[𝔲][:] .&= (ᾱs[𝔲] .≥ 0)

        return (ᾱs, 𝔩, 𝔲)
    end
    function sortperm_ᾱs(ᾱs, 𝔩, 𝔲)
        P = findall(𝔩 .| 𝔲)
        return (x -> P[x]).(sortperm(ᾱs[𝔩[:] .| 𝔲[:]]))
    end
    function line_search(∇L, μ, d)
        Eᵀμ = E'*μ
        Eᵀd = E'*d
        dᵀb = d'*b
        Qx̃ = get_Qx̃(μ)
        ᾱs, 𝔩, 𝔲 = get_ᾱs(Qx̃, Eᵀd)
        P_ᾱs = sortperm_ᾱs(ᾱs, 𝔩, 𝔲)
        𝔅 = zeros(Bool, size(Qx̃, 1), 3)
        on_box_side!(Qx̃, 𝔅)
        
        ∂L = d'*∇L
        ∂L̄ = ∂L
        ᾱ = 0.
        for i_ᾱ in P_ᾱs
            Δ = (Eᵀd[𝔅[:, 2]]' * Q̃[𝔅[:, 2], 𝔅[:, 2]] * Eᵀd[𝔅[:, 2]])
            Δα̃ = ∂L̄ / Δ
            println("0. ≤ $Δα̃  ≤ $(ᾱs[i_ᾱ]-ᾱ)")
            if 0. ≤ Δα̃  ≤ ᾱs[i_ᾱ]-ᾱ
                return ᾱ + Δα̃
            end
            # The faster idea was to add just the increment to ∂L, something like
            # ∂L̄ += (ᾱs[i_ᾱ]-ᾱ) * Δ, but it would be unstable (?) and should be customised 
            # for the case Qᵢᵢ = 0 (?maybe). To begin with, I'll try with the next one, more expensive:
            𝔅[i_ᾱ[1], :] = (i_ᾱ[2] == 1) ? [𝔩[i_ᾱ] false 𝔲[i_ᾱ]] : [false true false]
            ᾱ = ᾱs[i_ᾱ]
            x̄ = get_x̅(get_Qx̃(μ + ᾱ*d), 𝔅)
            ∂L̄ = Eᵀd'*x̄ - dᵀb
        end
    end

    Qx̃ = get_Qx̃(μ)
    𝔅 = zeros(Bool, size(E, 2), 3)
    on_box_side!(Qx̃, 𝔅)
    x̅ = get_x̅(Qx̃, 𝔅)
    ∇L₁ = get_∇L(x̅)
    d = ∇L₁
    while norm(∇L₁) ≥ ε
        α = line_search(∇L₁, μ, d)
        μ += α*d
        ∇L₀ = ∇L₁
        Qx̃ = get_Qx̃(μ)
        on_box_side!(Qx̃, 𝔅)
        x̅ = get_x̅(Qx̃, 𝔅)
        ∇L₁ = get_∇L(x̅)
        d = ∇L₁ + d*(∇L₁'*∇L₁ - ∇L₁'*∇L₀) / (∇L₀'*∇L₀)
        println("|∇L₁| = $(norm(∇L₁))")
    end

    return (get_L(x̅, μ), x̅, μ, ∇L₁)
end