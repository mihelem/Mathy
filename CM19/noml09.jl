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

# Assumptions:
# constrained space ≠ ∅
# If this is not guaranteed, add a check - it should return nothing by now :)

function solve_quadratic_min_flow(; 𝔓, μ, ε=1e-15, ϵ=1e-15, ϵₘ=1e-15, reset₀=Inf)
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
            if norm(∇L) > 1
                Qx̃ = get_Qx̃(μ)
                on_box_side!(Qx̃, 𝔅)
                x̅ = get_x̅(Qx̃, 𝔅)
                ∇L = get_∇L(x̅)
            end
            if reset == 0
                reset = reset₀;
                d[:] = ∇L 
            else
                reset = reset-1
                d = ∇L + d*(∇L'*∇L - ∇L'*∇L₀) / (∇L₀'*∇L₀)
                if d'*∇L < 0
                    d = ∇L
                end
                println("dᵀ*∇L = $(d'*∇L)")
            end
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