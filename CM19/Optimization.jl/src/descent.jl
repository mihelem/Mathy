"""
Code Snippets built upon
    Algorithms for Optimization
by Wheeler & Kochenderfer, ©2019 MIT Press
"""

module Descent

using LinearAlgebra
using Parameters
using Statistics
using ..Optimization
using ..Optimization.Utils

export  ZerothOrder,
    DescentMethod,
    init!,
    step!,
    NelderMead,
    GradientDescent,
    ConjugateGradientDescent,
    MomentumDescent,
    NesterovMomentumDescent,
    AdagradDescent,
    RMSPropDescent,
    AdadeltaDescent,
    AdamDescent,
    HyperGradientDescent,
    HyperNesterovMomentumDescent,
    NoisyDescent

# Zeroth Order Methods
abstract type ZerothOrder <: LocalizationMethod end

mutable struct NelderMead <: ZerothOrder
    α       # reflection parameter > 0
    β       # expansion parameter > max(1, α)
    γ       # contraction parameter ∈ [0, 1]
    cmp     # comparison operator, default <

    S       # simplex points
    y       # simplex evaluated on given function
    Δ       # std dev of evaluated simplex
    params
    NelderMead(; α=1.0, β=2.0, γ=0.5, cmp=<) = begin
        M = new(max(α, 0.0), maximum([β, α, 1]), min(1.0, max(γ, 0.0)), cmp)

        M.Δ = () -> std(M.y, corrected=false)
        M.params = Dict(:α => [0.0, Inf], :β => [1, Inf], :γ => [0.0, 1.0])
        M
    end
end
function init!(M::NelderMead, f=nothing)
    M
end
function init!(M::NelderMead, f, S)
    M.S, M.y = S, f.(S)
end
function init!(M::NelderMead, f, S, y)
    M.S, M.y = S, y
end
"""
**Pedices**
* `l`, `h-1`, `h` : lowest, second-highest, highest
* `r` : reflection
* `e` : expansion
* `t` : contraction
"""
import Statistics: mean, std
function apply_by_kw(f, V::Array{Dict})
    array_dict_to_dict_array(V) |>
        da -> Dict([s=>f(v) for (s, v) in da])
end
function array_dict_to_dict_array(V::Array{Dict})
    if length(V) < 1
        return Dict()
    end
    res = Dict()
    for (s, r) in V[1]
        push!(res, s=>[])
    end
    for v in V
        for (s, r) in v
            push!(res[s], r)
        end
    end
    res
end
function mean(V::Array{Dict})
    apply_by_kw(mean, V)
end
function std(V::Array{Dictionary}, corrected::Bool)
    apply_by_kw(std, V)
end
function step!(M::NelderMead, f)
    @unpack S, y, cmp = M

    p = sortperm(y, lt=cmp)
    S[:] = S[p]
    y[:] = y[p]
    xₗ, xₕ₋₁, xₕ = S[1], S[end-1], S[end]
    yₗ, yₕ₋₁, yₕ = y[1], y[end-1], y[end]
    x̄ = mean(S[1:end-1])       # centroid
    xᵣ = xₘ + α*(x̄ - xₕ)       # reflection point
    yᵣ = f(xᵣ)

    if cmp(yᵣ, yₗ)
        xₑ = x̄ + β*(xᵣ - x̄)   # expansion point
        yₑ = f(xₑ)
        S[end], y[end] = cmp(yₑ, yᵣ) ? (xₑ, yₑ) : (xᵣ, yᵣ)
    elseif cmp(yₕ₋₁, yᵣ)
        if !cmp(yₕ, yᵣ)
            xₕ, yₕ, S[end], y[end] = xᵣ, yᵣ, xᵣ, yᵣ
        end
        xₜ = xₘ + γ*(xₕ - x̄)   # contraction point
        yₜ = f(xₜ)
        if cmp(yₕ, yₜ)               # shrinkage
            for i in 2:length(y)
                S[i] = (S[i] + xₗ) / 2
                y[i] = f(S[i])
            end
        else
            S[end], y[end] = xₜ, yₜ
        end
    else
        S[end], y[end] = xᵣ, yᵣ
    end

    function amin(y, cmp)
        if length(y) < 1
            return
        end
        am, m = 1, y[1]
        for i in 2:length(y)
            if cmp(y[i], m)
                am, m = i, y[i]
            end
        end
        am
    end
    amin(y, lt=cmp) |> i -> (S[i], y[i])
end

# First Order Methods

abstract type DescentMethod <: LocalizationMethod end
mutable struct GradientDescent <: DescentMethod
    α

    params
    GradientDescent(;α=0.1) = begin
        M = new(α)
        M.params = Dict(:α => [0.0, 1.0])
        M
    end
end
init!(M::GradientDescent, f, ∇f, x) = M
function step!(M::GradientDescent, f, ∇f, x)
    α, g = M.α, ∇f(x)
    return x - α*g
end

mutable struct ConjugateGradientDescent <: DescentMethod
    d
    g
    line_search

    params
    ConjugateGradientDescent() = begin
        M = new()
        M.params = Dict()
    end
end
function init!(M::ConjugateGradientDescent, f, ∇f, x)
    M.g = ∇f(x)
    M.d = -M.g
    return M
end
# mmh...
function step!(M::ConjugateGradientDescent, f, ∇f, x)
    d, g, line_search = M.d, M.g, M.line_search
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
    params
    MomentumDescent(;α=1.0, β=0.5) = begin
        M = new(α, β)
        M.params = Dict(:α => [0.0, 1.0], :β => [0.0, 1.0])
        M
    end
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

    params
    NesterovMomentumDescent() = begin
        M = new(0.5, 0.1)
        M.params = Dict(:α => [0.0, 1.0], :β => [0.0, 1.0])
        M
    end
end
function init!(M::NesterovMomentumDescent, f, ∇f, x)
    M.v = zeros(length(x))
end
function step!(M::NesterovMomentumDescent, f, ∇f, x)
    α, β, v = M.α, M.β, M.v
    v[:] = β*v - α*∇f(x + β*v)
    return x+v
end

mutable struct AdagradDescent <: DescentMethod
    α # learning rate
    ϵ # small value

    s # sum of squared gradient
    params
    AdagradDescent() = begin
        M = new(0.01, 1e-8)
        M.params = Dict(:α => [0.0, 1.0], :ϵ => [0.0, 1e-4])
        M
    end
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
    params
    RMSPropDescent() = begin
        M = new(0.01, 0.9, 1e-8, )
        M.params = Dict(:α => [0.0, 1.0], :γ => [0.0, 1.0], :ϵ => [0.0, 1e-4])
        M
    end
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
    params
    AdadeltaDescent() = begin
        M = new(0.9, 0.9, 1e-8)
        M.params = Dict(:γs => [0.0, 1.0], :γx => [0.0, 1.0], :ϵ => [0.0, 1e-4])
        M
    end
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
    params
    AdamDescent() = begin
        M = new(0.001, 0.9, 0.999, 1e-8)
        M.params = Dict(:α => [0.0, 1.0], :γv => [0.0, 1.0], :γs => [0.0, 1.0], :ϵ => [0.0, 1e-4])
        M
    end
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
    params
    HyperGradientDescent(;α₀=1.0, μ=1.0) = begin
        M = new(α₀, μ)
        M.params = Dict(:α₀ => [0.0, 1.0], :μ => [0.0, 1.0])
        M
    end
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
    α₀ # initial learning rate
    µ # learning rate of the learning rate
    β # momentum decay

    v # momentum
    α # current learning rate
    g_prev # previous gradient
    params
    HyperNesterovMomentumDescent(; α₀=1.0, μ=1.0, β=0.5) = begin
        M = new(α₀, μ, β)
        M.params = Dict(:α₀ => [0.0, 1.0], :μ => [0.0, 1.0], :β => [0.0, 1.0])
        M
    end
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
    params
    NoisyDescent(;submethod=nothing, σ=nothing) = begin
        M = new()
        @some M.submethod = submethod
        @some M.σ = σ
        M.params = Dict(:σ => [0.0, Inf])
        M
    end
end
function init!(M::NoisyDescent, f, ∇f, x)
    init!(M.submethod, f, ∇f, x)
    M.k = 1
    return M
end
function step!(M::NoisyDescent, f, ∇f, x)
    x = step!(M.submethod, f, ∇f, x)
    σ = M.σ(M.k)
    x += σ.*randn(length(x))
    M.k += 1
    return x
end

end     # end module Descent
