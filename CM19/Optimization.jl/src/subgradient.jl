module Subgradient

using LinearAlgebra
using ..Optimization
using ..Optimization.Utils

import ..Optimization.Descent: init!, step!
export  SubgradientMethod, DeflectedSubgradientMethod, init!, step!

# Subgradient methods
abstract type SubgradientMethod end

mutable struct WithStopCriterion{M} <: SubgradientMethod where M <: SubgradientMethod
    method::M
    R               # (estimate) upper bound on the size of the region
    l               # lower bound on the objective function
    l′              # best lower bound
end
function init!(M::WithStopCriterion{<:SubgradientMethod}, f, ∂f, x)
    M.l′ = -Inf
    M.l = [-M.R*M.R, 0.0]
end
function step!(M::WithStopCriterion{<:SubgradientMethod}, f, ∂f, x)
    f₀, ∂f₀ = f(x), ∂f(x)
    norm∂f₀ = norm(∂f₀)

    (x, α, d) = step!(M.method, f, ∂f, x)
    M.l[:] = M.l + [α*(2.0*f₀ - α*norm∂f₀*norm∂f₀), 2.0*α]
    M.l′ = max(M.l′, M.l[1]/M.l[2])
    return (x, α, d)
end

struct FixedStepSize <: SubgradientMethod
    α
end
"""
Guarantee
    f_best - f_opt ≤ RG/√iter
"""
function get_optimal_step_size(_::FixedStepSize, R, G, iters)
    R/(G*√iters)
end
init!(M::FixedStepSize, f, ∂f, x) = M
function step!(M::FixedStepSize, f, ∂f, x)
    α, sg = M.α, ∂f(x)
    return (x - α*sg, α, sg)
end

mutable struct NoMemoryStepSize <: SubgradientMethod
    i
    gen_α
end
function init!(M::NoMemoryStepSize, f, ∂f, x)
    i=0
end
function step!(M::NoMemoryStepSize, f, ∂f, x)
    sg = ∂f(x)
    α = gen_α(M.i+=1, f, ∂f, x, sg)
    return (x - α*sg, α, sg)
end

struct FixedStepLength <: SubgradientMethod
    γ
end
init!(M::FixedStepLength, f, ∂f, x) = M
function step!(M::FixedStepLength, f, ∂f, x)
    γ, sg = M.γ, ∂f(x)
    γ/norm(sg) |>
        α -> (x - α*sg, α, sg)
end
mutable struct NoMemoryStepLength <: SubgradientMethod
    i
    gen_γ
end
function init!(M::NoMemoryStepLength, f, ∂f, x)
    i=0
end
function step!(M::NoMemoryStepLength, f, ∂f, x)
    sg = ∂f(x)
    γ = gen_γ(M.i+=1, f, ∂f, x, sg)
    γ/norm(sg) |>
        α -> (x - α*sg, α, sg)
end

mutable struct PolyakStepSize <: SubgradientMethod
    f_opt       # optimum objective value if available
    gen_γ       # estimated error in objective

    i           # iteration counter, needed for γ
    f_best      # useful when f_opt should be estimated
    gen_α       # Polyak step size generator

    PolyakStepSize(;f_opt=nothing, gen_γ=nothing) = begin
        M = new()
        @some M.f_opt = f_opt
        @some M.gen_γ = gen_γ

        if f_opt !== nothing
            M.gen_α = (f, ∂f) -> (f-M.f_opt) / (∂f'∂f)
        elseif gen_γ !== nothing
            M.gen_α = (f, ∂f) -> begin
                M.f_best = min(M.f_best, f)
                (M.gen_γ(M.i+=1) |>
                γ -> (f-M.f_best+γ) / (∂f'∂f))
            end
        end
        M
    end
end
function init!(M::PolyakStepSize, f, ∂f, x)
    M.i = 0
    M.f_best = Inf
end
function step!(M::PolyakStepSize, f, ∂f, x)
    sg = ∂f(x)
    α = M.gen_α(f(x), sg)
    (x-α*sg, α, sg)
end

abstract type DeflectedSubgradientMethod <: SubgradientMethod end

mutable struct PolyakEllipStepSize <: DeflectedSubgradientMethod
    gen_γ
    ϵ

    i
    B
    g
    f_best
    f_opt
    PolyakEllipStepSize(;f_opt=nothing, gen_γ=nothing, ϵ=1e-14) = begin
        M = new()
        M.ϵ = ϵ
        @some M.gen_γ = gen_γ
        if f_opt !== nothing
            M.f_opt = () -> f_opt
        else
            M.f_opt = () -> M.f_best - M.gen_γ(M.i+=1)
        end
        M
    end
end
function init!(M::PolyakEllipStepSize, f, ∂f, x)
    M.i = 0
    M.B = I
    M.f_best = Inf
    M.g = ∂f(x)
end
function step!(M::PolyakEllipStepSize, f, ∂f, x)
    f_val = f(x)
    M.f_best = min(M.f_best, f_val)

    f_opt = M.f_opt()
    g = copy(M.g)
    Bᵀg = M.B'g
    normBᵀg = norm(Bᵀg)
    h = (f_val - f_opt) / normBᵀg
    ξ = Bᵀg / normBᵀg
    Bξ = M.B*ξ
    x′ = x - h*Bξ
    M.g = ∂f(x′)
    Bᵀg′ = M.B'M.g
    normBᵀg′ = norm(Bᵀg′)
    ξ′ = Bᵀg′ / normBᵀg′
    μ = ξ⋅ξ′
    if μ < 0.0
        η = ϵ+√(1.0-min(μ*μ, 1.0)) |> γ -> (1.0/γ - 1.0)*ξ′ - μ*ξ/γ
        M.B += (M.B*η)*ξ′'
    end
    (x′, h, Bξ)
end

mutable struct Adagrad <: DeflectedSubgradientMethod
    α # learning rate, typical 0.01 → 1.0
    ϵ # small value, typical 1e-8

    s # sum of squared gradient
    Adagrad(;α=nothing, ϵ=1e-14) = begin
        M = new()
        @some M.α = α
        M.ϵ = ϵ
        M
    end
end
function init!(M::Adagrad, f, ∂f, x)
    M.s = zeros(length(x))
    M
end
function step!(M::Adagrad, f, ∂f, x)
    α, ϵ, s, g = M.α, M.ϵ, M.s, ∂f(x)
    s[:] += g .* g
    d = g ./ (sqrt.(s) .+ ϵ)
    (x - α*d, α, d)
end

# TODO
mutable struct AdagradFull <: DeflectedSubgradientMethod
    α   # learning rate
    ϵ   # small value

    G   # ∑ggᵀ
end
function init!(M::AdagradFull, f, ∂f, x)

end
function step!(M::AdagradFull, f, ∂f, x)

end

mutable struct NesterovMomentum <: DeflectedSubgradientMethod
    α # learning rate
    β # momentum decay
    v # momentum

    NesterovMomentum(;α=nothing, β=nothing) = begin
        M = new()
        @some M.α = α
        @some M.β = β
        M
    end
end
function init!(M::NesterovMomentum, f, ∂f, x)
    M.v = zeros(length(x))
end
function step!(M::NesterovMomentum, f, ∂f, x)
    α, β, v = M.α, M.β, M.v
    g = ∂f(x + β*v)
    v[:] = β*v - α*g
    (x+v, α, g, β, v)
end

mutable struct RMSProp <: DeflectedSubgradientMethod
    α # learning rate, e.g. 0.00007, better
    γ # decay, e.g. 0.99999, better 0.999999
    ϵ # small value

    s # sum of squared gradient
    RMSProp(; α=nothing, γ=nothing, ϵ=1e-14) = begin
        M = new()
        @some M.α = α
        @some M.γ = γ
        M.ϵ = ϵ
        M
    end
end
function init!(M::RMSProp, f, ∂f, x)
    M.s = zeros(length(x))
    M
end
function step!(M::RMSProp, f, ∂f, x)
    α, γ, ϵ, s, g = M.α, M.γ, M.ϵ, M.s, ∂f(x)
    s[:] = γ*s + (1-γ)*(g .* g)
    g′ = g ./ (sqrt.(s) .+ ϵ)
    (x - α*g′, α, g′)
end

mutable struct Adadelta <: DeflectedSubgradientMethod
    γs # gradient decay, tried  0.9
    γx # update decay, tried 0.9
    ϵ # small value, typical 1e-8

    s # sum of squared gradients
    u # sum od squared gradients
    AdadeltaDescent(;γs=nothing, γx=nothing, ϵ=1e-14) = begin
        M = new()
        @some M.γs = γs
        @some M.γx = γx
        M.ϵ = ϵ
        M
    end
end
function init!(M::Adadelta, f, ∂f, x)
    M.s = zeros(length(x))
    M.u = zeros(length(x))
    return M
end
function step!(M::Adadelta, f, ∂f, x)
    γs, γx, ϵ, s, u, g = M.γs, M.γx, M.ϵ, M.s, M.u, ∂f(x)
    s[:] = γs*s + (1-γs)*g.*g
    Δx = - (sqrt.(u) .+ ϵ) ./ (sqrt.(s) .+ ϵ) .* g
    u[:] = γx*u + (1-γx)*Δx.*Δx
    (x+Δx, nothing, Δx)
end

mutable struct Adam <: DeflectedSubgradientMethod
    α # learning rate, e.g. 0.001
    γv # decay, e.g. 0.9
    γs # decay,e.g. 0.999
    ϵ # small value,

    k # step counter
    v # 1st moment estimate
    s # 2nd moment estimate
    AdamDescent(;α=nothing, γv=nothing, γs=nothing, ϵ=1e-14) = begin
        M = new()
        @some M.α = α
        @some M.γv = γv
        @some M.γs = γs
        M.ϵ = ϵ
        M
    end
end
function init!(M::Adam, f, ∂f, x)
    M.k = 0
    M.v = zeros(length(x))
    M.s = zeros(length(x))
    M
end
function step!(M::Adam, f, ∂f, x)
    α, γv, γs, ϵ, k = M.α, M.γv, M.γs, M.ϵ, M.k
    s, v, g = M.s, M.v, ∇f(x)
    v[:] = γv*v + (1-γv)*g
    s[:] = γs*s + (1-γs)*g.*g
    M.k = k += 1
    v̂ = v ./ (1 - γv^k)
    ŝ = s ./ (1 - γs^k)
    (x - α*v̂ ./ (sqrt.(ŝ) .+ ϵ), nothing, nothing)
end

mutable struct HyperGradient <: DeflectedSubgradientMethod
    α₀ # initial learning rate
    µ # learning rate of the learning rate

    α # current learning rate
    g_prev # previous gradient
    HyperGradient(; α₀=nothing, μ=nothing) = begin
        M = new()
        @some M.α₀ = α₀
        @some M.μ = μ
        M
    end
end
function init!(M::HyperGradient, f, ∂f, x)
    M.α = M.α₀
    M.g_prev = zeros(length(x))
    M
end
function step!(M::HyperGradient, f, ∂f, x)
    α, µ, g, g_prev = M.α, M.µ, ∂f(x), M.g_prev
    α = α + µ*(g⋅g_prev)
    M.g_prev, M.α = g, α
    (x - α*g, α, g)
end

mutable struct HyperNesterovMomentum <: DeflectedSubgradientMethod
    α₀ # initila learning rate
    µ # learning rate of the learning rate
    β # momentum decay

    v # momentum
    α # current learning rate
    g_prev # previous gradient
    HyperNesterovMomentum(;α₀=nothing, μ=nothing, β=nothing) = begin
        M = new()
        @some M.α₀ = α₀
        @some M.μ = μ
        @some M.β = β
        M
    end
end
function init!(M::HyperNesterovMomentum, f, ∂f, x)
    M.α = M.α₀
    M.v = zeros(length(x))
    M.g_prev = zeros(length(x))
    M
end
function step!(M::HyperNesterovMomentum, f, ∂f, x)
    α, β, µ = M.α, M.β, M.µ
    v, g, g_prev = M.v, ∂f(x), M.g_prev
    α = α - µ*(g⋅(-g_prev - β*v))
    v[:] = β*v + g
    M.g_prev, M.α = g, α
    g′ = g + β*v
    (x - α*g′, α, g′)
end

mutable struct Noisy <: DeflectedSubgradientMethod
    submethod::DeflectedSubgradientMethod
    σ

    k
    Noisy(;submethod=nothing, σ=nothing) = begin
        M = new()
        @some M.submethod = submethod
        @some M.σ = σ
        M
    end
end
function init!(M::Noisy, f, ∂f, x)
    init!(M.submethod, f, ∂f, x)
    M.k = 1
    M
end
function step!(M::Noisy, f, ∂f, x)
    x = step!(M.submethod, f, ∂f, x)
    σ = M.σ(M.k)
    x += σ.*randn(length(x))
    M.k += 1
    (x, nothing, nothing)
end

end     # end module Subgradient
