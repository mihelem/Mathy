module Subgradient

using LinearAlgebra
using Parameters
using ..Optimization
using ..Optimization.Utils

import ..Optimization.set_param!
import ..Optimization.Descent: init!, step!
export  SubgradientMethod,
    DualSubgradientMethod,
    DeflectedSubgradientMethod,
    init!,
    step!,
    set_param!

# Subgradient methods
abstract type SubgradientMethod <: LocalizationMethod end

mutable struct WithStopCriterion{SM <: SubgradientMethod} <: SubgradientMethod
    method::SM
    R::AbstractFloat    # (estimate) upper bound on the size of the region

    l                   # lower bound on the objective function
    l′                  # best lower bound
    params
    WithStopCriterion{SM}(method; R=Inf) where {SM <: SubgradientMethod} = begin
        M = new{SM}(method, R)
        M.params = method.params
        M
    end
end
function set_param!(M::WithStopCriterion{<:SubgradientMethod}, s::Symbol, v)
    setfield!(M.method, s, v)
end
function init!(M::WithStopCriterion{<:SubgradientMethod}, f, ∂f, x)
    init!(M.method, f, ∂f, x)
    M.l′ = -Inf
    M.l = [-M.R*M.R, 0.0]
    M
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

    params
    FixedStepSize(α=nothing) = begin
        M = new()
        @some M.α = α
        M.params = Dict(:α => [0.0, Inf])
        M
    end
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
    gen_α
    α_mul   # redundant factor to be used in parameter search

    i
    params
    NoMemoryStepSize(gen_α; α_mul=1.0) = begin
        M = new(gen_α, α_mul)
        M.params = Dict(:α_mul => [0.0, Inf])
        M
    end
end
function init!(M::NoMemoryStepSize, f, ∂f, x)
    i=0
end
function step!(M::NoMemoryStepSize, f, ∂f, x)
    sg = ∂f(x)
    α = M.α_mul*M.gen_α(M.i+=1, f, ∂f, x, sg)
    return (x - α*sg, α, sg)
end

struct FixedStepLength <: SubgradientMethod
    γ

    params
    FixedStepLength(γ) = begin
        M = new(γ)
        M.params = Dict(:γ => [0.0, Inf])
        M
    end
end
init!(M::FixedStepLength, f, ∂f, x) = M
function step!(M::FixedStepLength, f, ∂f, x)
    γ, sg = M.γ, ∂f(x)
    γ/norm(sg) |>
        α -> (x - α*sg, α, sg)
end

mutable struct NoMemoryStepLength <: SubgradientMethod
    gen_γ
    γ_mul       # redundant factor to be used in parameter search

    i
    params
    NoMemoryStepLength(gen_γ; γ_mul=1.0) = begin
        M = new(gen_γ, γ_mul)
        M.params = Dict(:γ_mul => [0.0, 1.0])
        M
    end
end
function init!(M::NoMemoryStepLength, f, ∂f, x)
    M.i=0
    M
end
function step!(M::NoMemoryStepLength, f, ∂f, x)
    sg = ∂f(x)
    γ = M.γ_mul*M.gen_γ(M.i+=1, f, ∂f, x, sg)
    γ/norm(sg) |>
        α -> (x - α*sg, α, sg)
end

mutable struct PolyakStepSize <: SubgradientMethod
    f_opt       # optimum objective value if available
    gen_γ       # estimated error in objective
    β           # factor for the stepsize
    γ_mul       # redundant factor to be used in parameter search

    i           # iteration counter, needed for γ
    f_best      # useful when f_opt should be estimated
    gen_α       # Polyak step size generator
    params
    PolyakStepSize(;f_opt=nothing, gen_γ=nothing, γ_mul=1.0, β=1.0) = begin
        M = new()
        @some M.f_opt = f_opt
        @some M.gen_γ = gen_γ
        M.β = β
        M.γ_mul = γ_mul

        if f_opt !== nothing
            M.gen_α = (f, ∂f) -> M.β*(f-M.f_opt) / (∂f'∂f)
        elseif gen_γ !== nothing
            M.gen_α = (f, ∂f) -> begin
                M.f_best = min(M.f_best, f)
                (M.gen_γ(M.i+=1) |>
                γ -> M.β*(f-M.f_best+γ) / (∂f'∂f))
            end
        end
        M.params = Dict(:γ_mul => [0.0, 1.0], β => [0.0, 2.0])
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

mutable struct TargetLevelPolyakStepSize <: SubgradientMethod
    β   # step factor
    δ   # target threshold
    ρ   # decay of δ
    R   # 
    f_opt
    f_target
    get_f_target

    f_best
    r
    i
    params
    function TargetLevelPolyakStepSize(;
        β=1.0,
        δ=0.0,
        ρ=1.0,
        R=Inf,
        f_opt=nothing,
        f_best=nothing,
        f_target=nothing,
        get_f_target=nothing)

        M = new(β, δ, ρ, R)
        if f_opt !== nothing
            M.f_opt = f_opt
            M.get_f_target = M -> M.f_opt
        elseif get_f_target !== nothing
            M.get_f_target = get_f_target
        else
            M.get_f_target = M -> M.f_target-δ
        end
        M.params = Dict(
            :β=>[0.0, 2.0],
            :δ=>[0.0, Inf],
            :ρ=>[0.0, 1.0],
            :R=>[0.0, Inf])
        M
    end
end
function init!(M::TargetLevelPolyakStepSize, f, ∂f, x)
    M.i = 0
    M.r = 0.0
    M.f_best = M.f_target = f(x)
    M
end
function step!(M::TargetLevelPolyakStepSize, f, ∂f, x)
    M.i += 1

    f_val, g = f(x), ∂f(x)

    α = M.β*(f_val-M.get_f_target(M))
    if f_val ≤ M.f_target - M.δ/2
        M.f_target = M.f_best
    elseif M.r > M.R
        M.δ = M.δ*M.ρ
        M.r = 0.0
    else
        M.r += α*norm(g)
    end
    M.f_best = min(f_val, M.f_best)
    M
end

abstract type DualSubgradientMethod <: SubgradientMethod end

# Ergodic sequences of primal iterates have guarantee of convergence
# toward the primal space corresponding to the optimal dual
# In general
#       x̅ᵗ = ∑ᵗ⁻¹ ηₛᵗxˢ    with  ∑ηₛᵗ = 1  and  ηₛᵗ ≥ 0
# and defining
#       γₛᵗ = ηₛᵗ / αₛ
# with the assumptions
#   γₛ₋₁ᵗ ≤ γₛᵗ
#   Δγₘₐₓᵗ → 0
#   γ₀ᵗ → 0  and  ∃ Γ > 0 :  γₜ₋₁ᵗ ≤ Γ
# it is possible to prove the summentioned convergence in primal space.
# ---------------------------------------------------------------------
# Specific instances:
# (*) Harmonic + sᵏ-rule
# αₜ = 1 / (a + b*t)
# ηₛᵗ = (s+1)ᵏ / ∑ᵗ⁻¹(l+1)ᵏ
# Cit.
# 1) "Primal convergence from dual subgradient methods for convex optimization"
#       Emil Gustavsson, Michael Patriksson, Ann-Brith Stromberg
# 2) "Ergodic, primal convergence in dual subgradient schemes
#        for convex programming, II: the case of inconsistent primal problems"
#     Magnus Onnheim, Emil Gustavsson, Ann-Brith Stromberg,
#       Michael Patriksson, Torbjorn Larsson
mutable struct HarmonicErgodicPrimalStep <: DualSubgradientMethod
    k       # exponent, 4 seems nice
    a       # a in α = 1.0 / (a + bt)
    b       # b in ^

    x̅      # primal convex combination
    t      # iteration counter
    Sₜ     # ∑ᵗ⁻¹ (l+1)ᵏ
    Sₜ₋₁   # ∑ᵗ⁻¹ lᵏ
    params
    HarmonicErgodicPrimalStep(; k=0, a=nothing, b=nothing) = begin
        M = new()
        M.k = k
        @some M.a = a
        @some M.b = b
        M.params = Dict(:k => [0, 10], :a => [0.0, Inf], :b => [0.0, Inf])
        M
    end
end
function init!(M::HarmonicErgodicPrimalStep, get_x, is_le, L, ∂L, u)
    M.t = 0
    M.Sₜ, M.Sₜ₋₁ = 0.0, 0.0
    M.x̅ = get_x(u)
end
# @Input:
# ∂L : this is a dual specific method (so do not pass with minus sign e.g. -∂L)
function step!(M::HarmonicErgodicPrimalStep, get_x, is_le, L, ∂L, u)
    tᵏ = (M.t+=1)^M.k

    M.Sₜ₋₁, M.Sₜ = M.Sₜ, M.Sₜ+tᵏ
    x = get_x(u)
    M.x̅ = (M.Sₜ₋₁*M.x̅ + tᵏ*x) / M.Sₜ

    α = 1.0 / (M.a + M.b*M.t)
    d = ∂L(x, u)
    u = u + α*d
    u[is_le] = max.(u[is_le], 0.0)
    (u, α, d)
end

abstract type DeflectedSubgradientMethod <: SubgradientMethod end
mutable struct PolyakEllipStepSize <: DeflectedSubgradientMethod
    gen_γ
    γ_mul       # redundant factor to be used in parameter search
    ϵ

    i
    B
    g
    f_best
    f_opt
    params
    PolyakEllipStepSize(;f_opt=nothing, gen_γ=nothing, ϵ=1e-14, γ_mul=1.0) = begin
        M = new()
        M.ϵ = ϵ
        @some M.gen_γ = gen_γ
        if f_opt !== nothing
            M.f_opt = () -> f_opt
        else
            M.f_opt = () -> M.f_best - M.gen_γ(M.i+=1)
        end
        M.params = Dict(:γ_mul => [0.0, Inf])
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

# For β=0, the next one is Polyak
mutable struct FilteredPolyakStepSize <: DeflectedSubgradientMethod
    β           # β ∈ [0, 1] memory
    f_opt       # optimum objective value if available
    gen_γ       # estimated error in objective
    γ_mul       # redundant factor for parameter search

    i           # iteration counter, needed for γ
    f_best      # useful when f_opt should be estimated
    gen_α       # Filtered Polyak step size generator
    s           # filtered subgradient
    params
    FilteredPolyakStepSize(; f_opt=nothing, gen_γ=nothing, β=nothing, γ_mul=1.0) = begin
        M = new()
        @some M.f_opt = f_opt
        @some M.gen_γ = gen_γ
        @some M.β = β
        M.γ_mul = γ_mul

        if f_opt !== nothing
            M.gen_α = (f, s) -> (f-M.f_opt) / (s's)
        elseif gen_γ !== nothing
            M.gen_α = (f, s) -> begin
                M.f_best = min(M.f_best, f)
                (M.gen_γ(M.i+=1)*M.γ_mul |>
                γ -> (f-M.f_best+γ) / (s's))
            end
        end
        M.params = Dict(:γ_mul => [0.0, Inf], :β => [0.0, 1.0])
        M
    end
end
function init!(M::FilteredPolyakStepSize, f, ∂f, x)
    M.f_best = Inf
    M.i = 0
    M.s = ∂f(x)
end
function step!(M::FilteredPolyakStepSize, f, ∂f, x)
    sg, s, β, gen_α = ∂f(x), M.s, M.β, M.gen_α

    s[:] = (1-β)*sg + β*s
    α = gen_α(f(x), s)

    (x-α*s, α, s)
end

# TODO Camerini, Fratta, Maffioli
mutable struct CFMStepSize <: DeflectedSubgradientMethod
    get_γ   # γ ∈ [0, 2] - suggested value 1.5

end
function init!(M::CFMStepSize, f, ∂f, x)
end
function step!(M::CFMStepSize, f, ∂f, x)

end

mutable struct Adagrad <: DeflectedSubgradientMethod
    α # learning rate, typical 0.01 → 1.0
    ϵ # small value, typical 1e-8

    s # sum of squared gradient
    params
    Adagrad(;α=nothing, ϵ=1e-14) = begin
        M = new()
        @some M.α = α
        M.ϵ = ϵ
        M.params = Dict(:α => [0.0, 1.0])
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
    params
end
function init!(M::AdagradFull, f, ∂f, x)

end
function step!(M::AdagradFull, f, ∂f, x)

end

mutable struct NesterovMomentum <: DeflectedSubgradientMethod
    α # learning rate
    β # momentum decay

    v # momentum
    params
    NesterovMomentum(;α=nothing, β=nothing) = begin
        M = new()
        @some M.α = α
        @some M.β = β
        M.params = Dict(:α => [0.0, 1.0], :β => [0.0, 1.0])
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
    γ # decay, e.g. 0.99999, better 0.9999999 (1-1e-8)
    ϵ # small value

    s # sum of squared gradient
    params
    RMSProp(; α=nothing, γ=nothing, ϵ=1e-14) = begin
        M = new()
        @some M.α = α
        @some M.γ = γ
        M.ϵ = ϵ
        M.params = Dict(:α => [0.0, 1.0], :γ => [0.0, 1.0])
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
    u # sum of squared gradients
    params
    Adadelta(;γs=nothing, γx=nothing, ϵ=1e-8) = begin
        M = new()
        @some M.γs = γs
        @some M.γx = γx
        M.ϵ = ϵ
        M.params = Dict(:γs => [0.0, 1.0], :γx => [0.0, 1.0])
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
    params
    Adam(;α=nothing, γv=nothing, γs=nothing, ϵ=1e-14) = begin
        M = new()
        @some M.α = α
        @some M.γv = γv
        @some M.γs = γs
        M.ϵ = ϵ
        M.params = Dict(:α => [0.0, 1.0], :γv => [0.0, 1.0], :γs => [0.0, 1.0])
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
    params
    HyperGradient(; α₀=nothing, μ=nothing) = begin
        M = new()
        @some M.α₀ = α₀
        @some M.μ = μ
        M.params = Dict(:α₀ => [0.0, 1.0], :μ => [0.0, 1.0])
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
    α₀ # initial learning rate
    µ # learning rate of the learning rate
    β # momentum decay

    v # momentum
    α # current learning rate
    g_prev # previous gradient
    params
    HyperNesterovMomentum(;α₀=nothing, μ=nothing, β=nothing) = begin
        M = new()
        @some M.α₀ = α₀
        @some M.μ = μ
        @some M.β = β
        M.params = Dict(:α₀ => [0.0, 1.0], :μ => [0.0, 1.0], :β => [0.0, 1.0])
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
    params
    Noisy(;submethod=nothing, σ=nothing) = begin
        M = new()
        @some M.submethod = submethod
        @some M.σ = σ
        M.params = Dict(:σ => [0.0, Inf])
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
