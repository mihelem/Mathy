module Subgradient

using LinearAlgebra
using ..Optimization
using ..Optimization.Utils

import ..Optimization.Descent: init!, step!
export  SubgradientMethod, init!, step!

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

end     # end module Subgradient
