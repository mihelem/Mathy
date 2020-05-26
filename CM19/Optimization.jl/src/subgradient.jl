module Subgradient

using LinearAlgebra
using ..Optimization

import ..Optimization.Descent: init!, step!
export  SubgradientMethod, init!, step!

# Subgradient methods
abstract type SubgradientMethod end

struct FixedStepSize <: SubgradientMethod
    α
end
init!(M::FixedStepSize, f, ∂f, x) = M
function step!(M::FixedStepSize, f, ∂f, x)
    α, sg = M.α, ∂f(x)
    return x - α*sg
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
    return x - α*sg
end

struct FixedStepLength <: SubgradientMethod
    γ
end
init!(M::FixedStepLength, f, ∂f, x) = M
function step!(M::FixedStepLength, f, ∂f, x)
    γ, sg = M.γ, ∂f(x)
    return x - γ*sg/norm(sg)
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
    return x - γ*sg/norm(sg)
end

end     # end module Subgradient
