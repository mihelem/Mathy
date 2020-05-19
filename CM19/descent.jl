using LinearAlgebra
# Line line_search

function bracket_minimum(f, x=0; s=1e-2, k=2.0)
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya
        a, ya, b, yb = b, yb, a, ya
        s = -s
    end
    while true
        c, yc = b + s, f(b + s)
        if yc > yb
            return a < c ? (a, c) : (c, a)
        end
        a, ya, b, yb = b, yb, c, yc
        s *= k
    end
end

function get_fibonaccis(n::Int)
    v = zeros(Int64, n)
    v[1:2] = [1, 1]
    for i = 3:n
        v[i] = v[i-1] + v[i-2]
    end
    return v
end

function fibonacci_search(f, a, b, n; ϵ=0.01)
    F = get_fibonaccis(n+1)
    ρ::Float64 = convert(Float64, F[n])/convert(Float64, F[n+1])
    # φ = (1+√5)/2
    # s = (1-√5)/(1+√5)
    # ρ = (1-s^n) / (φ*(1-s^(n+1)))

    d = (1-ρ)*a + ρ*b
    yd = f(d)
    for i = n:-1:4
        c = ρ*a + (1-ρ)*b
        yc = f(c)
        if yc < yd
            d, b, yd = c, d, yc
        else 
            a, b = b, c
        end
        ρ = convert(Float64, F[i-1])/convert(Float64, F[i])
        #ρ = (1-s^(i-1)) / (φ*(1-s^i))
    end
    c = ϵ*a + (1-ϵ)*d
    yc = f(c)
    if yc < yd
        b, d, yd = d, c, yc
    else
        b, a = c, b
    end 

    return a < b ? [a, b] : [b, a]
end

function fibonacci_as_power_search(f, a, b, n; ϵ=0.01)
    φ = (1+√5)/2
    s = (1-√5)/(1+√5)
    ρ = (1-s^(n-1)) / (φ*(1-s^n))
    d = ρ*b + (1-ρ)*a
    yd = f(d)
    for i = 1 : n-1
        if i == n-1
            c = ϵ*a + (1-ϵ)*d
        else
            c = ρ*a + (1-ρ)*b
        end
        yc = f(c)
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
        ρ = 1 / (φ*(1-s^(n-i+1))/(1-s^(n-i)))
    end
    return a < b ? [a, b] : [b, a]
end

function golden_section_search(f, a, b, n)
    φ = (1+√5)/2
    ρ = φ-1
    d = ρ*b + (1-ρ)*a
    yd = f(d)
    for i = 1:n-1
        c = ρ*a + (1-ρ)*b
        yc = f(c)
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
    end
    return a < b ? [a, b] : [b, a]
end

function line_search(f, x, d)
    objective = α -> f(x + α*d)
    a, b = bracket_minimum(objective)
   # α = minimize(objective, a, b)
   # return x + α*d
end

# Zeroth Order Methods

# First Order Methods
abstract type DescentMethod end

mutable struct GradientDescent <: DescentMethod
    α

    GradientDescent() = new(0.1)
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

    ConjugateGradientDescent() = new()
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

    NesterovMomentumDescent() = new(0.5, 0.1)
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

    AdagradDescent() = new(0.01, 1e-8)
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

    RMSPropDescent() = new(0.01, 0.9, 1e-8, )
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

    AdadeltaDescent() = new(0.9, 0.9, 1e-8)
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

    AdamDescent() = new(0.001, 0.9, 0.999, 1e-8)
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
function step!(M::NoisyDescent, f, ∇f, x)
    x = step!(M.submethod, f, ∇f, x)
    σ = M.σ(M.k)
    x += σ.*randn(length(x))
    M.k += 1
    return x
end