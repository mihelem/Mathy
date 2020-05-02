using LinearAlgebra

function quadratic_problem_generator( ecc )
end

# Naive Implementation
# Assumptions:
#  Everything invertible, Q positive definite

function solve_reduced_KKT(Q, q, A, b, x, ϵ)
    Q˜ = inv(Q)
    M = A*Q˜*A'
    μ̅ = -inv(M)*(b + A*Q˜*q)
    x̅ = -Q˜*(q+A'*μ̅)

    return (x̅, μ̅)
end

function active_set_method_quadratic(Q, q, A, b, x, ϵ)
    B = zeros(Bool, size(A, 1))
    while true
        x̅, μ̅  = solve_reduced_KKT(Q, q, A[B, :], b[B], x[:], ϵ)
        C = .!B .& (A*x̅ .> b)
        println(B)
        if any(C)
            d = x̅-x
            α = minimum( (b[C]-A[C, :]*x) ./ (A[C, :]*d) )
            x = x + α*d
            B = A*x .== b
        else
            μ̅ .< 0
            O = findall(x -> x, B)[μ̅ .< 0]
            if isempty(O)
                return x̅
            else
                B[O[1]] = false
            end
        end
    end
end

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
function step!(M::ConjugateGradientDescent, f, ∇f, x)
    d, g = M.d, M.g
    g´ = ∇f(x)
end