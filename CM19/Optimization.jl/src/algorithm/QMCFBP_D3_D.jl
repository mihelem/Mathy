"""
# ---------------------------- Dual algorithm D3 -----------------------------
                 WIP: really, just copy pasted from old commit!
# Null Space method + Box Constraints dualised
"""
mutable struct QMCFBPAlgorithmD3D <: OptimizationAlgorithm{QMCFBProblem}
end
function set!(algorithm::QMCFBPAlgorithmD3D, 𝔓::QMCFBProblem)
end
function run!(algo::QMCFBPAlgorithmD3D, 𝔓::QMCFBProblem)
    @unpack Q, q, l, u, E, b = 𝔓
    ϵ = 1e-8 # todo all

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
    ∇L₁ = q_N - Ẽ_BE_N' * (q_B + Q_B*Ẽ_Bb)
    L₀ = 0.5 * Ẽ_Bb'Q_B*Ẽ_Bb + q_B'Ẽ_Bb




    function test()
        return split_eq_constraint(ϵ)
    end

    return test()
end
