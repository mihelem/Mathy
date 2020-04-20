using LinearAlgebra

function GMRES_naive(A, b, k, ϵ, ϵₐ)
    m, n = size(A)

    H = UpperHessenberg(zeros(eltype(A), (k+1, k)))
    h = i -> view(H, 1:i+1, i)
    Q = zeros(eltype(A), (m, k+1))
    q = i -> view(Q, :, i)

    Γ = zeros(eltype(A), (2, k))
    γ = i -> view(Γ, :, i)
    T = copy(H)
    t = i -> view(T, 1:i, i)

    𝖇 = sqrt(b'*b)
    q(1)[:] = b / 𝖇

    # intermediate solutions
    y₀ = Vector{eltype(A)}(undef, 0)
    y₁ = copy(y₀)
    # y₂ = copy(y₀)
    # y₃ = copy(y₀)

    # error calculation
    y₁₁ = copy(y₀)
    H̃ᵢy = copy(y₀)
    rᵢ = copy(y₀)
    𝖗ᵢ::eltype(A) = 0

    y = copy(y₀)

    Λ = copy(y₀)
    for i=1:k
        # Arnoldi iteration
        q(i+1)[:] = A*q(i)
        for j=1:i
            H[j, i] = q(j)'*q(i+1)
            q(i+1)[:] -= q(j)*H[j, i]
        end
        H[i+1, i] = sqrt(q(i+1)'*q(i+1)) #todo: check ϵₐ
        q(i+1)[:] /= H[i+1, i]

        # Hessenberg → UpperTriangular via Householder
        T[1:i+1, i] = H[1:i+1, i]
        for j=1:i-1
            T[j:j+1, i] -= 2γ(j) * (γ(j)'*T[j:j+1, i])
        end
        𝖙ᵢ = sign(T[i, i]) * sqrt(T[i:i+1, i]'*T[i:i+1, i])
        γ(i)[:] = [T[i, i] + 𝖙ᵢ, T[i+1, i]]
        γ(i)[:] /= sqrt(γ(i)'*γ(i))
        T[i, i] = -𝖙ᵢ

        Λ = [Λ; T[i, i]]
        T[1:i, i] ./= Λ

        
        # update solution via pseudoinversion
        y₀ = [y₀; H[1, i]' * 𝖇]
        y₁ = [y₁; y₀[i] - T[1:i-1, i]'*y₁]
        
        # calculate error ||Axᵢ-b||₂ in O(i)
        y₁₁ = [y₁₁; y₁[i]/Λ[i]']
        H̃ᵢy = [y₁₁; 0]
        for l=i:-1:1
            H̃ᵢy[l:l+1] -= 2γ(l) * (γ(l)' * H̃ᵢy[l:l+1])
        end
        rᵢ = [H̃ᵢy[1]-𝖇; H̃ᵢy[2:end]]
        𝖗ᵢ = sqrt(rᵢ'*rᵢ)
        println(𝖗ᵢ)

        # BEGIN - for Testing Purposes
        # this is Σ O(i^2) = O(k^3), in final version calculate y₃ outside the 
        # for cycle with a single inversion of triangular, so O(k^2)
        # y₂ = y₁ ./ (transpose(Λ') .* Λ)
        # Δy₃ = zeros(eltype(Λ), size(y₂))
        # Δy₃[end] = y₂[end]
        # for j=i:-1:1
        #     Δy₃[1:j-1] -= T[1:j-1, j] * Δy₃[j]
        # end
        # y₃ = [y₃; 0] + Δy₃
        # END - for Testing Purposes

        if 𝖗ᵢ < ϵ*𝖇
            break
        end
    end

    y = y₁ ./ (transpose(Λ') .* Λ)
    for i=size(y)[1]-1:-1:1
        y[1:i] -= T[1:i, i+1] * y[i+1]
    end

    return (H, Q, Γ, UpperTriangular(T[1:end-1, :]), Λ, y)
end