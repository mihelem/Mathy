using LinearAlgebra

# Naive implementations of Numerical Analysis algos

diff_forward(f, x; h=sqrt(eps(Float64))) = (f(x+h) - f(x))/h;
diff_central(f, x; h=cbrt(eps(Float64))) = (f(x+h/2) - f(x-h/2))/h;
diff_backward(f, x; h=sqrt(eps(Float64))) = (f(x) - f(x+h))/h;
diff_complex(f, x; h=1e-20) = imag(f(x + h*im)) / h;

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

function QR_gram_schmidt(A)
    m, n = size(A)
    Q = copy(A)
    R = zeros(eltype(A), (n, n))
    for i=1:n
        R[i, i] = sqrt(A[1:m, i]' * A[1:m, i])
        Q[1:m, i] = A[1:m, i] ./ R[i, i]
        for j=i+1:n
            R[i, j] = A[1:m, i]' * A[1:m, j]
            Q[1:m, j] -= Q[1:m, i] .* R[i, j]
        end
    end

    return (Q, R)
end

function hessenberg_gram_schmidt(A, b, n)
    m, _ = size(A)
    Q = zeros(eltype(A), (m, n+1))
    H = zeros(eltype(A), (n+1, n))

    v = b ./ sqrt(b' * b)
    Q[1:m, 1] = v 
    
    for i=1:n
        v = A*Q[1:m, i]
        for j=1:i
            H[j, i] = Q[1:m, j]'*v
            v -= H[j, i]*Q[1:m, j]
        end
        H[i+1, i] = sqrt(v' * v)
        Q[1:m, i+1] = v ./ H[i+1, i]
    end

    return (H, Q)
end

# only for square matrices by now
function gaussian_elimination(A)
    m, n = size(A)
    R = copy(A)
    L = zeros(eltype(A), (m, m))
    for i=1:m
        L[i, i] = 1
        for j=i+1:m
            L[j, i] = R[j, i] / R[i, i]
            R[j, i:n] -= L[j, i] * R[i, i:n]
        end
    end

    return (L, R)
end

function gaussian_elimination_row_pivot(A)
    m, n = size(A)
    R = copy(A)
    L = zeros((m, m))
    ϖ = [1:m;]
    println(ϖ)
    for i=1:m
        L[i, i] = 1
        k = i
        for j=i+1:m
            if abs(R[k, i]) < abs(R[j, i])
                k = j
            end
        end
        if i != k
            ϖ[i], ϖ[k] = ϖ[k], ϖ[i]
            R[i, i:n], R[k, i:n] = R[k, i:n], R[i, i:n]
        end
        if R[i, i] == 0
            continue
        end
        for j=i+1:m
            L[j, i] = R[j, i]/R[i, i]
            R[j, i:n] -= L[j, i]*R[i, i:n]
        end
    end
    return (L, R, ϖ)
end

function cholevski_factorisation(A)
    V = copy(A)
    m, n = size(A)
    for i=1:m
        V[i, i] = sqrt(max(V[i, i], 0))
        if V[i, i] == 0.
            V[i:end, i:end] .= 0.
            break
        end
        V[i+1:m, i] /= V[i, i]
        for j=i+1:m
            for k=i+1:j
                V[j, k] -= V[j, i]*V[k, i]'
            end
        end
    end
    for i=1:m
        for j=i+1:m
            V[i, j] = 0
        end
    end
    return V
end

# Only for square matrices
function hessenberg_via_householder(A)
    m = size(A)[1]
    Q = zeros(m-1, m-2)
    H = copy(A)
    for i=1:m-2
        α = sign(A[i+1, i]) * sqrt(A[i+1:m, i]'*A[i+1:m, i])
        Q[i, i] = A[i+1, i] + α; Q[i+1:m-1, i] = A[i+2:m, i]
        Q[i:m-1, i] /= sqrt(Q[i:m-1, i]'*Q[i:m-1, i]) # optional

        H[i+1, i] = -α; H[i+2:m, i] .= 0
        H[i+1:m, i+1:m] -= 2*Q[i:m-1, i]*Q[i:m-1, i]'*H[i+1:m, i+1:m]
        H[1:m, i+1:m] -= 2*H[1:m, i+1:m]*Q[i:m-1, i]*Q[i:m-1, i]'
    end

    return (Hessenberg(H), Q)
end

function rayleigh_iteration(A, v, n)
    λ = 0
    for i=1:n
        v = A*v
        v /= sqrt(v'*v)
        λ = v'*A*v
        println("$v -> $λ")
    end
    println(A*v-v*λ)

    return (v, λ)
end


function rayleigh_inverse_iteration(A, v, n, λ)
    m, m = size(A)
    for i=1:n
        v = (A-λ.*I(m))\v 
        v /= norm(v)
        λ = v'*A*v
        println("$v -> $λ")
    end

    return (v, λ)
end

# QR algorithm  :  TODO
function pure_QR_algorithm_iteration(A, qr)
    Q, R = qr(A)
    return R*Q
end

function QR_algorithm_iteration(A, ϵ)

end

function QR_algorithm(A::Hermitian, ϵ)

end

function Arnoldi_iterations(A, Q₀, H₀, k, ϵ)
    m, n = size(A)
    m, n₀ = size(Q₀)

    n₁ = n₀+k
    Q = zeros(eltype(Q₀), (m, n₁))
    Q[1:m, 1:n₀] = Q₀[:, :]

    H = zeros(eltype(H₀), (n₁, n₁-1))
    H[1:n₀, 1:n₀-1] = H₀[:, :]

    for i=n₀+1:n₁
        Q[1:m, i] = A*Q[1:m, i-1]
        for j=1:i-1
            H[j, i-1] = Q[1:m, j]'*Q[1:m, i]
            Q[1:m, i] -=  Q[1:m, j] .* H[j, i-1]
        end
        H[i, i-1] = sqrt(Q[1:m, i]'*Q[1:m, i])
        Q[1:m, i] ./= H[i, i-1]
        if H[i, i-1] < ϵ
            return (Q[1:m, 1:i-1], H[1:i, 1:i-1], i-n₀-1)
        end
    end
    return (Q, H, k)
end

function Arnoldi_naive(A, b, k₀, ϵ)
    m, n = size(A)
    Q = b
    H = zeros(eltype(A), (1, 0))
    while true
        Q, H, k = Arnoldi_iterations(A, Q, H, k₀, ϵ)
        println(eigvals(H[1:end-1, :])) 
        if k != k₀
            return (Q, H)
        end
    end
end

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

# handmade bidiagonal
# Requires big precision in floating-point operations
# A = V*J'*U'
using SparseArrays

function bidiagonal_decomposition_handmade(A, x₀, ϵᵤ = 0.)
    m, n = size(A)
    l = min(m, n)
    U = zeros(eltype(A), n, l+1)
    V = zeros(eltype(A), m, l)
    J = zeros(eltype(A), l+1, l)

    U[:, 1] = x₀ / norm(x₀)
    𝔲 = zeros(1)
    𝔳 = zeros(1)
    for i=1:l
        v = A*U[:, i]
        if i > 1
            v -= V[:, i-1]*(V[:, i-1]'*v)
        end
        𝔳 = [𝔳; norm(v)]
        V[:, i] = v/norm(v)

        u = A'*V[:,i]

        J[i, i] = U[:, i]'*u
        u -= U[:, i]*J[i, i]

        𝔲 = [𝔲; norm(u)]
        J[i+1, i] = norm(u)
        U[:, i+1] = u/J[i+1, i]
    end

    return (U, V, J, 𝔲, 𝔳)
end

# Unfortunately, the two-term iteration is not stable at all ...
# since vector are less and less orthogonal
function bidiagonal_decomposition_handmade2(A, x₀, ϵᵤ = 0.)
    m, n = size(A)
    l = min(m, n)
    U = zeros(eltype(A), n, l+1)
    V = zeros(eltype(A), m, l)
    J = spzeros(eltype(A), l+1, l)

    U[:, 1] = x₀ / norm(x₀)
    #𝔲 = zeros(1)
    #𝔳 = zeros(1)
    for i=1:l
        v = A*U[:, i]
        for j=1:i-1
            v -= V[:, j]*(V[:, j]'*v)
        end
        #𝔳 = [𝔳; norm(v)]
        V[:, i] = v/norm(v)

        u = A'*V[:,i]
        for j=1:i-1
            u -= U[:, j]*(U[:, j]'*u)
        end
        J[i, i] = U[:, i]'*u
        u -= U[:, i]*J[i, i]
        #𝔲 = [𝔲; norm(u)]
        J[i+1, i] = norm(u)
        U[:, i+1] = u/J[i+1, i]
    end

    return (U, V, J) #, 𝔲, 𝔳)
end

function test_bi(m, n, d=1., ϵ=1e-6)
    A = Float64.(rand(1:m*n, m, n))
    x = [1.; zeros(n-1)]
    
    function test(f)
        U, V, J = f(A, x)

        𝔄 = zeros(size(A'*A))
        𝔄[:, 1] = (A'*A)[:, 1]
        𝔄[:, 1] /= norm(𝔄[:, 1])
        for i=2:n
            𝔄[:, i] = A'*A*𝔄[:, i-1]
            for j=1:i-1
                𝔄[:, i] -= 𝔄[:, j]*(𝔄[:, j]')*𝔄[:, i]
            end
            # println(norm(𝔄[:, i]))
            𝔞 = norm(𝔄[:, i])
            𝔄[:, i] /= 𝔞
            if 𝔞 < ϵ
                break
            end
        end
        println("rank(A) = ", rank(𝔄))
        # println("𝔲 : ", 𝔲)
        # println("𝔳 : ", 𝔳)
        
        println("|A-V*J'*U'|/|A| = ", norm(A-V*J'*U')/norm(A))
        #m, n = size(J)
        #for i=1:n-2
        #    J[i, i+1:end] .= 0
        #end
        #println("|A-V*J̃'*U'|/|A| = ", norm(A-V*J'*U')/norm(A))
        println("|U'*U-I| = ", norm(U[:, 1:end-1]'*U[:, 1:end-1]-I(size(U[:, 1:end-1]'*U[:, 1:end-1], 1))))
        println("|V'*V-I| = ", norm(V'*V-I(size(V'*V, 1))))

        return U, V, J
    end

    # test(bidiagonal_decomposition_handmade)
    U, V, J =  test(bidiagonal_decomposition_handmade2)
    return (A, x, U, V, J)
end

