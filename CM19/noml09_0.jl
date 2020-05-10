using LinearAlgebra
using SparseArrays

# Problem description
# minₓ { ½xᵀQx + qᵀx  with  x s.t.  Ex = b  &  l ≤ x ≤ u }
# Q ∈ { diag ≥ 0 }
struct problem
    Q
    q
    l
    u
    E
    b
end

function solver_D1(;P::problem, μ, ϵ)
    Q, q, l, u, E, b = P.Q, P.q, P.l, P.u, P.E, P.b

    struct state
        x
        μ
    end

    function find_αs(p::state, d)

    end

    function exact_line_search(p::state, d)
    end
end

function solver_D2(;P::problem, ν, ϵ)

end

function solver_D3(;P::problem, λ, ϵ)

end