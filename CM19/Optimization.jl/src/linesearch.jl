module LineSearch

export bracket_minimum, fibonacci_search, fibonacci_as_power_search, golden_section_search, line_search

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

end     # end module LineSearch