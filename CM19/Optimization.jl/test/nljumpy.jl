# solve with JuMP, to double check our implementation

using JuMP
using Ipopt
# using GLPK

function get_solution_quadratic(Q, q, A, b, x₀)
    model = Model()
    set_optimizer(model, Ipopt.Optimizer)

    @variable(model, x[1:size(x₀, 1)])
    @objective(model, Min, 0.5*x'*Q*x + q'*x)
    objective_function(model, QuadExpr)
    @constraint(model, con, A*x .<= b )

    optimize!(model)

    return value.(x)
end

function check_nonempty_constraints(𝔓, x₀)
    Q, q, l, u, E, b = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b)
    model = Model()
    set_optimizer(model, Ipopt.Optimizer)

    @variable(model, x[1:size(x₀, 1)])
    @objective(model, Min, (E*x-b)'*(E*x-b))
    objective_function(model, QuadExpr)
    #@constraint(model, con, E*x .== b )
    @constraint(model, con2, l .<= x .<= u)

    optimize!(model)

    return value.(x)
end

function get_solution_quadratic_box_constrained(𝔓, x₀)
    Q, q, l, u, E, b = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b)
    model = Model()
    set_optimizer(model, Ipopt.Optimizer)

    @variable(model, x[1:size(x₀, 1)])
    @objective(model, Min, 0.5*x'*Q*x + q'*x)
    objective_function(model, QuadExpr)
    @constraint(model, con, E*x .== b)
    @constraint(model, con2, l .<= x .<= u)

    optimize!(model)

    return value.(x)
end


"""
Examples:
```julia
subgradient.α=1.0;  # Adagrad + RMSProp + NesterovMomentum
#subgradient.γ=0.9;  # RMSProp
algorithm.max_iter = 4000;
algorithm.μ₀ = rand(1000);
Ls = [];
is = [];
for i in 1:40
    run!(test);
    push!(Ls, test.result.memoria["L′"]...);
    push!(is,((i-1)*algorithm.max_iter .+ test.result.memoria["i′"])...);
    set!(algorithm, test.result);
    algorithm.μ₀ = test.result.result["μ′"]
    algorithm.stopped = false;
    subgradient.α /= 2.0    # Adagrad + RMSProp
    #subgradient.γ = 1.0-(1.0-subgradient.γ)/4.0
end

for i in 1:25
    run!(test);
    push!(Ls, test.result.memoria["L′"]...);
    push!(is,((i-1)*algorithm.max_iter .+ test.result.memoria["i′"])...);
    set!(algorithm, test.result);
    algorithm.μ₀ = test.result.result["μ′"]
    algorithm.stopped = false;
    subgradient.α /= 3.0    # Adagrad + RMSProp
    subgradient.γ = 1.0-(1.0-subgradient.γ)/3.0
end
```

"""
