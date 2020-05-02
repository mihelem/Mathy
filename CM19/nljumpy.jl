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