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