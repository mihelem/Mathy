using JuMP, Gurobi, Parameters

function solveQMCFBP(problem, x₀=nothing)
    @unpack Q, q, l, u, E, b = problem
    if x₀ === nothing
        x₀ = l + rand(size(E)[2]).*(u-l)
    end
    model = Model(Gurobi.Optimizer)

    @variable(model, x[1:size(x₀, 1)])
    @objective(model, Min, 0.5*x'*Q*x + q'*x)
    objective_function(model, QuadExpr)
    @constraint(model, con, E*x .== b)
    @constraint(model, con2, l .<= x .<= u)

    optimize!(model)

    return model
end

#=
# Example
using Optimization

problem = generate_quadratic_min_cost_flow_boxed_problem(Float64, 100, 1000; singular=500, active=600)
model = solveQMCFBP(problem)

objective_value(model)
=#
