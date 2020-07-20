include("benchmark.jl")

m, n, singular, active = 100, 400, 130, 90
problem = generate_quadratic_min_cost_flow_boxed_problem(Float64, m, n; singular=singular, active=active)

results = Dict()

open("exp.csv", "a") do io
    results["nm"] =
        run_bench(
            io,
            Float64,
            Subgradient.NesterovMomentum,
            problem,
            singular,
            active;
            todos=Set{String}(["dual", "min_grad", "mg_EK", "mg_SPEK", "mg_SPEKn"]))
    results["fss"] =
        run_bench(
            io,
            Float64,
            Subgradient.FixedStepSize,
            problem,
            singular,
            active,
            Tuple(1.0),
            NamedTuple();
            max_iter=8000,
            todos=Set{String}(["dual", "min_grad", "mg_EK", "mg_SPEK", "mg_SPEKn"]))
end

ϵᵣ = r -> (r.f_mg_SPEKn - r.f_dual) / abs(r.f_mg_SPEKn)
ϵᵣ(results["nm"])
ϵᵣ(results["fss"])

# distanza da feasible: n |Δb|₁
# errore su f:   |Qx+q|*^
ϵᵤ = r -> n*r.unf_mg_SPEKn_1*r.df_mg_SPEKn/abs(r.f_mg_SPEKn)
ϵᵤ(results["nm"])
ϵᵤ(results["fss"])
