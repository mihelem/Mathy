include("benchmark.jl")

m, n, singular, active = 100, 400, 130, 90
problem = generate_quadratic_min_cost_flow_boxed_problem(Float64, m, n; singular=singular, active=active)

raw_results = Dict{DataType, Tuple}()

open("exp.csv", "a") do io
    raw_results[Subgradient.NesterovMomentum] =
        run_bench(
            io,
            Float64,
            Subgradient.NesterovMomentum,
            problem,
            singular,
            active;
            todos=Set{String}(["dual", "min_grad", "EK", "mg_EK", "SPEK", "mg_SPEK", "SPEKn", "mg_SPEKn"]))
    raw_results[Subgradient.FixedStepSize] =
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
            todos=Set{String}(["dual", "min_grad", "EK", "mg_EK", "SPEK", "mg_SPEK", "SPEKn", "mg_SPEKn"]))
end

# possible heuristics:
# min_grad, EK, mg_EK, SPEK, mg_SPEK, SPEKn, mgSPEKn
heus = ["min_grad", "EK", "mg_EK", "SPEK", "mg_SPEK", "SPEKn", "mg_SPEKn"]
mutable struct Result
    ϵᵣ
    Δϵᵣ
end
function ϵs(problem::QMCFBProblem, result::BenchResult, heuristic::String)
    m, n = size(problem.E)

    f_dual = result.f_dual
    f_heu = getfield(result, Symbol("f_"*heuristic))
    unf_heu_1 = getfield(result, Symbol("unf_"*heuristic*"_1"))
    df_heu =  getfield(result, Symbol("df_"*heuristic))

    Result(
        (f_heu-f_dual) / abs(f_heu),            # ϵᵣₑₗ
        n * unf_heu_1 * df_heu / abs(f_heu)     # Δϵᵣₑₗ
    )
end

results = Dict{DataType, Dict{String, Result}}()
for (key, val) in raw_results
    results[key] = Dict{String, Result}()
    for heu in heus
        results[key][heu] = ϵs(problem, val[1], heu)
    end
end
