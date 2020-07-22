include("benchmark.jl")
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

# Full-fledged comparison between subgradient methods and heuristics
ms, ns, ss, as = Int64[], [], [], []
for i in 1:4
    push!(ms, rand(50:1000))
    push!(ns, rand())
    push!(ss, rand())
    push!(as, rand())
    sort!(ms)
    sort!(ns)
    sort!(ss)
    sort!(as)
end

open("20200723NMvsFSS_res.csv", "a") do iore
    open("20200723NMvsFSS.csv", "a") do io
        for m in ms
            for dn in ns
                n = convert(Int64, m*(1 + floor(dn/2 * (m-3))))
                for ds in ss
                    singular = convert(Int64, floor(ds*n))
                    for da in as
                        active = convert(Int64, floor(da*n))
                        problem = generate_quadratic_min_cost_flow_boxed_problem(Float64, m, n; singular=singular, active=active)
                        results = Dict{DataType, Dict{String, Result}}()
                        raw_results = Dict{DataType, Tuple}()
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
                        print(iore, "nodes:$m arcs:$n singular:$singular active:$active ")
                        for (key, val) in raw_results
                            results[key] = Dict{String, Result}()
                            for heu in heus
                                results[key][heu] = ϵs(problem, val[1], heu)
                                print(iore, string(key), ':', heu, ':', results[key][heu], ' ')
                            end
                        end
                        println(iore)
                    end
                end
            end
        end
    end
end
