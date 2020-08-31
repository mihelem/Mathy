include("benchmark.jl")
using Parameters, Optimization

heus = ["min_grad", "EK", "mg_EK", "SPEK", "mg_SPEK", "SPEKn", "mg_SPEKn"]
todos = ["dual", "min_grad", "mg_EK", "mg_SPEK"]
mutable struct ϵResult
    ϵᵣ::Number    # relative error on objective value
    Δϵᵣ::Number     # error on relative error because of unfeasibility

    function ϵResult(ϵ::Number, Δϵ::Number)
        new(ϵ, Δϵ)
    end
    function ϵResult(problem::QMCFBProblem, result::BenchResult, heuristic::String)
        m, n = size(problem.E)

        f_dual = result.f_dual
        f_heu = getfield(result, Symbol("f_"*heuristic))
        unf_heu_1 = getfield(result, Symbol("unf_"*heuristic*"_1"))
        df_heu =  getfield(result, Symbol("df_"*heuristic))

        new(
            (f_heu-f_dual) / abs(f_heu),            # ϵᵣₑₗ
            n * unf_heu_1 * df_heu / abs(f_heu)     # Δϵᵣₑₗ
        )
    end
end

mutable struct TestResult{V}
    bench::BenchResult{V}
    is
    Ls
    ϵs::Dict{String, ϵResult}
end

function get_active(problem::QMCFBProblem)
    @unpack Q, q, l, u = problem
    (-q .< Q*l) .| (-q .> Q*u)
end

function get_singular(problem::QMCFBProblem)
    @unpack Q = problem
    Q╲ = view(Q, [CartesianIndex(i, i) for i in 1:size(Q, 1)])
    Q╲ .== 0
end

function get_ϵs(bench::BenchResult)

end

function run_test(
    out_filename::String,
    ::Type{V},
    ::Type{SG},
    problem::QMCFBProblem,
    sg_args::Tuple=(),
    sg_kwargs::NamedTuple=(α=1.0, β=0.99),
    sg_update=sg->sg.α/=2.0;
    μ₀=nothing,
    max_iter::Int64=4000,
    max_hiter::Int64=40,
    restart::Bool=true,
    todos::Set=Set{String}(["dual", "min_grad", "mg_EK", "mg_SPEK", "mg_SPEKn",
        "JuMP", "JuMP_EK", "JuMP_SPEK", "JuMP_SPEKn", "time", "plot"])) where {V, SG<:SubgradientMethod}

    @unpack Q, q, l, u, E, b = problem
    𝔨 = get_singular(problem)
    singular, active = count(𝔨), count(get_active(problem))
    m, n = size(E)
    μ₀ = μ₀ === nothing ? rand(size(problem.E, 1)) : μ₀
    raw_results = Dict{DataType, Tuple}()

    open(out_filename*".bench", "w") do io
        raw_results[SG] =
            run_bench(
                io,
                V,
                SG,
                problem,
                singular,
                active,
                sg_args,
                sg_kwargs,
                sg_update;
                μ₀=μ₀,
                max_iter=max_iter,
                max_hiter=max_hiter,
                todos=todos)
    end
    ϵs = Dict{String, ϵResult}()
    for heu in heus
        ϵs[heu] = ϵResult(problem, raw_results[SG][1], heu)
    end
    TestResult(raw_results[SG]..., ϵs)
end
