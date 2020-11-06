# Run some benchmarks

include("benchmark.jl")

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
for i in 1:1
    push!(ms, rand(100:100))
    push!(ns, rand())
    push!(ss, rand())
    push!(as, rand())
end
sort!(ms)
sort!(ns)
sort!(ss)
sort!(as)
ss=[0.5]
as=[0.87]

problem = generate_quadratic_min_cost_flow_boxed_problem(Float64, 100, 1100; singular=550, active=937)
push!(problems, problem)

problems = QMCFBProblem[]
all_results = Dict{DataType, Dict{String, Result}}[]
all_raw_results = Dict{DataType, Tuple}[]

μ₀ = rand(100)
function doit(max_iter, all::Bool; rnm=true, rsg=true)
    open("test00_res.csv", "a") do iore
        open("test00.csv", "a") do io
            for m in ms
                for dn in ns
                    n = convert(Int64, m*(1 + floor(dn/2 * (m-3))))
                    for ds in ss
                        singular = convert(Int64, floor(ds*n))
                        for da in as
                            active = convert(Int64, floor(da*n))
                            #problem = generate_quadratic_min_cost_flow_boxed_problem(Float64, m, n; singular=singular, active=active)
                            #push!(problems, problem)
                            problem = problems[end]
                            results = Dict{DataType, Dict{String, Result}}()
                            raw_results = Dict{DataType, Tuple}()
                            if rnm
                                raw_results[Subgradient.NesterovMomentum] =
                                    run_bench(
                                        io,
                                        Float64,
                                        Subgradient.NesterovMomentum,
                                        problem,
                                        singular,
                                        active;
                                        max_iter=max_iter ÷ 2,
                                        max_hiter=40,
                                        todos=all ?
                                            Set{String}(["dual",
                                                "min_grad",
                                                "mg_EK", "EK",
                                                "mg_SPEK", "SPEK",
                                                "mg_SPEKn", "SPEKn"#,
                                                #="plot"=#]) :
                                            Set{String}(["dual"]))#,
                                                #"min_grad",
                                                #"mg_EK",
                                                #"mg_SPEK",
                                                #"mg_SPEKn",
                                                #"plot"]))
                            end

                            if rsg
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
                                        μ₀=μ₀,
                                        max_iter=max_iter,
                                        todos=all ?
                                            Set{String}(["dual",
                                                "min_grad",
                                                "mg_EK",
                                                "mg_SPEK",
                                                "mg_SPEKn"#,
                                                #="plot"=#]) :
                                            Set{String}(["dual"]))#,
                                                #"min_grad",
                                                #"mg_EK",
                                                #"mg_SPEK",
                                                #"mg_SPEKn",
                                                #"plot"]))
                            end
                            print(iore, "nodes:$m arcs:$n singular:$singular active:$active ")
                            for (key, val) in raw_results
                                results[key] = Dict{String, Result}()
                                for heu in heus
                                    results[key][heu] = ϵs(problem, val[1], heu)
                                    print(iore, string(key), ':', heu, ':', results[key][heu], ' ')
                                end
                            end
                            println(iore)
                            push!(all_results, results)
                            push!(all_raw_results, raw_results)
                        end
                    end
                end
            end
        end
    end
end

# using Plots
# pyplot()
using LaTeXStrings
max_iter = 1
doit(max_iter, false)
#=is, Ls = all_raw_results[end][Optimization.Subgradient.FixedStepSize][end-1],
    all_raw_results[end][Optimization.Subgradient.FixedStepSize][end]=#
L = all_raw_results[end][Optimization.Subgradient.NesterovMomentum][1].f_dual
#=p = plot(
        is/max_iter,
        (L .- Ls) ./ abs(L),
        title=L"RSG with $2^i$ iterations per stage",
        fillcolor = :lightgray,
        color=:black,
        yaxis=:log2,
        xlabel="stage",
        ylabel=L"\epsilon_{rel}",
        legend=false)=#
ts = [1:10; 15:5:95; 100:50:950; 1000:200:1800; 2000:2000:14000; 32000]
Ls = []
ts = [64000]
Ls_SGR, Ls_SNM = [], []
for t in ts
    doit(t, true; rsg=false)
    #push!(Ls_SNM, all_raw_results[end][Optimization.Subgradient.NesterovMomentum][1].f_dual)
    #push!(Ls_SGR, all_raw_results[end][Optimization.Subgradient.FixedStepSize][1].f_dual)
    #is, Ls = all_raw_results[end][Optimization.Subgradient.FixedStepSize][end-1],
    #    all_raw_results[end][Optimization.Subgradient.FixedStepSize][end]
    #plot!(p, is/t, (L .- Ls) ./ abs(L), yaxis=:log2, color=:black)
end

L = all_raw_results[end][Optimization.Subgradient.NesterovMomentum][1].f_dual
Ls = Dict{String, Array{Float64, 1}}()
for heu in heus
    Ls[heu] = Array{Float64, 1}()
end
for t in ts
    doit(t, true; rsg=false)
    for heu in heus
        res = getfield(
                all_raw_results[end][Optimization.Subgradient.NesterovMomentum][1],
                Symbol("f_"*heu))
        push!(Ls[heu], res)
    end
    #push!(Ls_SNM, all_raw_results[end][Optimization.Subgradient.NesterovMomentum][1].f_dual)
    #push!(Ls_SGR, all_raw_results[end][Optimization.Subgradient.FixedStepSize][1].f_dual)
    #is, Ls = all_raw_results[end][Optimization.Subgradient.FixedStepSize][end-1],
    #    all_raw_results[end][Optimization.Subgradient.FixedStepSize][end]
    #plot!(p, is/t, (L .- Ls) ./ abs(L), yaxis=:log2, color=:black)
end
Ls
ts

j=0
good = .~(isnan.(Ls["SPEK"]) .| isnan.(Ls["mg_SPEK"]))
plot!(p,
        ts[good],
        -(L .- Ls["mg_SPEK"][good]) ./ abs(L),
        title=L"Projections~on~S_{(P)}",
        fillcolor = :lightgray,
        color=:black,
        yaxis=:log2,
        xaxis=:log2,
        label="mg_SPEK",
        xlabel="iterations per stage",
        linestyle=:dashdot,
        ylabel=L"\epsilon_{rel}",
        legend=true)
plot!(p,
        ts[good],
        -(L .- Ls["mg_SPEKn"][good]) ./ abs(L),
        title="SPEKn vs mg_SPEKn",
        fillcolor = :lightgray,
        linestyle=:dash,
        color=:black,
        yaxis=:log2,
        xaxis=:log2,
        label="mg_SPEKn",
        xlabel="iterations per stage",
        ylabel=L"\epsilon_{rel}",
        legend=true)
savefig("projections.png")

j=27
p = plot(
        ts[1:end-j],
        (L .- Ls_SGR[1:end-j]) ./ abs(L),
        title="RSG vs RNM",
        fillcolor = :lightgray,
        color=:black,
        yaxis=:log2,
        xaxis=:log2,
        label="RSG",
        xlabel="iterations per stage",
        ylabel=L"\epsilon_{rel}",
        legend=true)
j=30
plot!(p,
        ts[1:end-j],
        (L .- Ls_SNM[1:end-j]) ./ abs(L),
        title="RSG vs RNM",
        fillcolor = :lightgray,
        linestyle=:dash,
        color=:black,
        yaxis=:log2,
        xaxis=:log2,
        label="RNM",
        xlabel="iterations per stage",
        ylabel=L"\epsilon_{rel}",
        legend=true)
pyplot()
pgfplots()
display(p)
savefig("RSGvsRNM_iterperstage4sing0act0.png")

is, Ls = all_raw_results[end][Optimization.Subgradient.NesterovMomentum][end-1],
    all_raw_results[end][Optimization.Subgradient.NesterovMomentum][end]
plot!(p, 2*is, (L .- Ls) ./ abs(L), yaxis=:log2, label="NesterovMomentum Restarted, β=0.99")
savefig(p, "RSGvsRNM200.png")
