include("testsetrun.jl")
include("dmacio.jl")
include("grb.jl")
using LaTeXStrings
using Printf
using Plots
using JLD
using HDF5
pyplot()
problems2 = parse_dir(NetgenDIMACS, "Optimization.jl/test/gen/set2/")

#=
    Testing subgradient with parameter search
=#

function test_with_param_search(
    problem;
    μ₀=nothing,
    localization,
    param_ranges,
    fixed_params=Dict(),
    searcher_iter,
    algorithm_iter,
    algorithm_iter_per_search,
    δ,
    restart_params=false)

    searcher = NelderMead()
    algorithm = QMCFBPAlgorithmD1SG(
        localization=localization,
        verbosity=1,
        max_iter=1000,
        μ₀=μ₀,
        ε=1e-6,
        ϵ=0.0)
    searcher = NelderMead()
    halgorithm = WithParameterSearch{QMCFBProblem, typeof(algorithm), NelderMead}(
        algorithm=algorithm,
        searcher=searcher,
        objective="L_best",
        cmp=(a, b)->a>b,
        searcher_iter=searcher_iter,
        algorithm_iter=algorithm_iter,
        algorithm_iter_per_search=algorithm_iter_per_search,
        δ=δ,
        fixed_params=fixed_params,
        param_ranges=param_ranges,
        restart_params=restart_params)
    instance = get_test(halgorithm, problem=problem)
    instance.solver.options.memoranda = Set(["L_best", "params_best", "result_best"])
    run!(instance)
    instance
end

problem = problems["netgen-1000-1-1-b-a-ns-0660"]
m, n = size(problem.E)
μ₀ = rand(Float64, 89)
ϵ = 1e-14
localization = Subgradient.Adam(; α=0.5, γv=1.0, γs=0.995)
param_ranges=Dict(:α => [0.3, 0.9], :γv => [0.5, 0.8], :γs => [0.5, 0.8])
fixed_params=Dict()
searcher_iter = 200
algorithm_iter = 8000
algorithm_iter_per_search = 400
δ = 1.0
restart_params = true
instance = test_with_param_search(problem;
    μ₀=μ₀,
    localization=localization,
    param_ranges=param_ranges,
    fixed_params=fixed_params,
    searcher_iter=searcher_iter,
    algorithm_iter=algorithm_iter,
    algorithm_iter_per_search=algorithm_iter_per_search,
    δ=δ,
    restart_params=restart_params)

pyplot()
function get_par(par::Symbol, instance)
    pars = []
    for r in instance.result.memoria["params_best"]
        push!(pars, r[par])
    end
    pars
end
αs = get_par(:α, instance)
#γss = max.(min.(γss, 1.0), 0.0)
αs .+= 1e-16
count((αs.<=0))
maximum(αs)
minimum(αs)
argmin(αs)
αs[2607] = 1e-16
αs[αs .< 0.0] .= 1e-16
#βs = get_par(:β, instance)
#βs = max.(min.(βs, 1.0), 0.0)
γss = get_par(:γs, instance)
γss = max.(min.(γss, 1.0), 0)

γvs = get_par(:γv, instance)
γvs = max.(min.(γvs, 1.0), 0.0)
plot(αs[searcher_iter:searcher_iter:end], yaxis=:log2)

pγs₀ = deepcopy(pγs)
pγs = deepcopy(pγs₀)
pα₀ = deepcopy(pα)
pα = deepcopy(pα₀)
#pα = plot(
plot!(pα,
    αs[searcher_iter:searcher_iter:end];
    yaxis=:log2,
    label="Adam i.p.s.=$algorithm_iter_per_search s.i.=$searcher_iter",
    xlabel="stage",
    ylabel="best α",
    #ylim=(0,1),
    color=:black,
    linestyle=:dash)
savefig(pα, "Adam_1000-1-1-b-a-ns-0660_bestpar_per_stage100200400ips_alpha.png")

pβ₀ = deepcopy(pβ)
pβ = deepcopy(pβ₀)
#pβ = plot(
#plot!(pβ,
#pγs = plot(
plot!(pγs,
    γss[searcher_iter:searcher_iter:end];
    #yaxis=:log2,
    label="Adam i.p.s.=$algorithm_iter_per_search s.i.=$searcher_iter",
    xlabel="stage",
    ylabel="best γs",
    ylim=(0,1.05),
    color=:black,
    linestyle=:dash)

savefig(pγs, "Adam_1000-1-1-b-a-ns-0660_bestpar_per_stage100200400ips_gammas.png")

#pγv = plot(
plot!(pγv,
    γvs[searcher_iter:searcher_iter:end];
    #yaxis=:log2,
    label="Adam i.p.s.=$algorithm_iter_per_search s.i.=$searcher_iter",
    xlabel="stage",
    ylabel="best γv",
    ylim=(0,1.05),
    color=:black,
    linestyle=:dash)

savefig(pγv, "Adam_1000-1-1-b-a-ns-0660_bestpar_per_stage100200400ips_gammav.png")

L_best = Dict{DataType, Dict{Int64, Float64}}()
L_best[typeof(localization)] = Dict{Int64, Float64}()
L_best[typeof(localization)][algorithm_iter]=instance.result.memoria["result_best"][end-1].result["L_best"]

L_best
function save_jld(filename::String, L::typeof(L_best))
    L′ = Dict{String, Dict{Int64, Float64}}()
    for (type, val) in L
        L′[string(type)] = Dict{Int64, Float64}()
        for (i, f) in val
            L′[string(type)][i] = f
        end
    end
    save(filename, "L_best", L′)
end
function load_jld(filename::String)
    L′ = load(filename)["L_best"]
    L = Dict{DataType, Dict{Int64, Float64}}()
    for (type_name, val) in L′
        type = eval(Meta.parse(type_name))
        L[type] = Dict{Int64, Float64}()
        for (i, f) in val
            L[type][i] = f
        end
    end
    L
end
save_jld("L_best.jld", L_best)
L_best = load_jld("L_best.jld")
#=
    End of testing subgradient with parameter search
=#


results = Dict{String, Dict{DataType, TestResult{Float64}}}()
iter = Dict{String, Dict{DataType, NamedTuple{(:max_iter, :max_hiter, :iter),Tuple{Int64,Int64,Int64}}}}()

function run_experiments(type::DataType)
    todos = Set{String}(["dual", "min_grad", "mg_EK", "mg_SPEK", "plot"])
    for (name, problem) in problems
        μ₀ = rand(Float64, size(problem.E, 1))
        if !haskey(results, name)
            results[name] = Dict{DataType, TestResult{Float64}}()
        end
        if !haskey(iter, name)
            iter[name] = Dict{DataType, NamedTuple{(:max_iter, :max_hiter, :iter),Tuple{Int64,Int64,Int64}}}()
        end
        for mie in 6:16
            max_iter = 2^mie
            results[name][type] =
                run_test(
                    "Optimization.jl/test/result/"*name,
                    Float64,
                    type,
                    problem,
                    Tuple(1.0),
                    NamedTuple(),
                    sg->sg.α/=2.0;
                    μ₀=μ₀,
                    max_iter=max_iter,
                    max_hiter=40,
                    restart=true,
                    todos=todos
                )
            ϵs = results[name][type].ϵs["mg_SPEK"]
            if ϵs.ϵᵣ + ϵs.Δϵᵣ < 1e-6
                Ls = results[name][type].Ls
                is = results[name][type].is
                L = results[name][type].bench.f_mg_SPEK
                for i in 1:length(is)
                    if (L-Ls[i])/abs(L) + ϵs.Δϵᵣ < 1e-6
                        @show (max_iter, 40, is[i])
                        iter[name][type] = (max_iter=max_iter, max_hiter=40, iter=is[i])
                        break
                    end
                end
                break
            end
        end
    end
end

run_experiments(Subgradient.FixedStepSize)
iter

save("20200809.jld", "results", results)
save("20200809iter.jld", "iter", iter)

open("iter_1000_netgen.txt", "w") do io
    for (name, val) in iter
        print(io, name)
        for type in keys(val)
            print(io, " ", type, " ",
                val[type].max_iter, " ",
                val[type].max_hiter, " ",
                val[type].iter)
        end
        println(io)
    end
end
open("iter_1000_netgen_res.txt", "w") do io
    for (name, val) in results
        print(io, name)
        for type in keys(val)
            print(io, " ", type, " ",
                val[type].bench.f_dual, " ",
                val[type].bench.f_mg_EK, " ",
                val[type].bench.f_mg_SPEK, " ",
                val[type].is[end], " ",
                val[type].ϵs["mg_EK"].ϵᵣ, " ",
                val[type].ϵs["mg_EK"].Δϵᵣ, " ",
                val[type].ϵs["mg_SPEK"].ϵᵣ, " ",
                val[type].ϵs["mg_SPEK"].Δϵᵣ)
        end
        println(io)
    end
end
myr = results["netgen-1000-1-1-a-a-ns-0330"][Subgradient.FixedStepSize]
iter["netgen-1000-1-1-a-a-ns-0330"]
L = results["netgen-1000-1-1-a-a-ns-0330"][Subgradient.NesterovMomentum].bench.f_mg_SPEK
Ls = results["netgen-1000-1-1-a-a-ns-0330"][Subgradient.FixedStepSize].Ls
is = results["netgen-1000-1-1-a-a-ns-0330"][Subgradient.FixedStepSize].is
plot!(p, is, (L .- Ls) ./ abs(L);
    yaxis=:log2,
    label="RSG", xlabel="iteration", ylabel=L"\epsilon_{rel}",
    color=:black, linestyle=:dot)


iter = load("20200809iter.jld")
jldopen("20200809iter.jld") do io
    read(io, "iter")
end

piter = parse.(TestgenParams, keys(iter))
ipiter = [zip(piter, keys(iter))...]
import Base.Order.isless
function isless(a::TestgenParams, b::TestgenParams)
    (a.n, a.singular, a.ρ, a.cf, a.cq, a.scale) <
        (b.n, b.singular, b.ρ, b.cf, b.cq, b.scale)
end

sort!(ipiter)
media_iter = Dict{String, Dict{DataType, NamedTuple{(:max_iter, :max_hiter, :iter),Tuple{Int64,Int64,Int64}}}}()
stddev_iter = Dict{String, Dict{DataType, NamedTuple{(:max_iter, :max_hiter, :iter),Tuple{Int64,Int64,Int64}}}}()
a = deepcopy(ipiter[1][1])

function get_mean_stddev(iterations, types)
    piter = parse.(TestgenParams, keys(iterations))
    ipiter = [zip(piter, keys(iterations))...]
    sort!(ipiter)
    media_iter = Dict{String, Dict{DataType, NamedTuple{(:max_iter, :max_hiter, :iter),Tuple{Float64,Float64,Float64}}}}()
    stddev_iter = Dict{String, Dict{DataType, NamedTuple{(:max_iter, :max_hiter, :iter),Tuple{Float64,Float64,Float64}}}}()

    for i in 1:5:240
        par = ipiter[i][1]
        par.k = 0
        name = string(par)
        media_iter[name] = Dict{DataType, NamedTuple{(:max_iter, :max_hiter, :iter),Tuple{Float64,Float64,Float64}}}()
        stddev_iter[name] = Dict{DataType, NamedTuple{(:max_iter, :max_hiter, :iter),Tuple{Float64,Float64,Float64}}}()
        for type in types
            m_i, m_h, ite = 0.0, 0.0, 0.0
            mm_i, mm_h, mite = Inf, Inf, Inf
            Mm_i, Mm_h, Mite = 0.0, 0.0, 0.0
            js = Int64[]
            for j in i:i+4
                p, ps = ipiter[j]
                if haskey(iterations[ps], type)
                    m_i += iterations[ps][type].max_iter
                    m_h += iterations[ps][type].max_hiter
                    ite += iterations[ps][type].iter
                    mm_i = min(mm_i, iterations[ps][type].max_iter)
                    mm_h = min(mm_h, iterations[ps][type].max_hiter)
                    mite = min(mite, iterations[ps][type].iter)
                    Mm_i = max(Mm_i, iterations[ps][type].max_iter)
                    Mm_h = max(Mm_h, iterations[ps][type].max_hiter)
                    Mite = max(Mite, iterations[ps][type].iter)
                    push!(js, j)
                end
            end
            if length(js) > 0
                m_i = m_i / length(js)
                m_h = m_h / length(js)
                ite = ite / length(js)
                media_iter[name][type] = (max_iter=m_i, max_hiter=m_h, iter=ite)
                stddev_iter[name][type] = (
                    max_iter=min(m_i-mm_i, Mm_i-m_i)|>x->x==0 ? 0.05*media_iter[name][type].max_iter : x,
                    max_hiter=min(m_h-mm_h, Mm_h-mm_h)|>x->x==0 ? 0.05*media_iter[name][type].max_hiter : x,
                    iter=min(ite-mite, Mite-ite)|>x->x==0 ? 0.05*media_iter[name][type].iter : x)
            end
        end
    end

    media_iter, stddev_iter
end

media_iter2, stddev_iter2 = get_mean_stddev(iter, [Subgradient.NesterovMomentum, Subgradient.FixedStepSize])
iter["netgen-1000-1-4-a-a-ns-0330"][Subgradient.NesterovMomentum]

for (key, val) in stddev_iter2
    println(length(val))
end

media_iter
stddev_iter
results["netgen-1000-1-4-a-a-ns-0330"][Subgradient.NesterovMomentum]

function print_latex_full_table(io::IO, iter, results)
    println(io,
        "\\begin{center}\n",
        "\\begin{longtable}{| l || c | c | c | c | c | c || c | c | c | c | c | c |}\n",
        "\\hline\n",
        "\\multirow{3}[0]{*}{Instance} & \n",
        "\\multicolumn{6}{|c||}{RSG} & \n",
        "\\multicolumn{6}{|c|}{RNM} \\\\\n",
        "\\cline{2-13}\n",
        " & \\multicolumn{2}{|c|}{mg EK} \n",
        " & \\multicolumn{2}{|c|}{mg SPEK} \n",
        " & \\multirow{2}[0]{*}{Max iter} \n",
        " & \\multirow{2}[0]{*}{Iter} \n",
        " & \\multicolumn{2}{|c|}{mg EK} \n",
        " & \\multicolumn{2}{|c|}{mg SPEK}\n",
        " & \\multirow{2}[0]{*}{Max iter} \n",
        " & \\multirow{2}[0]{*}{Iter}  \\\\\n",
        "\\cline{2-5}\\cline{8-11}\n",
        " & \$\\epsilon_{rel}\$ \n",
        " & \$\\Delta \\epsilon_{rel}\$ \n",
        " & \$\\epsilon_{rel}\$ \n",
        " & \$\\Delta\\epsilon_{rel}\$ \n",
        " & \n",
        " & \n",
        " & \$\\epsilon_{rel}\$ \n",
        " & \$\\Delta\\epsilon_{rel}\$\n",
        " & \$\\epsilon_{rel}\$ \n",
        " & \$\\Delta\\epsilon_{rel}\$ \n",
        " & \n",
        " & \\\\\n",
        "\\hline")

    names = sort([keys(iter)...])
    for name in names
        ival = iter[name]
        result = results[name]
        println(io,
            name[8:end],
            " & ")
        if haskey(result, Subgradient.FixedStepSize) && haskey(ival, Subgradient.FixedStepSize)
            println(io,
                @sprintf("%.0e", result[Subgradient.FixedStepSize].ϵs["mg_EK"].ϵᵣ),
                " & ",
                @sprintf("%.0e", result[Subgradient.FixedStepSize].ϵs["mg_EK"].Δϵᵣ),
                " & ",
                @sprintf("%.0e", result[Subgradient.FixedStepSize].ϵs["mg_SPEK"].ϵᵣ),
                " & ",
                @sprintf("%.0e", result[Subgradient.FixedStepSize].ϵs["mg_SPEK"].Δϵᵣ),
                " & ",
                @sprintf("%.0e", ival[Subgradient.FixedStepSize].max_iter),
                " & ",
                @sprintf("%.0e", ival[Subgradient.FixedStepSize].iter),
                " & ")
        else
            println(io,
                " & & & & & & ")
        end
        if haskey(result, Subgradient.NesterovMomentum) && haskey(ival, Subgradient.NesterovMomentum)
            println(io,
                @sprintf("%.0e", result[Subgradient.NesterovMomentum].ϵs["mg_EK"].ϵᵣ),
                " & ",
                @sprintf("%.0e", result[Subgradient.NesterovMomentum].ϵs["mg_EK"].Δϵᵣ),
                " & ",
                @sprintf("%.0e", result[Subgradient.NesterovMomentum].ϵs["mg_SPEK"].ϵᵣ),
                " & ",
                @sprintf("%.0e", result[Subgradient.NesterovMomentum].ϵs["mg_SPEK"].Δϵᵣ),
                " & ",
                @sprintf("%.0e", ival[Subgradient.NesterovMomentum].max_iter),
                " & ",
                @sprintf("%.0e", ival[Subgradient.NesterovMomentum].iter),
                " \\\\ ")
        else
            println(io,
                " & & & & & \\\\ ")
        end
    end

    println(io,
        "\\caption{Iterations up to \$\\epsilon_{rel}<1^{-6}\$}\n",
        "\\label{table:FullRestartedDIMACS1000}\n",
        "\\end{longtable}\n",
        "\\end{center}\n")
end

open("prova_fulltable.tex", "w") do io
    print_latex_full_table(io, iter, results)
end

function print_latex_table(io::IO, media, stdde)
    println(io, "\\begin{center}")
    println(io, "\\begin{longtable}{|l || c | c |}")
    println(io, "\\hline")
    println(io,
        "Instance & ",
        "RSG & ",
        "RNM \\\\")
    println(io, "\\hline\\hline")

    names = []
    for (name, med) in media
        push!(names, name)
    end
    sort!(names)
    for name in names
        println(io, name,
            " & ",
            @sprintf("%.0f", media[name][Subgradient.FixedStepSize].iter),
            "\$\\pm\$", @sprintf("%.0f", stdde[name][Subgradient.FixedStepSize].iter),
            " & ",
            @sprintf("%.0f", media[name][Subgradient.NesterovMomentum].iter),
            "\$\\pm\$", @sprintf("%.0f", stdde[name][Subgradient.NesterovMomentum].iter),
            "\\\\ \\hline")
    end

    println(io, "\\caption{Iterations to \$\\epsilon_{rel}<1e^{-6}\$}")
    println(io, "\\label{table:RestartedDIMACS1000}")
    println(io, "\\end{longtable}")
    println(io, "\\end{center}")
end

open("prova_tavola2.tex", "w") do io
    print_latex_table(io, media_iter2, stddev_iter2)
end

results
iter
a=0

function run_experiment(problem, max_iter, type, result, todos=Set{String}(["dual"]))
    result[type] =
        run_test(
            "Optimization.jl/test/result/"*pname,
            Float64,
            type,
            problem,
            (),
            (α=1.0, β=0.995),
            sg->sg.α/=2.0;
            μ₀=μ₀,
            max_iter=max_iter,
            max_hiter=40,
            restart=true,
            todos=todos
        )
end

#=
    START NEEDED ITERATIONS MEASURE
=#

using Optimization
problems = parse_dir(NetgenDIMACS, "CM19/Optimization.jl/test/gen/set/")

function exp_search(f, rng::Tuple{Int64, Int64})

    b, e = rng
    step = 1
    x = b
    u = e+1
    while (x ≤ e) && !f(x)
        print("*")
        x = b+step
        step *= 2
    end
    step = step ÷ 2
    bin_search(f, (x-step, min(e, x)))
end

function bin_search(f, rng::Tuple{Int64, Int64})
    b, e = rng
    while e≥b
        print("°")
        m = (b+e)÷2
        if !f(m)
            b = m+1
        else
            e = m-1
        end
    end
    e+1
end

function solvable_to(
    problem::QMCFBProblem,
    L::Float64,
    ϵ::Float64,
    hiter::Int64,
    iter::Int64,
    ::Type{SG},
    sg_args::Tuple,
    sg_kwargs::NamedTuple,
    sg_update,
    restart::Bool;
    μ₀::Array{Float64, 1}) where {SG <: SubgradientMethod}

    subgradient = SG(sg_args...; sg_kwargs...)
    algorithm = QMCFBPAlgorithmD1SG(;
        localization=subgradient,
        verbosity=0,
        max_iter=iter,
        μ₀=μ₀,
        ε=1e-6,
        ϵ=1e-12)
    test = get_test(algorithm; problem=problem, type=Float64)
    for i in 1:hiter
        run!(test)
        set!(algorithm, test.result)      # set initial status of algorithm
        algorithm.stopped = !restart      # not stopped implies re-init subgradient method
        sg_update(subgradient)
    end
    L_best = test.result.result["L_best"]
    abs(L_best-L)/abs(L) < ϵ
end

function table_of_needed_iterations(
    problem::QMCFBProblem,
    ϵs::Array{Float64, 1},
    hiters::Array{Int64, 1},
    max_total_iter::Int64,
    ::Type{SG},
    sg_args::Tuple,
    sg_kwargs::NamedTuple,
    sg_update,
    restart::Bool) where {SG <: SubgradientMethod}

    table = Int64[]

    model=solveQMCFBP(problem)
    L = begin
        if termination_status(model) == MOI.OPTIMAL
            objective_value(model)
        else
            Inf
        end
    end

    if L == Inf
        return table
    end

    m, n = size(problem.E)
    μ₀ = rand(Float64, m)

    # Table: ϵ x hiter
    max_total_iter′ = max_total_iter
    for ϵ in ϵs
        for hiter in hiters
            function f(iter::Int64)
                solvable_to(
                    problem,
                    L,
                    ϵ,
                    hiter,
                    iter,
                    SG,
                    sg_args,
                    sg_kwargs,
                    sg_update,
                    restart;
                    μ₀=μ₀)
            end
            max_iter = max_total_iter′÷hiter
            iter = exp_search(f, (1, max_iter))
            push!(table, iter≤max_iter ? iter : -1)
        end
    end
    table
end

tables = Dict{String, Array{Int64, 2}}()
pname = "netgen-1000-1-4-a-a-ns-0330"
tables["netgen-1000-1-4-a-a-ns-0330"] =
    table_of_needed_iterations(
        problems[pname],
        [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        100000,
        Subgradient.NesterovMomentum,
        (),
        (α=1.0, β=0.995),
        sg->sg.α/=2,
        true)
#=
    END NEEDED ITERATIONS MEASURE
=#



#=
    START PERFORMANCE MEASURE
=#
times["netgen-1000-1-4-a-a-ns-0330"] = 0.25
@elapsed model=solveQMCFBP(problems["netgen-1000-1-4-a-a-ns-0330"])

times = Dict{String, Float64}()
sols = Dict{String, Float64}()
for (name, probl) in problems2
    times[name] = @elapsed model=solveQMCFBP(probl)
    if termination_status(model) == MOI.OPTIMAL
        sols[name] = objective_value(model)
    end
end
sols2 = Dict{String, Float64}()
times2 = Dict{String, Float64}()
times2
max_hiter = 30
max_iter = 50
μ₀ = rand(Float64, 89)
restart = true
for (name, sol) in sols
    println(name)
    problem = problems2[name]
    m, n = size(problem.E)
    μ₀ = rand(Float64, m)
    to_do = true
    max_it = max_iter
    while to_do
        subgradient = Subgradient.NesterovMomentum(;α=1.0, β=0.995)
        algorithm = QMCFBPAlgorithmD1SG(;
            localization=subgradient,
            verbosity=0,
            max_iter=max_it,
            μ₀=μ₀,
            ε=1e-6,
            ϵ=1e-12)
        test = get_test(algorithm; problem=problem, type=Float64)
        println(max_it)
        newtime = @elapsed begin
            for i in 1:max_hiter
                run!(test)
                set!(algorithm, test.result)      # set initial status of algorithm
                algorithm.stopped = !restart      # not stopped implies re-init subgradient method
                subgradient.α /= 2
            end
        end
        L = test.result.result["L_best"]
        if abs(L-sols[name])/abs(sols[name]) < 1e-5
            to_do = false
            sols2[name] = L
            times2[name] = newtime
        else
            max_it *= 2
        end
    end
end

mtimes = Dict{String, Tuple{Float64, Float64, Float64}}()
mtimes2 = Dict{String, Tuple{Float64, Float64, Float64}}()
piter = parse.(TestgenParams, keys(problems))
ipiter = [zip(piter, keys(problems))...]
sort!(ipiter)
for i in 1:5:length(problems)
    cnt=0
    fname = ""
    t, mt, Mt = 0.0, Inf, 0.0
    t2, mt2, Mt2 = 0.0, Inf, 0.0
    for j in i:i+4
        name = ipiter[j][2]
        fname = string(name[1:14],'*',name[16:end])
        if name in keys(sols2)
            cnt += 1
            t += times[name]
            t2 += times2[name]
            mt = min(mt, times[name])
            mt2 = min(mt2, times2[name])
            Mt = max(Mt, times[name])
            Mt2 = max(Mt2, times2[name])
        end
    end
    if cnt>0
        t /= cnt
        t2 /= cnt
        mtimes[fname] = (mt, t, Mt)
        mtimes2[fname] = (mt2, t2, Mt2)
    end
end
mtimes["netgen-1000-1-*-a-a-ns-0330"]
mtimes
mtimes2

mtimes_5 = deepcopy(mtimes)
mtimes2_5 = deepcopy(mtimes2)
times2_5 = deepcopy(times2)
sols2_5 = deepcopy(sols2)

mtimes_8 = deepcopy(mtimes)
mtimes2_8 = deepcopy(mtimes2)
times2_8 = deepcopy(times2)
sols2_8 = deepcopy(sols2)

best_mid_worst = Float64[Inf, 0.0, 0.0, 0]
for (k, v) in mtimes
    r = mtimes2[k][2]/v[2]
    best_mid_worst[1] = min(r, best_mid_worst[1])
    best_mid_worst[3] = max(r, best_mid_worst[3])
    best_mid_worst[2] += r
    best_mid_worst[4] += 1.0
end
best_mid_worst[2] /= best_mid_worst[4]
best_mid_worst
save("mtimes1e-5.jld", "mt", (mtimes, mtimes2))

function print_latex_table_m(io::IO, m, m2)
    println(io, "\\begin{center}")
    println(io, "\\begin{longtable}{|l || c | c |}")
    println(io, "\\hline")
    println(io,
        "Instance & ",
        "RNM (ms) & ",
        "Gurobi (ms) \\\\")
    println(io, "\\hline\\hline")

    names = []
    for (name, med) in m
        push!(names, name)
    end
    sort!(names)
    for name in names
        println(io, name,
            " & ",
            @sprintf("%.0f", m2[name][2]*1000),
            "{\\raisebox{0.5ex}{\\tiny\$\\substack{+",
            @sprintf("%.0f", (m2[name][3]-m2[name][2])*1000),
            " \\\\ -",
            @sprintf("%.0f", (m2[name][2]-m2[name][1])*1000),
            "}\$}}",
            " & ",
            @sprintf("%.0f", (m[name][2])*1000),
            "{\\raisebox{0.5ex}{\\tiny\$\\substack{+",
            @sprintf("%.0f", (m[name][3]-m[name][2])*1000),
            " \\\\ -",
            @sprintf("%.0f", (m[name][2]-m[name][1])*1000),
            "}\$}}",
            "\\\\ \\hline")
    end

    println(io, "\\caption{Performance of RNM to \$\\epsilon_{rel}<10^{-5}\$ against Gurobi quadratic solver}")
    println(io, "\\label{table:RestartedDIMACS1000Perf5}")
    println(io, "\\end{longtable}")
    println(io, "\\end{center}")
end

open("table_perf5_.tex", "w") do io
    print_latex_table_m(io, mtimes, mtimes2)
end
#=
    END PERFORMANCE MEASURE
=#









L
result
function experiment1()
    max_iter=16*(2^14)
    iters = []
    Les = []
    L = 0
    for i=15:15
        if i==15
            run_experiment(max_iter, type, result, Set{String}(["dual", "min_grad", "mg_EK", "mg_SPEK"]))
            L = result[type].bench.f_mg_SPEK
        else
            run_experiment(max_iter, type, result, Set{String}(["dual"]))
        end
        push!(iters, max_iter)
        push!(Les, result[type].bench.f_dual)
        #is = result[type].is
        #Ls = result[type].Ls
        # plot!(p, is/max_iter, (L .- Ls) ./ abs(L); yaxis=:log2, label="RSG $max_iter")
        max_iter *= 2
    end
    iters, Les, L
end
it2, Le2, L2 = experiment1()
iters1, Les1, L1 = experiment1()
L1
L
push!(iters, iters1...)
push!(Les, Les1...)


display(p)
p2 = deepcopy(p)
p = deepcopy(p2)
h5write("20200808.h5", "Les", Les)
iters = Int64.(iters)
Les = Float64.(Les)

iters = h5read("20200808.h5", "iters")
Les = h5read("20200808.h5", "Les")

L = L1
plot!(p, it2, (L .- Le2) ./ abs(L);
    yaxis=:log10, xaxis=:log2,
    label="RNM singular=0.33", xlabel="iteration per stage", ylabel=L"\epsilon_{rel}",
    color=:black, linestyle=:dot)
savefig(p, "RSG-RNM_2_40_-1000-1-4-a-a-ns-0330.png")
p=plot()
savefig()