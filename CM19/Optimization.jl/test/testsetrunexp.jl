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
    TEST PRIVATO PER COMPARARLO COL C++
=#
problems = parse_dir(NetgenDIMACS, "CM19/Optimization.jl/src/cpp/bin/scaling/")
filename = "netgen-50000-1-1-a-a-ns-33000"
problem = problems["netgen-50000-1-1-a-a-ns-33000"]
model = solveQMCFBP(problem)
L = begin
    if termination_status(model) == MOI.OPTIMAL
        objective_value(model)
    else
        Inf
    end
end
propath = "CM19/Optimization.jl/src/cpp/bin/scaling/"
problems_little = Dict([(name, problem) for (name, problem) in problems if parse(Int64, split(name, "-")[2]) < 30000])
problems_big = Dict([(name, problem) for (name, problem) in problems if parse(Int64, split(name, "-")[2]) ≥ 30000])
function best_stater(path, file, tot_iter, sta, sta_b, sta_e, α, β)
    Ls, tims, stas = Float64[], Float64[], Int64[]
    sta_e = min(sta_e, tot_iter)
    sta = min(sta, sta_e)
    L_best = -Inf
    sta_best = sta
    tim_best = 0.0
    visited = zeros(Bool, sta_e-sta_b+1)
    function visit(sta)
        if sta < sta_b || sta > sta_e || visited[sta-sta_b+1]
            false
        else
            visited[sta-sta_b+1] = true
            true
        end
    end
    while sta_b≤sta≤sta_e
        iter = tot_iter ÷ sta
        @show (sta, iter)
        out = read(
                pipeline(
                    `CM19/Optimization.jl/src/cpp/bin/test $path$file $sta $iter $α $β`),
                    #stdout="CM19/Optimization.jl/src/cpp/bin/tmp.tmp"),
                String)
        print(out)
        lines = split(out, '\n')
        line = lines[lines .!= ""][end]
        L, tim = split(line, ' ') |> tok -> (parse(Float64, tok[1]), parse(Float64, tok[end]))
        push!(Ls, L)
        push!(tims, tim)
        push!(stas, sta)

        if sta == sta_best
            @show sta == sta_best
            if @show L > L_best(1+sign(L_best)*1e-9)

                L_best = L
                tim_best = tim
            end
            if visit(sta+1)
                @show sta += 1
            elseif visit(sta-1)
                @show sta -= 1
            else
                @show break
            end
        elseif @show L > L_best*(1+sign(L_best)*1e-9)
            step = sta ≥ sta_best ? 1 : -1
            @show L_best = L
            sta_best = sta
            tim_best = tim

            if @show visit(sta+step)
                sta += step
            else
                @show break
            end
        elseif @show L < L_best*(1-sign(L_best)*1e-10)
            @show step = sta_best > sta ? 1 : -1
            if @show visit(sta_best+step)
                sta = sta_best+step
            else
                @show break
            end
        else
            @show step = sta_best > sta ? 1 : -1
            if visit(sta+step)
                @show sta += step
            elseif visit(sta-step)
                @show sta -= step
            elseif visit(sta_best+1)
                @show sta = sta_best+1
            elseif visit(sta_best-1)
                @show sta = sta_best-1
            else
                @show break
            end
        end
    end

    (Ls, L_best, sta_best, tot_iter÷sta_best, tim_best)
end
Ls, L′, stage, iter, tim = best_stater(propath, "netgen-1024-2-1-a-a-ns-0000", 100, 1, 4, 1.0, 0.975)
Ls
L′
stage
iter

results = Tuple{String, Float64, Int64, Int64, Float64, Float64}[]
results
save("scaling20201014.jld", "results", results)

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
function scaling(path, problems, ϵ_rel, res, α, β, max_iters)
    stage = [1]
    for (name, problem) in problems
        model = solveQMCFBP(problem)
        L = begin
            if termination_status(model) == MOI.OPTIMAL
                objective_value(model)
            else
                Inf
            end
        end
        if L < Inf
            function check_ϵ(tot_iter)
                Ls, L′, stage′, iter, tim = best_stater(path, name, tot_iter, stage[1], 1, 30, α, β)
                println("--------------")
                stage[1] = stage′
                ϵ_rel′ = (L-L′)/abs(L)
                push!(res, (name, ϵ_rel, tot_iter, stage′, ϵ_rel′, tim))
                ϵ_rel′ ≤ ϵ_rel
            end
            exp_search(check_ϵ, (1, max_iters))
        end
    end
end

results_little = Tuple{String, Float64, Int64, Int64, Float64, Float64}[]
scaling(propath, problems_little, 1e-1, results_little, 1.0, 0.975, 800000)

function RNMsolver(problem::QMCFBProblem, n_iters, n_stages, α, β, α_div)
    @unpack Q, q, E, b, l, u = problem
    m, n = size(E)
    Q╲ = view(Q, [CartesianIndex(i, i) for i in 1:n])
    invQ╲ = 1. ./ Q╲
    sing = Q╲ .== 0
    nsing = .~sing

    v = zeros(Float64, m)
    g = zeros(Float64, m)
    μ = zeros(Float64, m)
    L_best = -Inf
    μ_best = zeros(Float64, m)
    function mid(a, b, c)
        m, M = min(a, b), max(a, b)
        min(max(c, m), M)
    end
    function get_x(μ)
        x = zeros(Float64, n)
        x[nsing] = mid.(invQ╲[nsing] .* (-q-E'μ)[nsing], l[nsing], u[nsing])
        x[sing] = (-q-E'μ)[sing]
        mi, ma = x[sing].<0, x[sing].≥0
        x[sing][mi] = l[sing][mi]
        x[sing][ma] = u[sing][ma]
        x
    end
    function get_∂L(x)
        E*x-b
    end
    function get_L(μ, x, ∂L)
        x⋅(0.5*Q╲ .* x + q) + μ⋅∂L
    end

    for stage in 1:n_stages
        for iter in 1:n_iters
            g[:] .= get_∂L(get_x(μ+β*v))
            v[:] .= β*v + α*g
            μ[:] .= μ + v
            L = get_x(μ) |> x -> get_L(μ, x, get_∂L(x))
            if L > L_best
                L_best = L
                μ_best[:] .= μ
            end
        end
        μ[:] .= μ_best
        α /= α_div
    end
    return L_best, μ_best
end
L′, μ = RNMsolver(problem, 8000, 20, 1.0, 0.99975, 2.0)
#=
    START NEEDED ITERATIONS MEASURE
=#
include("testsetrun.jl")
include("dmacio.jl")
include("grb.jl")
using LaTeXStrings
using Printf
using Plots
using JLD
using HDF5
using Distributed
addprocs(24)

@everywhere push!(LOAD_PATH, "/home/mihele/Source/julia/git/CM19/Optimization.jl/")
@everywhere using Optimization
@everywhere include("/home/mihele/Source/julia/git/CM19/Optimization.jl/test/dmacio.jl")
@everywhere include("/home/mihele/Source/julia/git/CM19/Optimization.jl/test/grb.jl")

problems = parse_dir(NetgenDIMACS, "CM19/Optimization.jl/src/cpp/bin/")

@everywhere function exp_search(f, rng::Tuple{Int64, Int64})

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

@everywhere function bin_search(f, rng::Tuple{Int64, Int64})
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

@everywhere function solvable_to(
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

@everywhere function table_of_needed_iterations(
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
        return reshape(table, length(hiters), :)
    end

    m, n = size(problem.E)
    μ₀ = rand(Float64, m)

    # Table: ϵ x hiter
    max_total_iter′ = max_total_iter
    for ϵ in ϵs
        print("\nϵ=$ϵ")
        for hiter in hiters
            print("\n\thiter=", lpad(string(hiter), 2, '0'), "  ::  ")
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
    reshape(table, length(hiters), :)
end

tables = Dict{String, Array{Int64, 2}}()

tables
plot(tables["netgen-1000-1-4-a-a-ns-0330"] .* [1:20;])
function make_tables(
    problems::Dict{String, QMCFBProblem},
    tables::Dict{String, Array{Int64, 2}},
    ϵs::Array{Float64, 1},
    hiters::Array{Int64, 1},
    max_total_iter::Int64,
    ::Type{SG},
    sg_args::Tuple,
    sg_kwargs::NamedTuple,
    sg_update,
    restart::Bool) where {SG <: SubgradientMethod}

    names = [pname for pname in keys(problems) if !haskey(tables, pname)]

    fs = Dict{String,Future}()
    @sync for pname in names
        @async fs[pname] = @spawn table_of_needed_iterations(
            problems[pname],
            ϵs,
            hiters,
            max_total_iter,
            SG,
            sg_args,
            sg_kwargs,
            sg_update,
            restart)
    end

    @sync for pname in names
        @async tables[pname] = fetch(fs[pname])
    end
end
ϵs = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
#ϵs = [0.1]
#hiters = [1:2;]
hiters = [1:20;]
make_tables(
    problems,
    tables,
    ϵs,
    hiters,
    100000,
    Subgradient.NesterovMomentum,
    (),
    (α=1.0, β=0.995),
    sg->sg.α/=2,
    true)

tables
tables_tot = Dict{String, Array{Int64,2}}()
for (name, arr) in tables
    tables[name] = arr .* [1:20;]
end
tables_tot = deepcopy(tables)
for (name, arr) in tables_tot
    if size(arr)[2] > 0
        arr[arr .< 0] .= maximum(arr)+1
    end
end
tables_tot
for (name, arr) in tables_tot
    if size(arr, 2) > 0
        arr[arr .== maximum(arr)] .= 100001
    end
end
tables
snames = sort([keys(tables_tot)...])
tables_tot_minite = Dict{String, Array{Int64, 1}}()
tables_tot_miniph = Dict{String, Array{Int64, 1}}()
tables_tot_minhit = Dict{String, Array{Int64, 1}}()
unsolvables = [name for (name, arr) in tables_tot if size(arr)[2]==0]
sort!(unsolvables)
tables_tot
tables_tot_miniph
for (name, arr) in tables_tot
    if size(arr)[2]>0
        tables_tot_miniph[name] =
        tables_tot_minite[name] = [minimum(c) for c in eachcol(arr)]
        tables_tot_minite[name][tables_tot_minite[name] .== 100001] .= 200000
            [(minimum(arr[:, i]) == 100001 ? 100001 : tables[name][argmin(arr[:, i]), i]) for i in 1:size(arr, 2)]
        tables_tot_minhit[name] =
            [(minimum(c) == 100001 ? 40 : argmin(c)) for c in eachcol(arr)]
    end
end
tables_tot_minite
function plot_by_instance(pname)
    ss = ["0000", "0330", "0660", "1000"]
    styls = [:solid, :dash, :dashdot, :dot]
    mshapes = [:circle, :dtriangle, :utriangle, :x]
    colors = [:grey, :white, :white, :black]
    malphas = [0.7, 0.5, 0.5, 1.0]
    pu = plot()
    # Plot by instance comparing singular percentage
    for i in eachindex(ss)
        let arr = tables_tot_minite[pname*"-"*ss[i]]
            plot!(pu,
                title=pname,
                10.0 .^(.- [1, 2, 3, 4, 5, 6]),
                seriestype=:scatter,
                legend=:topleft,
                xlabel=L"\epsilon_{rel}",
                ylabel="total iterations",
                arr,
                xaxis=:log10,
                yaxis=:log10,
                ylims=(0.7*minimum(arr), min(maximum(arr), 100000)),
                xflip=true,
                markeralpha=malphas[i],
                markershape=mshapes[i],
                color=colors[i],
                strokecolor=:black,
                label=ss[i])
        end
    end
    pu
end
function plot_by_setup(
    n::Int64,       # arcs
    ρ::Int64,       # graph density ∈ [1, 2, 3]
    ks::Array{Int64, 1},
    cf::Char,       # linear to fixed cost, 'a': high, 'b': low (CHECK)
    cq::Char,       # quadratic to fixed cost, 'a': high, 'b': low (CHECK)
    scale::String,  # scale arc capacity by 0.7 if "s", no action if "ns"
    singular::Int64,
    tables::Dict{String, Array{Int64, 2}})

    setups = [TestgenParams(PargenParams(n, ρ, k, cf, cq, scale), singular) for k in ks]
    names = [string(setup) for setup in setups]
    styls = [:solid, :dash, :dashdot, :dot]
    mshapes = [:circle, :dtriangle, :utriangle, :x, :+, :diamond]
    colors = [:grey, :white, :white, :black, :black, :black, :grey]
    malphas = [0.7, 0.5, 0.5, 1.0, 1.0, 1.0, 0.7]
    pu = plot()
    # Plot by instance comparing singular percentage
    title = get_setup_name(setups)
    ylimsmin = Inf
    ylimsmax = -Inf
    for i in eachindex(ks)
        curname = string(setups[i])
        if !haskey(tables_tot_minite, curname)
            continue
        end
        let arr = tables_tot_minite[curname]
            ylimsmin = min(ylimsmin, 0.7*minimum(arr))
            ylimsmax = begin
                maxarr = maximum(arr)
                max(ylimsmax, min(1.3*maxarr, 100000))
            end
            @show (ylimsmin, ylimsmax)
            plot!(pu,
                10.0 .^(.- [1, 2, 3, 4, 5, 6]),
                arr,
                seriestype=:scatter,
                legend=:topleft,
                xlabel=L"\epsilon_{rel}",
                ylabel="iterations per stage",
                title=title,
                xaxis=:log10,
                yaxis=:log10,
                ylims=(ylimsmin, 100000),
                xflip=true,
                markeralpha=malphas[i],
                markershape=mshapes[i],
                color=colors[i],
                strokecolor=:black,
                label="* ← "*string(ks[i]))
        end
    end
    title, pu
end

using PyPlot
using PyCall
using Plots
Plots.backend()

pyplot()
function get_setup_name(setups::Array{TestgenParams, 1})
    split(string(setups[1]), "-") |>
        tokens -> begin
            tokens[4] = "*"
            join(tokens, "-")
        end
end
function plot_by_setup_h(
    n::Int64,       # arcs
    ρ::Int64,       # graph density ∈ [1, 2, 3]
    ks::Array{Int64, 1},
    cf::Char,       # linear to fixed cost, 'a': high, 'b': low (CHECK)
    cq::Char,       # quadratic to fixed cost, 'a': high, 'b': low (CHECK)
    scale::String,  # scale arc capacity by 0.7 if "s", no action if "ns"
    singular::Int64)

    setups = [TestgenParams(PargenParams(n, ρ, k, cf, cq, scale), singular) for k in ks]
    names = [string(setup) for setup in setups]
    styls = [:solid, :dash, :dashdot, :dot]
    mshapes = [:circle, :dtriangle, :utriangle, :x, :+, :diamond]
    colors = [:grey, :white, :white, :black, :black, :black, :grey]
    malphas = [0.7, 0.5, 0.5, 1.0, 1.0, 1.0, 0.7]
    pu = plot()
    # Plot by instance comparing singular percentage
    title = get_setup_name(setups)
    ylimsmin = Inf
    ylimsmax = -Inf
    for i in eachindex(ks)
        curname = string(setups[i])
        if !haskey(tables_tot_minhit, curname)
            continue
        end
        let arr = tables_tot_minhit[curname]
            #ylimsmin = min(ylimsmin, 0.7*minimum(arr))
            #ylimsmax = ylimsmax = maximum([arr[arr .≤ 20] .+ 1; ylimsmax])
            #ylimsmin = min(ylimsmin, minimum([arr[arr .> 1]; 2])-0.5)
            #@show (ylimsmin, ylimsmax)
            plot!(pu,
                10.0 .^(.- [1, 2, 3, 4, 5, 6]),
                arr,
                seriestype=:scatter,
                legend=:topleft,
                xlabel=L"\epsilon_{rel}",
                ylabel="stages",
                title=title,
                xaxis=:log2,
                #yaxis=:log10,
                #ylims=(ylimsmin, ylimsmax),
                ylims=(1.5, 20.5),
                xflip=true,
                markeralpha=malphas[i],
                markershape=mshapes[i],
                color=colors[i],
                strokecolor=:black,
                label="* ← "*string(ks[i]))
        end
    end
    title, pu
end
pu = plot_by_setup_h(1000, 1, [1:5;], 'a', 'a', "ns", 1000)
pus2 = Dict{String, typeof(pu)}()
for ρ in 1:3
    for cf in ['a', 'b']
        for cq in ['a', 'b']
            for sing in [0, 330, 660, 1000]
                #title, pu = plot_by_setup_h(1000, ρ, [1:5;], cf, cq, "ns", sing)
                title, pu = plot_by_setup(1000, ρ, [1:5;], cf, cq, "ns", sing, tables)
                pus2[title] = pu
            end
        end
    end
end
pus2
pusnames = sort([keys(pus)...])
display(pus2[pusnames[4]])
for n in pusnames
    png(pus2[n], "CM19/report/result/netgen/"*"itepst_fixlim_"*n)
end
png(pu, "prova.png")
display(pu)
pu
using JLD
save("tables0907.jld", "tables", tables)
tables = load("tables0907.jld", "tables")
pname = "netgen-1000-1-4-a-a-ns-0330"
tables["netgen-1000-1-4-a-a-ns-0330"] =
    table_of_needed_iterations(
        problems[pname],
        [0.1],
        [1:2;],
        #[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
        #[1:20;],
        100000,
        Subgradient.NesterovMomentum,
        (),
        (α=1.0, β=0.995),
        sg->sg.α/=2,
        true)

table = tables["netgen-1000-1-4-a-a-ns-0330"] .* [1:20;]
plot(table)
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
