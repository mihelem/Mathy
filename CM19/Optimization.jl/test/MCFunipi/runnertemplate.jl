#First, generate your plot using Plots.jl:
# using Plots
# hdf5() #Select HDF5-Plots "backend"
# p = plot(...) #Construct plot as usual
#After you re-open a new Julia session, you can re-read the .hdf5 plot:
# using Plots
# pyplot() #Must first select some backend
# pread = Plots.hdf5plot_read("plotsave.hdf5")
# display(pread)
using Optimization, Parameters, Plots, LinearAlgebra, Printf
hdf5()

include("../nljumpy.jl")
include("../dmacio.jl")

function dir_read_run_restarted_Nesterov(path::String, savepath::String)
    subgradient = Subgradient.NesterovMomentum(α=1.0, β=0.99)
    algorithm = QMCFBPAlgorithmD1SG(;
        localization=subgradient,
        verbosity=-1,
        max_iter=4000,
        ε=1e-6,
        ϵ=1e-12);
    function update(instance)
        set!(instance.solver.algorithm, instance.result)
        instance.solver.algorithm.stopped = false
        instance.solver.algorithm.localization.α /= 2.0
    end

    dir_read_run_restarted(path, savepath;
        algorithm=algorithm,
        label="Nesterov Momentum, β=$(algorithm.localization.β)",
        update=update,
        max_restart=40)
end

function read_run_restarted(filename::String;
    algorithm,
    update,
    max_restart)
    # Preamble: get the instance from filename
    # Note: discard fixed costs
    E, b, l, u, q = parse_dmx(filename*".dmx");
    Q, 𝔮 = parse_qfc(filename*".qfc");
    problem = QMCFBProblem(Q, q, l, u, E, b, false);

    #@unpack Q, q, l, u, E, b = problem;
    m, n = size(E);

    algorithm.μ₀ = rand(eltype(Q), m);
    instance = get_test(algorithm;
        problem=problem);
    instance.solver.options.memoranda = Set(["L_best","i_best"]);

    Ls = [];
    is = [];
    for i in 1:max_restart
        run!(instance);
        push!(Ls, instance.result.memoria["L_best"]...);
        push!(is,((i-1)*algorithm.max_iter .+ instance.result.memoria["i_best"])...);
        update(instance)
    end

    get_L = x -> x'*(0.5Q*x + q);

    # calculate min-norm subgradient
    μ = instance.result.result["μ_best"];
    x = Optimization.MinCostFlow.primal_from_dual(problem, μ;
        ϵ=1e-4, ε=1e-10, max_iter=2000);
    x_ub2, Δ = BFSHeuristic(problem, x; ϵ=1e-12) |> heu -> (init!(heu); run!(heu));
    L_ub2 = get_L(x_ub2)

    # use JuMP+Ipopt (then heuristic) to get an upper bound
    x_ub = get_solution_quadratic_box_constrained(problem, zeros(Float64, n));
    x_ub[:] = max.(min.(x_ub, u), l);
    heu = BFSHeuristic(problem, x_ub; ϵ=1e-12);
    init!(heu);
    x_ub_h, Δ = run!(heu);
    L_ub = get_L(x_ub_h)

    return μ, x, is, Ls, x_ub_h, L_ub, x_ub2, L_ub2, problem
end

function dir_read_run_restarted(path::String, savepath::String;
    algorithm,
    label,
    update,
    max_restart)

    filenames = Set([name[1:end-4] for name in readdir(path) if name[max(end-3, 1):end]==".dmx"])
    prefix, saveprefix =
        (path -> (length(path)==0 || path[end] == '/') ? path : path*"/") |>
        normalize -> (normalize(path), normalize(savepath))
    function elaborate(filename)
        println("Reading $filename...")
        fullname = prefix*filename
        algorithm′ = deepcopy(algorithm)
        μ, x, is, Ls, x_ub, L_ub, x_ub2, L_ub2, problem =
            try
                read_run_restarted(fullname;
                    algorithm=algorithm′,
                    update=update,
                    max_restart=max_restart)
            catch e
                println("Error in $filename")
                open("error.log", "a") do io
                    println(io, "$filename")
                end
                return
            end
        @unpack Q, q, l, u, E, b = problem
        for (Lub, ub) in [(L_ub, "Ipopt"), (L_ub2, "minsg")]
            plot(is,
                sign(Lub) .- Ls./abs(Lub);
                yaxis=:log10,
                label=label) |>
            p -> Plots.hdf5plot_write(p, saveprefix*filename*"_"*ub*".hdf5")
        end
        open(saveprefix*filename*".log", "w") do io
            println(io, "is")
            (i -> print(io, i, " ")).(is[1:end-1])
            println(io, is[end])
            println(io, "Ls")
            (i -> print(io, i, " ")).(Ls[1:end-1])
            println(io, Ls[end])
            i = argmax(Ls)
            println(io, "result : i is[i] Ls[i] L_ub_Ipopt normsgL_ub_Ipopt L_ub_minsg normsgL_ub_minsg")
            println(io, i, " ", is[i], " ", Ls[i], " ", L_ub, " ", norm(E*x_ub-b), " ", L_ub2, " ", norm(E*x_ub2-b))
        end
    end
    elaborate.(filenames)
end

function move_used_data(from::String, to::String, result::String)
    from, to, result =
        (s -> (length(s)>0 && s[end]!='/') ? s*"/" : s).((from, to, result))

    from_f = readdir(from)
    result_f = readdir(result)

    for file in from_f
        if any(startswith.(result_f, file[1:end-4]))
            try
                s, d = from*file, to*file
                mv(s, d)
                println(s, " → ", d)
            catch e
                s, d = from*file, to*file*string(stat(from*file).mtime)
                mv(s, d)
                println(s, " → ", d)
            end
        end
    end
end

function latex_table_from_results(result::String, to::String;
    caption::String="",
    label::String="")

    result = (s -> (length(s)>0 && s[end]!='/') ? s*"/" : s)(result)
    results = [file for file in readdir(result) if endswith(file, ".log")]
    if length(results) == 0
        return
    end
    open(to, "a") do ioa
        println(ioa, "\\begin{center}")
        println(ioa, "\\begin{longtable}{|l || c | c | c | c | r|}")
        println(ioa, "\\hline")
        println(ioa,
            "Input & ",
            "\$\\frac{L_{ub}^{Ipopt}-L_{lb}}{\\left | L_{ub}^{Ipopt} \\right |}\$ & ",
            "\$\\left\\lVert Ex_{ub}^{Ipopt} - b \\right\\rVert\$ & ",
            "\$\\frac{L_{ub}^{\\min\\left\\lVert \\partial L \\right\\rVert}-L_{lb}}{\\left | L_{ub}^{\\min\\left\\lVert \\partial L \\right\\rVert} \\right |}\$ &",
            "\$\\left\\lVert Ex_{ub}^{\\min\\left\\lVert \\partial L \\right\\rVert} - b\\right\\rVert \$ & ",
            "count \\\\")
        println(ioa, "\\hline\\hline")

        for file in results
            println("Opening $(result*file)")
            open(result*file, "r") do ior
                while !eof(ior)
                    if startswith(readline(ior), "result")
                        words = split(readline(ior), " ")
                        # i is[i] Ls[i] L_ub_Ipopt normsgL_ub_Ipopt L_ub_minsg normsgL_ub_minsg
                        i, i_best, L_best, L_ub_Ipopt, normsgL_ub_Ipopt, L_ub_minsg, normsgL_ub_minsg =
                            (j -> parse([Int64, Int64, Float64, Float64, Float64, Float64, Float64][j], words[j])).([1:length(words);])
                        print(ioa, "\\texttt{", file[1:end-4], "} & ")
                        (w -> print(ioa, "\$", w, "\$ & ")).(
                            [@sprintf("%.1e", sign(L_ub_Ipopt)-L_best/abs(L_ub_Ipopt)),
                             @sprintf("%.1e", normsgL_ub_Ipopt),
                             @sprintf("%.1e", sign(L_ub_minsg)-L_best/abs(L_ub_minsg)),
                             @sprintf("%.1e", normsgL_ub_minsg)
                            ])
                        println(ioa, "\$", i_best, "\$ \\\\")
                        println(ioa, "\\hline")
                    end
                end
            end
        end
        #println(ioa, "\\caption{Nesterov Restarted on instances from MCF}")
        #println(ioa, "\\label{ltb:NesterovRestartedDIMACS}")
        if caption != ""
            println(ioa, "\\caption{$caption}")
        end
        if label != ""
            println(ioa, "\\label{$label}")
        end
        println(ioa, "\\end{longtable}")
        println(ioa, "\\end{center}")
    end
end
