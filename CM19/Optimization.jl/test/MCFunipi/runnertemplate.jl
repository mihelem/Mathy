#First, generate your plot using Plots.jl:
# using Plots
# hdf5() #Select HDF5-Plots "backend"
# p = plot(...) #Construct plot as usual
#After you re-open a new Julia session, you can re-read the .hdf5 plot:
# using Plots
# pyplot() #Must first select some backend
# pread = Plots.hdf5plot_read("plotsave.hdf5")
# display(pread)
using Optimization, Parameters, Plots, LinearAlgebra
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
