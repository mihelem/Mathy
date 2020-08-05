# Benchmark Subgradient
# nodes arcs singular active
# L_lb L_min_grad L_EK L_SPEK L_SPEKn L_JuMP L_JuMP_EK L_JuMP_SPEK L_JuMP_SPEKn
# time_dual time_min_grad time_EK time_SPEK time_SPEKn time_JuMP time_JuMP_EK time_JuMP_SPEK time_JuMP_SPEKn
# (time is in nanoseconds)
using Optimization, LinearAlgebra, Parameters, BenchmarkTools, Profile
include("../test/nljumpy.jl")

mutable struct BenchResult{V}
    nodes::Int64
    arcs::Int64
    singular::Int64
    active::Int64

    x_dual::V
    x_min_grad::V
    x_mg_EK::V
    x_EK::V
    x_mg_SPEK::V
    x_SPEK::V
    x_mg_SPEKn::V
    x_SPEKn::V
    x_JuMP::V
    x_JuMP_EK::V
    x_JuMP_SPEK::V
    x_JuMP_SPEKn::V

    f_dual::V
    f_min_grad::V
    f_mg_EK::V
    f_EK::V
    f_mg_SPEK::V
    f_SPEK::V
    f_mg_SPEKn::V
    f_SPEKn::V
    f_JuMP::V
    f_JuMP_EK::V
    f_JuMP_SPEK::V
    f_JuMP_SPEKn::V

    df_dual::V
    df_min_grad::V
    df_mg_EK::V
    df_EK::V
    df_mg_SPEK::V
    df_SPEK::V
    df_mg_SPEKn::V
    df_SPEKn::V
    df_JuMP::V
    df_JuMP_EK::V
    df_JuMP_SPEK::V
    df_JuMP_SPEKn::V

    time_dual::Float64
    time_min_grad::Float64
    time_mg_EK::Float64
    time_EK::Float64
    time_mg_SPEK::Float64
    time_SPEK::Float64
    time_mg_SPEKn::Float64
    time_SPEKn::Float64
    time_JuMP::Float64
    time_JuMP_EK::Float64
    time_JuMP_SPEK::Float64
    time_JuMP_SPEKn::Float64

    unf_dual::V
    unf_min_grad::V
    unf_mg_EK::V
    unf_EK::V
    unf_mg_SPEK::V
    unf_SPEK::V
    unf_mg_SPEKn::V
    unf_SPEKn::V
    unf_JuMP::V
    unf_JuMP_EK::V
    unf_JuMP_SPEK::V
    unf_JuMP_SPEKn::V

    unf_dual_1::V
    unf_min_grad_1::V
    unf_mg_EK_1::V
    unf_EK_1::V
    unf_mg_SPEK_1::V
    unf_SPEK_1::V
    unf_mg_SPEKn_1::V
    unf_SPEKn_1::V
    unf_JuMP_1::V
    unf_JuMP_EK_1::V
    unf_JuMP_SPEK_1::V
    unf_JuMP_SPEKn_1::V

    BenchResult{V}() where {V} = new{V}()
    BenchResult{V}(nodes, arcs, singular, active) where {V} = new{V}(nodes, arcs, singular, active)
end
Base.show(io::IO, result::BenchResult{V}) where{V} = begin
    names = fieldnames(BenchResult)
    for name in names[1:end-1]
        print(io, getfield(result, name) |> a-> (is_error_v(a) ? "error" : a), " ")
    end
    print(io, getfield(result, names[end]) |> a-> (is_error_v(a) ? "error" : a), " ")
end
function legenda(io::IO, ::Type{BenchResult{V}}) where {V}
    names = fieldnames(BenchResult{V})
    for name in names[1:end-1]
        print(io, name, " ")
    end
    println(io, names[end])
end
function Base.parse(::Type{BenchResult{V}}, s::String; legend=nothing) where {V}
    result = BenchResult{V}()

    tokens = split(s, ' ') |> tk -> tk[(x -> length(x)>0).(tk)]
    names = fieldnames(BenchResult{V})
    types = (name -> typeof(getfield(result, name))).(names)
    ((name, type, token) ->
        if token != "error"
            setfield!(result, name, parse(type, token))
        else
            setfield!(result, name, error_v(type))
        end).(names, types, tokens)

    result
end

error_v(::Type{T}) where {T<:AbstractFloat} = convert(T, NaN)
error_v(::Type{T}) where {T<:Integer} = typemax(T)
is_error_v(value::T) where {T} = value == error_v(T)
is_error_v(value::T) where {T<:AbstractFloat} = isnan(value)

function run_bench(
    ::Type{V},
    ::Type{SG},
    problem::QMCFBProblem,
    singular::Int64, active::Int64,
    sg_args::Tuple=(),
    sg_kwargs::NamedTuple=(α=1.0, β=0.99),
    sg_update=sg->sg.α/=2.0;
    μ₀=nothing,
    max_iter::Int64=4000,
    max_hiter::Int64=40,
    restart::Bool=true,
    todos::Set=Set{String}(["dual", "min_grad", "mg_EK", "mg_SPEK", "mg_SPEKn",
        "JuMP", "JuMP_EK", "JuMP_SPEK", "JuMP_SPEKn", "time", "plot"])) where {V, SG<:SubgradientMethod}

    m, n = size(problem.E)
    result = BenchResult{V}(m, n, singular, active)

    μ₀ = μ₀ === nothing ? rand(m) : μ₀
    subgradient = SG(sg_args...; sg_kwargs...)

    algorithm = QMCFBPAlgorithmD1SG(;
        localization=subgradient,
        verbosity=1,
        max_iter=max_iter,
        μ₀=μ₀,
        ε=1e-6,
        ϵ=1e-12);
    test = get_test(algorithm;
        problem=problem,
        type=V)

    if "plot" in todos
        test.solver.options.memoranda = Set(["L_best", "i_best"])
    end

    problem = test.problem
    @unpack Q, q, l, u, E, b = problem
    Q╲ = view(Q, [CartesianIndex(i, i) for i in 1:size(Q, 1)])

    is = []
    Ls = []
    function runtest()
        for i in 1:max_hiter
            print("$i ")
            run!(test)
            if "plot" in todos
                push!(Ls, test.result.memoria["L_best"]...);
                push!(is, ((i-1)*algorithm.max_iter .+ test.result.memoria["i_best"])...);
            end
            set!(algorithm, test.result)      # set initial status of algorithm
            algorithm.stopped = !restart      # not stopped implies re-init subgradient method
            sg_update(subgradient)
        end
        println()
    end

    get_f = x -> x'*(0.5Q*x + q)
    get_df = x -> norm(Q*x + q)
    cache = Dict{String, Any}()

    # benchmark subgradient method
    println("dual")
    if "time" in todos
        bm = @benchmark $(runtest)() evals=1 samples=1 seconds=1e-9
        result.time_dual = minimum(bm).time
    else
        runtest()
        result.time_dual = error_v(Float64)
    end
    μ = test.result.result["μ_best"]
    result.f_dual = test.result.result["L_best"]
    x = Optimization.MinCostFlow.primal_from_dual(problem, μ; ϵ=0.0, ε=1e-10, max_iter=2000)
    result.df_dual = get_df(x)
    result.x_dual = norm(x)
    let Exb = E*x-b
        result.unf_dual = norm(Exb)
        result.unf_dual_1 = norm(Exb, 1)
    end

    # benchmark min-norm ϵ-subgradient
    println("min_grad")
    try
        if !("min_grad" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark ($(cache)["x"] = Optimization.MinCostFlow.primal_from_dual($(problem), $(μ);
                ϵ=1e-4, ε=1e-10, max_iter=2000)) evals=1 samples=1 seconds=1e-9
            result.time_min_grad = minimum(bm).time
        else
            cache["x"] = Optimization.MinCostFlow.primal_from_dual(problem, μ;
                ϵ=1e-4, ε=1e-10, max_iter=2000)
            result.time_min_grad = error_v(Float64)
        end
        result.f_min_grad = get_f(cache["x"])
        result.df_min_grad = get_df(cache["x"])
        result.x_min_grad = norm(cache["x"])
        let Exb = E*cache["x"]-b
            result.unf_min_grad = norm(Exb)
            result.unf_min_grad_1 = norm(Exb, 1)
        end
    catch err
        println("min grad : ($m, $n, $singular, $active) : $err")
        result.time_min_grad = error_v(V)
        result.f_min_grad = error_v(V)
        result.df_min_grad = error_v(V)
        result.x_min_grad = error_v(V)
        result.unf_min_grad = error_v(V)
        result.unf_min_grad_1 = error_v(V)
    end

    # benchmark EKHeuristic (after min-norm ϵ-subgradient)
    println("mg_EK")
    try
        if !("mg_EK" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["mg_EK"] =
                    Optimization.MinCostFlow.EKHeuristic($(problem), $(cache)["x"]; ϵ=1e-16) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_mg_EK = minimum(bm).time
        else
            cache["mg_EK"] =
                Optimization.MinCostFlow.EKHeuristic(problem, cache["x"]; ϵ=1e-16) |>
                heu -> (init!(heu); run!(heu))
            result.time_mg_EK = error_v(Float64)
        end
        cache["x_mg_EK"], cache["Δ_mg_EK"] = cache["mg_EK"]
        result.f_mg_EK = get_f(cache["x_mg_EK"])
        result.df_mg_EK = get_df(cache["x_mg_EK"])
        result.x_mg_EK = norm(cache["x_mg_EK"])
        let Exb = E*cache["x_mg_EK"]-b
            result.unf_mg_EK = norm(Exb)
            result.unf_mg_EK_1 = norm(Exb, 1)
        end
    catch err
        println("mg EK : ($m, $n, $singular, $active) : $err")
        result.time_mg_EK = error_v(V)
        result.f_mg_EK = error_v(V)
        result.df_mg_EK = error_v(V)
        result.x_mg_EK = error_v(V)
        result.unf_mg_EK = error_v(V)
        result.unf_mg_EK_1 = error_v(V)
    end

    # benchmark EKHeuristic
    println("EK")
    try
        if !("EK" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["EK"] =
                    Optimization.MinCostFlow.EKHeuristic($(problem), $(x); ϵ=1e-16) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_EK = minimum(bm).time
        else
            cache["EK"] =
                Optimization.MinCostFlow.EKHeuristic(problem, x; ϵ=1e-16) |>
                heu -> (init!(heu); run!(heu))
            result.time_EK = error_v(Float64)
        end
        cache["x_EK"], cache["Δ_EK"] = cache["EK"]
        result.f_EK = get_f(cache["x_EK"])
        result.df_EK = get_df(cache["x_EK"])
        result.x_EK = norm(cache["x_EK"])
        let Exb = E*cache["x_EK"]-b
            result.unf_EK = norm(Exb)
            result.unf_EK_1 = norm(Exb, 1)
        end
    catch err
        println("EK : ($m, $n, $singular, $active) : $err")
        result.time_EK = error_v(V)
        result.f_EK = error_v(V)
        result.df_EK = error_v(V)
        result.x_EK = error_v(V)
        result.unf_EK = error_v(V)
        result.unf_EK_1 = error_v(V)
    end

    # benchmark strict SPEKHeuristic (after min-norm ϵ-subgradient)
    println("mg_SPEK")
    try
        if !("mg_SPEK" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["mg_SPEK"] =
                    Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(cache)["x"]+$(q), $(cache)["x"]; ϵ=1e-14, ϵₚ=1e-10) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_mg_SPEK = minimum(bm).time
        else
            cache["mg_SPEK"] =
                Optimization.MinCostFlow.SPEKHeuristic(problem, Q*cache["x"]+q, cache["x"]; ϵ=1e-14, ϵₚ=1e-10) |>
                heu -> (init!(heu); run!(heu))
            result.time_mg_SPEK = error_v(Float64)
        end
        cache["x_mg_SPEK"], cache["Δ_mg_SPEK"] = cache["mg_SPEK"]
        result.f_mg_SPEK = get_f(cache["x_mg_SPEK"])
        result.df_mg_SPEK = get_df(cache["x_mg_SPEK"])
        result.x_mg_SPEK = norm(cache["x_mg_SPEK"])
        let Exb = E*cache["x_mg_SPEK"]-b
            result.unf_mg_SPEK = norm(Exb)
            result.unf_mg_SPEK_1 = norm(Exb, 1)
        end
    catch err
        println("mg SPEK : ($m, $n, $singular, $active) : $err")
        result.time_mg_SPEK = error_v(V)
        result.f_mg_SPEK = error_v(V)
        result.df_mg_SPEK = error_v(V)
        result.x_mg_SPEK = error_v(V)
        result.unf_mg_SPEK = error_v(V)
        result.unf_mg_SPEK_1 = error_v(V)
    end

    # benchmark strict SPEKHeuristic
    println("SPEK")
    try
        if !("SPEK" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["SPEK"] =
                    Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(x)+$(q), $(x); ϵ=1e-14, ϵₚ=1e-10) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_SPEK = minimum(bm).time
        else
            cache["SPEK"] =
                Optimization.MinCostFlow.SPEKHeuristic(problem, Q*x+q, x; ϵ=1e-14, ϵₚ=1e-10) |>
                heu -> (init!(heu); run!(heu))
            result.time_SPEK = error_v(Float64)
        end
        cache["x_SPEK"], cache["Δ_SPEK"] = cache["SPEK"]
        result.f_SPEK = get_f(cache["x_SPEK"])
        result.df_SPEK = get_df(cache["x_SPEK"])
        result.x_SPEK = norm(cache["x_SPEK"])
        let Exb = E*cache["x_SPEK"]-b
            result.unf_SPEK = norm(Exb)
            result.unf_SPEK_1 = norm(Exb, 1)
        end
    catch err
        println("SPEK : ($m, $n, $singular, $active) : $err")
        result.time_SPEK = error_v(V)
        result.f_SPEK = error_v(V)
        result.df_SPEK = error_v(V)
        result.x_SPEK = error_v(V)
        result.unf_SPEK = error_v(V)
        result.unf_SPEK_1 = error_v(V)
    end

    # benchmark SPEKHeuristic (after min-norm ϵ-subgradient)
    println("mg_SPEKn")
    try
        if !("mg_SPEKn" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["mg_SPEKn"] =
                    Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(cache)["x"]+$(q), $(cache)["x"]; ϵ=1e-14, ϵₚ=1e-10, strict=false) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_mg_SPEKn = minimum(bm).time
        else
            cache["mg_SPEKn"] =
                Optimization.MinCostFlow.SPEKHeuristic(problem, Q*cache["x"]+q, cache["x"]; ϵ=1e-14, ϵₚ=1e-10, strict=false) |>
                heu -> (init!(heu); run!(heu))
            result.time_mg_SPEKn = error_v(Float64)
        end
        cache["x_mg_SPEKn"], cache["Δ_mg_SPEKn"] = cache["mg_SPEKn"]
        result.f_mg_SPEKn = get_f(cache["x_mg_SPEKn"])
        result.df_mg_SPEKn = get_df(cache["x_mg_SPEKn"])
        result.x_mg_SPEKn = norm(cache["x_mg_SPEKn"])
        let Exb = E*cache["x_mg_SPEKn"]-b
            result.unf_mg_SPEKn = norm(Exb)
            result.unf_mg_SPEKn_1 = norm(Exb, 1)
        end
    catch err
        println("mg SPEKn : ($m, $n, $singular, $active) : $err")
        result.time_mg_SPEKn = error_v(V)
        result.f_mg_SPEKn = error_v(V)
        result.df_mg_SPEKn = error_v(V)
        result.x_mg_SPEKn = error_v(V)
        result.unf_mg_SPEKn = error_v(V)
        result.unf_mg_SPEKn_1 = error_v(V)
    end

    # benchmark SPEKHeuristic (after min-norm ϵ-subgradient)
    println("SPEKn")
    try
        if !("SPEKn" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["SPEKn"] =
                    Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(x)+$(q), $(x); ϵ=1e-14, ϵₚ=1e-10, strict=false) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_SPEKn = minimum(bm).time
        else
            cache["SPEKn"] =
                Optimization.MinCostFlow.SPEKHeuristic(problem, Q*x+q, x; ϵ=1e-14, ϵₚ=1e-10, strict=false) |>
                heu -> (init!(heu); run!(heu))
            result.time_SPEKn = error_v(Float64)
        end
        cache["x_SPEKn"], cache["Δ_SPEKn"] = cache["SPEKn"]
        result.f_SPEKn = get_f(cache["x_SPEKn"])
        result.df_SPEKn = get_df(cache["x_SPEKn"])
        result.x_SPEKn = norm(cache["x_SPEKn"])
        let Exb = E*cache["x_SPEKn"]-b
            result.unf_SPEKn = norm(Exb)
            result.unf_SPEKn_1 = norm(Exb, 1)
        end
    catch err
        println("SPEKn : ($m, $n, $singular, $active) : $err")
        result.time_SPEKn = error_v(V)
        result.f_SPEKn = error_v(V)
        result.df_SPEKn = error_v(V)
        result.x_SPEKn = error_v(V)
        result.unf_SPEKn = error_v(V)
        result.unf_SPEKn_1 = error_v(V)
    end

    # benchmark JuMP
    println("JuMP")
    try
        if !("JuMP" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark ($(cache)["x_JuMP"] =
                get_solution_quadratic_box_constrained($(problem), zeros(Float64, $(n)))) evals=1 samples=1 seconds=1e-9
            result.time_JuMP = minimum(bm).time
        else
            cache["x_JuMP"] =
                get_solution_quadratic_box_constrained(problem, zeros(Float64, n))
            result.time_JuMP = error_v(Float64)
        end
        cache["x_JuMP"] = max.(min.(cache["x_JuMP"], u), l)
        result.f_JuMP = get_f(cache["x_JuMP"])
        result.df_JuMP = get_df(cache["x_JuMP"])
        result.x_JuMP = norm(cache["x_JuMP"])
        let Exb = E*cache["x_JuMP"]-b
            result.unf_JuMP = norm(Exb)
            result.unf_JuMP_1 = norm(Exb, 1)
        end
    catch err
        println("JuMP : ($m, $n, $singular, $active) : $err")
        result.time_JuMP = error_v(V)
        result.f_JuMP = error_v(V)
        result.df_JuMP = error_v(V)
        result.x_JuMP = error_v(V)
        result.unf_JuMP = error_v(V)
        result.unf_JuMP_1 = error_v(V)
    end


    # benchmark EKHeuristic (after JuMP)
    println("JuMP_EK")
    try
        if !("JuMP_EK" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["JuMP_EK"] =
                    Optimization.MinCostFlow.EKHeuristic($(problem), $(cache)["x_JuMP"]; ϵ=1e-16) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_JuMP_EK = minimum(bm).time
        else
            cache["JuMP_EK"] =
                Optimization.MinCostFlow.EKHeuristic(problem, cache["x_JuMP"]; ϵ=1e-16) |>
                heu -> (init!(heu); run!(heu))
            result.time_JuMP_EK = error_v(Float64)
        end
        cache["x_JuMP_EK"], cache["Δ_JuMP_EK"] = cache["JuMP_EK"]
        result.f_JuMP_EK = get_f(cache["x_JuMP_EK"])
        result.df_JuMP_EK = get_df(cache["x_JuMP_EK"])
        result.x_JuMP_EK = norm(cache["x_JuMP_EK"])
        let Exb = cache["Δ_JuMP_EK"]
            result.unf_JuMP_EK = norm(Exb)
            result.unf_JuMP_EK_1 = norm(Exb, 1)
        end
    catch err
        println("JuMP EK : ($m, $n, $singular, $active) : $err")
        result.time_JuMP_EK = error_v(V)
        result.f_JuMP_EK = error_v(V)
        result.df_JuMP_EK = error_v(V)
        result.x_JuMP_EK = error_v(V)
        result.unf_JuMP_EK = error_v(V)
        result.unf_JuMP_EK_1 = error_v(V)
    end

    # benchmark strict SPEKHeuristic (after JuMP)
    println("JuMP_SPEK")
    try
        if !("JuMP_SPEK" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["JuMP_SPEK"] =
                    Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(cache)["x_JuMP"]+$(q), $(cache)["x_JuMP"]; ϵ=1e-14, ϵₚ=1e-10) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_JuMP_SPEK = minimum(bm).time
        else
            cache["JuMP_SPEK"] =
                Optimization.MinCostFlow.SPEKHeuristic(problem, Q*cache["x_JuMP"]+q, cache["x_JuMP"]; ϵ=1e-14, ϵₚ=1e-10) |>
                heu -> (init!(heu); run!(heu))
            result.time_JuMP_SPEK = error_v(Float64)
        end
        cache["x_JuMP_SPEK"], cache["Δ_JuMP_SPEK"] = cache["JuMP_SPEK"]
        result.f_JuMP_SPEK = get_f(cache["x_JuMP_SPEK"])
        result.df_JuMP_SPEK = get_df(cache["x_JuMP_SPEK"])
        result.x_JuMP_SPEK = norm(cache["x_JuMP_SPEK"])
        let Exb = cache["Δ_JuMP_SPEK"]
            result.unf_JuMP_SPEK = norm(Exb)
            result.unf_JuMP_SPEK_1 = norm(Exb, 1)
        end
    catch err
        println("JuMP SPEK : ($m, $n, $singular, $active) : $err")
        result.time_JuMP_SPEK = error_v(V)
        result.f_JuMP_SPEK = error_v(V)
        result.df_JuMP_SPEK = error_v(V)
        result.x_JuMP_SPEK = error_v(V)
        result.unf_JuMP_SPEK = error_v(V)
        result.unf_JuMP_SPEK_1 = error_v(V)
    end

    # benchmark SPEKHeuristic (after min-norm ϵ-subgradient)
    println("JuMP_SPEKn")
    try
        if !("JuMP_SPEKn" in todos)
            error("SKIP")
        end
        if "time" in todos
            bm = @benchmark (
                $(cache)["JuMP_SPEKn"] =
                    Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(cache)["x_JuMP"]+$(q), $(cache)["x_JuMP"]; ϵ=1e-14, ϵₚ=1e-10, strict=false) |>
                    heu -> (init!(heu); run!(heu))) evals=1 samples=1 seconds=1e-9
            result.time_JuMP_SPEKn = minimum(bm).time
        else
            cache["JuMP_SPEKn"] =
                Optimization.MinCostFlow.SPEKHeuristic(problem, Q*cache["x_JuMP"]+q, cache["x_JuMP"]; ϵ=1e-14, ϵₚ=1e-10, strict=false) |>
                heu -> (init!(heu); run!(heu))
            result.time_JuMP_SPEKn = error_v(Float64)
        end
        cache["x_JuMP_SPEKn"], cache["Δ_JuMP_SPEKn"] = cache["JuMP_SPEKn"]
        result.f_JuMP_SPEKn = get_f(cache["x_JuMP_SPEKn"])
        result.df_JuMP_SPEKn = get_df(cache["x_JuMP_SPEKn"])
        result.x_JuMP_SPEKn = norm(cache["x_JuMP_SPEKn"])
        let Exb = cache["Δ_JuMP_SPEKn"]
            result.unf_JuMP_SPEKn = norm(Exb)
            result.unf_JuMP_SPEKn_1 = norm(Exb, 1)
        end
    catch err
        println("JuMP SPEKn : ($m, $n, $singular, $active) : $err")
        result.time_JuMP_SPEKn = error_v(V)
        result.f_JuMP_SPEKn = error_v(V)
        result.df_JuMP_SPEKn = error_v(V)
        result.x_JuMP_SPEKn = error_v(V)
        result.unf_JuMP_SPEKn = error_v(V)
        result.unf_JuMP_SPEKn_1 = error_v(V)
    end

    result, is, Ls
end
function run_bench(::Type{V}, ::Type{SG}, m::Int64, n::Int64, singular::Int64, active::Int64,
    args...; kwargs...) where {V, SG<:SubgradientMethod}

    if typeof(singular) <: AbstractFloat
        singular = min(ceil(eltype(m), n*singular), n)
    end
    if typeof(active) <: AbstractFloat
        active = min(ceil(eltype(m), n*active), n)
    end

    problem = generate_quadratic_min_cost_flow_boxed_problem(
        V, m, n; singular=singular, active=active)

    run_bench(V, problem, singular, active, SG, args...; kwargs...)
end
function run_bench(io::IO, ::Type{V}, ::Type{SG}, args...; kwargs...) where {V, SG<:SubgradientMethod}
    result = run_bench(V, SG, args...; kwargs...)
    println(io, result)
    flush(io)

    result
end

function random_benches(::Type{V}, filename::String, times, nodes::AbstractArray,
    ::Type{SG}=Subgradient.NesterovMomentum,
    sg_args::Tuple=(),
    sg_kwargs::NamedTuple=(α=1.0, β=0.99),
    sg_update=sg->sg.α/=2.0;
    max_iter=4000,
    max_hiter=40,
    restart=true,
    todos=Set(["dual", "min_grad", "mg_EK", "mg_SPEK", "mg_SPEKn"])) where {V, SG<:SubgradientMethod}

    open(filename, "a") do io
        legenda(io, BenchResult{V})
        for i in 1:times
            m = rand(nodes)
            n = rand(m:m*(m-1)÷2)
            singular = rand()
            active = rand()
            run_bench(io, V, SG, m, n, singular, active,
                sg_args,
                sg_kwargs,
                sg_update;
                max_iter=max_iter,
                max_hiter=max_hiter,
                restart=restart,
                todos=todos)
        end
    end
end

function read_bench(filename::String; type=Float64)
    data = BenchResult{type}[]
    open(filename, "r") do io
        readline(io)
        while !eof(io)
            line = readline(io)
            if length(line) < 5 || startswith(line, "nodes")
                continue
            end
            push!(data, parse(BenchResult{type}, line))
        end
    end
    data
end
