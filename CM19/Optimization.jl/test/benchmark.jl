# Benchmark Subgradient
# nodes arcs singular active
# L_lb L_min_grad L_EK L_SPEK L_SPEKn L_Ipopt L_Ipopt_EK L_Ipopt_SPEK L_Ipopt_SPEKn
# time_dual time_min_grad time_EK time_SPEK time_SPEKn time_Ipopt time_Ipopt_EK time_Ipopt_SPEK time_Ipopt_SPEKn
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
    x_mg_SPEK::V
    x_mg_SPEKn::V
    x_Ipopt::V
    x_Ipopt_EK::V
    x_Ipopt_SPEK::V
    x_Ipopt_SPEKn::V

    f_dual::V
    f_min_grad::V
    f_mg_EK::V
    f_mg_SPEK::V
    f_mg_SPEKn::V
    f_Ipopt::V
    f_Ipopt_EK::V
    f_Ipopt_SPEK::V
    f_Ipopt_SPEKn::V

    df_dual::V
    df_min_grad::V
    df_mg_EK::V
    df_mg_SPEK::V
    df_mg_SPEKn::V
    df_Ipopt::V
    df_Ipopt_EK::V
    df_Ipopt_SPEK::V
    df_Ipopt_SPEKn::V

    time_dual::Float64
    time_min_grad::Float64
    time_mg_EK::Float64
    time_mg_SPEK::Float64
    time_mg_SPEKn::Float64
    time_Ipopt::Float64
    time_Ipopt_EK::Float64
    time_Ipopt_SPEK::Float64
    time_Ipopt_SPEKn::Float64

    unf_dual::V
    unf_min_grad::V
    unf_mg_EK::V
    unf_mg_SPEK::V
    unf_mg_SPEKn::V
    unf_Ipopt::V
    unf_Ipopt_EK::V
    unf_Ipopt_SPEK::V
    unf_Ipopt_SPEKn::V

    unf_dual_1::V
    unf_min_grad_1::V
    unf_mg_EK_1::V
    unf_mg_SPEK_1::V
    unf_mg_SPEKn_1::V
    unf_Ipopt_1::V
    unf_Ipopt_EK_1::V
    unf_Ipopt_SPEK_1::V
    unf_Ipopt_SPEKn_1::V

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

function run_bench(::Type{V}, m, n, singular, active;
    max_iter=4000,
    decay=2.0,
    max_hiter=40,
    restart=true) where {V}

    if typeof(singular) <: AbstractFloat
        singular = min(ceil(eltype(m), n*singular), n)
    end
    if typeof(active) <: AbstractFloat
        active = min(ceil(eltype(m), n*active), n)
    end
    result = BenchResult{V}(m, n, singular, active)

    μ₀ = rand(m)
    subgradient = Subgradient.NesterovMomentum(α=1.0, β=0.99);
    algorithm = QMCFBPAlgorithmD1SG(;
        localization=subgradient,
        verbosity=1,
        max_iter=max_iter,
        μ₀=μ₀,
        ε=1e-6,
        ϵ=1e-12);
    test = get_test(algorithm;
        m=m,
        n=n,
        singular=singular,
        active=active,
        type=V);
    problem = test.problem;
    @unpack Q, q, l, u, E, b = problem;
    Q╲ = view(Q, [CartesianIndex(i, i) for i in 1:size(Q, 1)]);

    function runtest()
        for i in 1:max_hiter
            print("$i ")
            run!(test);
            set!(algorithm, test.result);  # set initial status of algorithm
            algorithm.stopped = !restart;     # not stopped implies re-init subgradient method
            subgradient.α /= decay
        end
        println()
    end

    get_f = x -> x'*(0.5Q*x + q)
    get_df = x -> norm(Q*x + q)
    cache = Dict{String, Any}()

    # benchmark subgradient method
    println("dual")
    bm = @benchmark $(runtest)() evals=1 samples=1
    μ = test.result.result["μ_best"]
    result.time_dual = minimum(bm).time
    result.f_dual = test.result.result["L_best"]
    let x = Optimization.MinCostFlow.primal_from_dual(problem, μ; ϵ=0.0, ε=1e-10, max_iter=2000)
        result.df_dual = get_df(x)
        result.x_dual = norm(x)
        let Exb = E*x-b
            result.unf_dual = norm(Exb)
            result.unf_dual_1 = norm(Exb, 1)
        end
    end

    # benchmark min-norm ϵ-subgradient
    println("min_grad")
    bm = @benchmark ($(cache)["x"] = Optimization.MinCostFlow.primal_from_dual($(problem), $(μ);
        ϵ=1e-4, ε=1e-10, max_iter=2000)) evals=1 samples=1
    result.time_min_grad = minimum(bm).time
    result.f_min_grad = get_f(cache["x"])
    result.df_min_grad = get_df(cache["x"])
    result.x_min_grad = norm(cache["x"])
    let Exb = E*cache["x"]-b
        result.unf_min_grad = norm(Exb)
        result.unf_min_grad_1 = norm(Exb, 1)
    end

    # benchmark EKHeuristic (after min-norm ϵ-subgradient)
    println("mg_EK")
    bm = @benchmark (
        $(cache)["mg_EK"] =
            Optimization.MinCostFlow.EKHeuristic($(problem), $(cache)["x"]; ϵ=1e-16) |>
            heu -> (init!(heu); run!(heu))) evals=1 samples=1
    cache["x_mg_EK"], cache["Δ_mg_EK"] = cache["mg_EK"]
    result.time_mg_EK = minimum(bm).time
    result.f_mg_EK = get_f(cache["x_mg_EK"])
    result.df_mg_EK = get_df(cache["x_mg_EK"])
    result.x_mg_EK = norm(cache["x_mg_EK"])
    let Exb = E*cache["x_mg_EK"]-b
        result.unf_mg_EK = norm(Exb)
        result.unf_mg_EK_1 = norm(Exb, 1)
    end

    # benchmark strict SPEKHeuristic (after min-norm ϵ-subgradient)
    println("mg_SPEK")
    try
        bm = @benchmark (
            $(cache)["mg_SPEK"] =
                Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(cache)["x"]+$(q), $(cache)["x"]; ϵ=1e-14, ϵₚ=1e-10) |>
                heu -> (init!(heu); run!(heu))) evals=1 samples=1
        cache["x_mg_SPEK"], cache["Δ_mg_SPEK"] = cache["mg_SPEK"]
        result.time_mg_SPEK = minimum(bm).time
        result.f_mg_SPEK = get_f(cache["x_mg_SPEK"])
        result.df_mg_SPEK = get_df(cache["x_mg_SPEK"])
        result.x_mg_SPEK = norm(cache["x_mg_SPEK"])
        let Exb = E*cache["x_mg_SPEK"]-b
            result.unf_mg_SPEK = norm(Exb)
            result.unf_mg_SPEK_1 = norm(Exb, 1)
        end
    catch err
        println("SPEK : ($m, $n, $singular, $active) : $err")
        result.time_mg_SPEK = error_v(V)
        result.f_mg_SPEK = error_v(V)
        result.df_mg_SPEK = error_v(V)
        result.x_mg_SPEK = error_v(V)
        result.unf_mg_SPEK = error_v(V)
        result.unf_mg_SPEK_1 = error_v(V)
    end

    # benchmark SPEKHeuristic (after min-norm ϵ-subgradient)
    println("mg_SPEKn")
    try
        bm = @benchmark (
            $(cache)["mg_SPEKn"] =
                Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(cache)["x"]+$(q), $(cache)["x"]; ϵ=1e-14, ϵₚ=1e-10, strict=false) |>
                heu -> (init!(heu); run!(heu))) evals=1 samples=1
        cache["x_mg_SPEKn"], cache["Δ_mg_SPEKn"] = cache["mg_SPEKn"]
        result.time_mg_SPEKn = minimum(bm).time
        result.f_mg_SPEKn = get_f(cache["x_mg_SPEKn"])
        result.df_mg_SPEKn = get_df(cache["x_mg_SPEKn"])
        result.x_mg_SPEKn = norm(cache["x_mg_SPEKn"])
        let Exb = E*cache["x_mg_SPEKn"]-b
            result.unf_mg_SPEKn = norm(Exb)
            result.unf_mg_SPEKn_1 = norm(Exb, 1)
        end
    catch err
        println("SPEKn : ($m, $n, $singular, $active) : $err")
        result.time_mg_SPEKn = error_v(V)
        result.f_mg_SPEKn = error_v(V)
        result.df_mg_SPEKn = error_v(V)
        result.x_mg_SPEKn = error_v(V)
        result.unf_mg_SPEKn = error_v(V)
        result.unf_mg_SPEKn_1 = error_v(V)
    end

    # benchmark Ipopt
    println("Ipopt")
    bm = @benchmark ($(cache)["x_Ipopt"] =
        get_solution_quadratic_box_constrained($(problem), zeros(Float64, $(n)))) evals=1 samples=1
    cache["x_Ipopt"] = max.(min.(cache["x_Ipopt"], u), l)
    result.time_Ipopt = minimum(bm).time
    result.f_Ipopt = get_f(cache["x_Ipopt"])
    result.df_Ipopt = get_df(cache["x_Ipopt"])
    result.x_Ipopt = norm(cache["x_Ipopt"])
    let Exb = E*cache["x_Ipopt"]-b
        result.unf_Ipopt = norm(Exb)
        result.unf_Ipopt_1 = norm(Exb, 1)
    end


    # benchmark EKHeuristic (after Ipopt)
    println("Ipopt_EK")
    bm = @benchmark (
        $(cache)["Ipopt_EK"] =
            Optimization.MinCostFlow.EKHeuristic($(problem), $(cache)["x_Ipopt"]; ϵ=1e-16) |>
            heu -> (init!(heu); run!(heu))) evals=1 samples=1
    cache["x_Ipopt_EK"], cache["Δ_Ipopt_EK"] = cache["Ipopt_EK"]
    result.time_Ipopt_EK = minimum(bm).time
    result.f_Ipopt_EK = get_f(cache["x_Ipopt_EK"])
    result.df_Ipopt_EK = get_df(cache["x_Ipopt_EK"])
    result.x_Ipopt_EK = norm(cache["x_Ipopt_EK"])
    let Exb = cache["Δ_Ipopt_EK"]
        result.unf_Ipopt_EK = norm(Exb)
        result.unf_Ipopt_EK_1 = norm(Exb, 1)
    end

    # benchmark strict SPEKHeuristic (after Ipopt)
    println("Ipopt_SPEK")
    try
        error("Ipopt_SPEK : SKIP")
        bm = @benchmark (
            $(cache)["Ipopt_SPEK"] =
                Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(cache)["x_Ipopt"]+$(q), $(cache)["x_Ipopt"]; ϵ=1e-14, ϵₚ=1e-10) |>
                heu -> (init!(heu); run!(heu))) evals=1 samples=1
        cache["x_Ipopt_SPEK"], cache["Δ_Ipopt_SPEK"] = cache["Ipopt_SPEK"]
        result.time_Ipopt_SPEK = minimum(bm).time
        result.f_Ipopt_SPEK = get_f(cache["x_Ipopt_SPEK"])
        result.df_Ipopt_SPEK = get_df(cache["x_Ipopt_SPEK"])
        result.x_Ipopt_SPEK = norm(cache["x_Ipopt_SPEK"])
        let Exb = cache["Δ_Ipopt_SPEK"]
            result.unf_Ipopt_SPEK = norm(Exb)
            result.unf_Ipopt_SPEK_1 = norm(Exb, 1)
        end
    catch err
        println("Ipopt SPEK : ($m, $n, $singular, $active) : $err")
        result.time_Ipopt_SPEK = error_v(V)
        result.f_Ipopt_SPEK = error_v(V)
        result.df_Ipopt_SPEK = error_v(V)
        result.x_Ipopt_SPEK = error_v(V)
        result.unf_Ipopt_SPEK = error_v(V)
        result.unf_Ipopt_SPEK_1 = error_v(V)
    end

    # benchmark SPEKHeuristic (after min-norm ϵ-subgradient)
    println("Ipopt_SPEKn")
    try
        error("Ipopt_SPEKn : SKIP")
        bm = @benchmark (
            $(cache)["Ipopt_SPEKn"] =
                Optimization.MinCostFlow.SPEKHeuristic($(problem), $(Q)*$(cache)["x_Ipopt"]+$(q), $(cache)["x_Ipopt"]; ϵ=1e-14, ϵₚ=1e-10, strict=false) |>
                heu -> (init!(heu); run!(heu))) evals=1 samples=1
        cache["x_Ipopt_SPEKn"], cache["Δ_Ipopt_SPEKn"] = cache["Ipopt_SPEKn"]
        result.time_Ipopt_SPEKn = minimum(bm).time
        result.f_Ipopt_SPEKn = get_f(cache["x_Ipopt_SPEKn"])
        result.df_Ipopt_SPEKn = get_df(cache["x_Ipopt_SPEKn"])
        result.x_Ipopt_SPEKn = norm(cache["x_Ipopt_SPEKn"])
        let Exb = cache["Δ_Ipopt_SPEKn"]
            result.unf_Ipopt_SPEKn = norm(Exb)
            result.unf_Ipopt_SPEKn_1 = norm(Exb, 1)
        end
    catch err
        println("Ipopt SPEKn : ($m, $n, $singular, $active) : $err")
        result.time_Ipopt_SPEKn = error_v(V)
        result.f_Ipopt_SPEKn = error_v(V)
        result.df_Ipopt_SPEKn = error_v(V)
        result.x_Ipopt_SPEKn = error_v(V)
        result.unf_Ipopt_SPEKn = error_v(V)
        result.unf_Ipopt_SPEKn_1 = error_v(V)
    end

    result
end

function run_bench(::Type{V}, io::IO, args...; kwargs...) where {V}
    result = run_bench(V, args...; kwargs...)
    println(io, result)
    flush(io)
end

function random_benches(::Type{V}, filename::String, times, nodes::AbstractArray) where {V}
    open(filename, "a") do io
        legenda(io, BenchResult{V})
        for i in 1:times
            m = rand(nodes)
            n = rand(m:m*(m-1)÷2)
            singular = rand()
            active = rand()
            run_bench(V, io, m, n, singular, active)
        end
    end
end

function read_bench(filename::String; type=Float64)
    data = BenchResult{type}[]
    open(filename, "r") do io
        readline(io)
        while !eof(io)
            line = readline(io)
            push!(data, parse(BenchResult{type}, line))
        end
    end
    data
end
