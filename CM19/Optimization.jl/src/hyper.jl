"""
### Tuner
Naive tentative dynamic hyperparameter search, via progressive localization
of best parameter.

**Example**

```julia
subgradient = Subgradient.HarmonicErgodicPrimalStep(k=4, a=0.01, b=0.1)
algorithm = QMCFBPAlgorithmD1SG(
    localization=subgradient,
    verbosity=1,
    max_iter=1000,
    ε=1e-6,
    ϵ=1e-12)
searcher = NelderMead()
halgorithm = WithParameterSearch{QMCFBProblem, typeof(algorithm), NelderMead}(
    algorithm=algorithm,
    searcher=searcher,
    objective="L_best",
    cmp=(a, b)->a>b,
    searcher_iter=30,
    algorithm_iter=10000,
    algorithm_iter_per_search=200,
    δ=0.2,
    param_ranges=Dict(:a=>[1e-8, 1.0], :b=>[1e-8, 1.0]))
test = get_test(halgorithm, m=200, n=300, singular=10)
test.solver.options.memoranda = Set(["norm∂L_best", "L_best", "params_best", "result_best"])
run!(test)

```

"""
module Hyper

export WithParameterSearch,
    set!,
    run!,
    plot,
    plot!
using Parameters
using ..Optimization
using ..Optimization.Utils

import ..Optimization: set!, run!, plot, plot!
import Plots
import Plots: plot, plot!

mutable struct WithParameterSearch{P<:OptimizationProblem, A<:OptimizationAlgorithm{P}, S<:LocalizationMethod} <: OptimizationAlgorithm{P}
    algorithm::A
    searcher::S
    objective::Union{String, Nothing}
    fixed_params                    # parameters kept outside of the search
    param_ranges                    # ranges of the parameters for searching
    searcher_iter                   # number of times the algorithm is run with
                                    # different parameters
    algorithm_iter                  # number of iterations in the algorithm
    algorithm_iter_per_search       # iterations per search
    δ                               # fraction of iterations evaluated in search
    restart_params                  # reset the params simplex in each cycle

    params                          # WIP
    function WithParameterSearch{P, A, S}(;
        algorithm::A,
        searcher::S,
        objective::Union{String, Nothing}=nothing,
        cmp=isless,
        fixed_params=Dict(),
        param_ranges=nothing,
        searcher_iter=nothing,
        algorithm_iter=nothing,
        algorithm_iter_per_search=nothing,
        δ=nothing,
        restart_params=false) where {P<:OptimizationProblem, S<:LocalizationMethod, A<:OptimizationAlgorithm{P}}

        M = new{P, A, S}(algorithm, searcher, objective)
        if objective !== nothing
            M.searcher.cmp = (a::OptimizationResult{P}, b::OptimizationResult{P}) ->
                cmp(a.result[objective], b.result[objective])
        end
        M.param_ranges = algorithm.localization.params

        M.params = Dict(:searcher_iter => [0.0, Inf], :δ => [0.0, 1.0])
        set!(M,
            fixed_params=fixed_params,
            param_ranges=param_ranges,
            searcher_iter=searcher_iter,
            algorithm_iter=algorithm_iter,
            algorithm_iter_per_search=algorithm_iter_per_search,
            δ=δ,
            restart_params=restart_params)
    end
end
function set!(M::WithParameterSearch{<:OptimizationProblem, <:OptimizationAlgorithm, <:LocalizationMethod};
    fixed_params=Dict(),
    param_ranges=nothing,
    searcher_iter=nothing,
    algorithm_iter=nothing,
    algorithm_iter_per_search=nothing,
    δ=nothing,
    restart_params=nothing)

    @some M.param_ranges = param_ranges
    M.fixed_params = fixed_params
    (k->if haskey(M.param_ranges, k) pop!(M.param_ranges, k) end).(keys(fixed_params))
    @some M.searcher_iter = searcher_iter
    @some M.algorithm_iter = algorithm_iter
    @some M.algorithm_iter_per_search = algorithm_iter_per_search
    @some M.δ = δ
    @some M.restart_params = restart_params
    M
end
function run!(
    M::WithParameterSearch{P, A, NelderMead},
    problem::P;
    memoranda=Set([])) where {P<:OptimizationProblem, A<:OptimizationAlgorithm}

    @init_memoria memoranda
    @unpack algorithm, searcher, fixed_params, param_ranges, searcher_iter,
            algorithm_iter, algorithm_iter_per_search, δ, restart_params = M
    localization = algorithm.localization

    # Simplex for Nelder Mead
    n = length(param_ranges)
    S = []

    function set_params!(algorithm, params)
        for p in params
            set_param!(algorithm.localization, p...)
        end
        algorithm
    end

    # params = Dict([(param, r[1]) for (param, r) in param_ranges])
    params_k = [keys(param_ranges)...]
    params_r = [values(param_ranges)...]
    params_v = [r[1] for r in params_r]
    push!(S, deepcopy(params_v))
    for i in 1:length(params_v)
        push!(S, deepcopy(params_v))
        S[end][i] = params_r[i][2]
    end
    S₀ = deepcopy(S)

    function gen_f(algorithm, iter)
        function f(params_v′)
            algorithm′ = deepcopy(algorithm)
            set_params!(algorithm′, Dict(zip(params_k, params_v′)))
            println("Setting $(params_k) = $(params_v′)")
            algorithm′.max_iter = iter
            # TODO: memoria
            # TODO: verbosity
            run!(algorithm′, problem, memoranda=memoranda)
        end
        f
    end

    if δ ≤ 0.0
        iter_to_search = 0
    else
        iter_to_search = min(Int(floor(algorithm_iter_per_search / δ)), algorithm_iter)
    end

    n_searches = algorithm_iter ÷ iter_to_search
    iter_with_no_search = iter_to_search - algorithm_iter_per_search
    set_params!(algorithm, fixed_params)
    algorithm_best = algorithm
    result_best, params_best = nothing, nothing
    for i in 1:n_searches
        f = gen_f(algorithm_best, algorithm_iter_per_search)
        init!(searcher, f, S)
        params_best, result_best = Dict(zip(params_k, searcher.S[1])), searcher.y[1]
        for j in 1:searcher_iter
            step!(searcher, f) |>
                x -> begin
                    @memento params_best = Dict(zip(params_k, x[1]))
                    @memento result_best = x[2]
                end
            # set!(algorithm_best, result_best)   # trial
        end
        S = begin
            if restart_params
                deepcopy(S₀)
            else
                searcher.S
            end
        end
        set!(algorithm_best, result_best)
        # loky = algorithm_best.localization
        # @memento loky = loky
        set_params!(algorithm_best, params_best)
        algorithm_best.max_iter = iter_with_no_search
        @memento result_best = run!(algorithm_best, problem, memoranda=memoranda)
        set!(algorithm_best, result_best)
    end
    hyper_result = @get_result params_best
    OptimizationResult{P}(
        memoria=@get_memoria,
        result=merge(result_best.result, hyper_result))
end

function plot_helper(plotter,
    d::Array{Dict{K, V}, 1};
    abscissa::Union{Nothing, K, Array}=nothing,
    ordinatas::Union{Nothing, Array{K, 1}, K}=nothing,
    args...) where {K, V}

    if length(d) < 1 return Plots.plot() end

    if isa(ordinatas, Nothing)
        ordinatas = [k for k in keys(d[1]) if !isa(abscissa, K) || k!=abscissa]
    elseif isa(ordinatas, K)
        ordinatas = [ordinatas]
    end

    if isa(abscissa, Nothing)
        abscissa = [1:length(d);]
    elseif isa(abscissa, K)
        abscissa = (x->x[abscissa]).(d)
    end

    data = [[el[k] for el in d if haskey(el, k)] for k in ordinatas]
    label = permutedims(string.(ordinatas))
    #return Plots.plot(abscissa, data, label=label)
    plotter(abscissa, data; label=label, args...)
end
function plot(d::Array{Dict{K, V}, 1};
    abscissa::Union{Nothing, K, Array}=nothing,
    ordinatas::Union{Nothing, Array{K, 1}, K}=nothing) where {K, V}

    plot_helper(
        (x, y; label, args...) -> Plots.plot(x, y; label=label, args...),
        d;
        abscissa=abscissa,
        ordinatas=ordinatas)
end
function plot!(p::Plots.Plot,
    d::Array{Dict{K, V}, 1};
    abscissa::Union{Nothing, K, Array}=nothing,
    ordinatas::Union{Nothing, Array{K, 1}, K}=nothing) where {K, V}

    plot_helper(
        (x, y; label, args...) -> Plots.plot!(p, x, y; label=label, args...),
        d;
        abscissa=abscissa,
        ordinatas=ordinatas)
end
end # end module Hyper

"""
```julia
using Optimization; subgradient = Subgradient.HarmonicErgodicPrimalStep(k=4, a=0.01, b=0.1);
algorithm = QMCFBPAlgorithmD1SG(localization=subgradient, verbosity=1, max_iter=1000, ε=1e-6, ϵ=1e-12);
searcher = NelderMead();
halgorithm = WithParameterSearch{QMCFBProblem, typeof(algorithm), NelderMead}(algorithm=algorithm, searcher=searcher, objective="L_best", cmp=(a, b)->a>b, searcher_iter=30, algorithm_iter=10000, algorithm_iter_per_search=500, δ=0.5, fixed_params=[:k=>4], param_ranges=Dict(:a=>[1e-8, 1.0], :b=>[1e-8, 1.0]));
test = get_test(halgorithm, m=200, n=300, singular=10);
test.solver.options.memoranda = Set(["norm∂L_best", "L_best", "params_best", "result_best"]);
run!(test);
plot(Dict{Symbol, Float64}.(test.result.memoria["params_best"]))

```


"""
