"""
### Tuner
Naive tentative automatic hyperparameter search, via progressive localization
of best parameter.

"""
module Hyper

export WithParameterSearch,
    set!,
    run!
    using Parameters
using ..Optimization
using ..Optimization.Utils

import ..Optimization: set!, run!

# TODO: select appropriate type for the explorer
mutable struct WithParameterSearch{P<:OptimizationProblem, A<:OptimizationAlgorithm{P}, S<:LocalizationMethod} <: OptimizationAlgorithm{P}
    algorithm::A
    searcher::S
    objective::Union{String, Nothing}
    param_ranges                    # ranges of the parameters for searching
    searcher_iter                   # number of times the algorithm is run with
                                    # different parameters
    algorithm_iter                  # number of iterations in the algorithm
    algorithm_iter_per_search       # iterations per search
    δ                               # fraction of iterations evaluated in search

    params                          # WIP
    function WithParameterSearch{P, A, S}(;
        algorithm::A,
        searcher::S,
        objective::Union{String, Nothing}=nothing,
        cmp=isless,
        param_ranges=nothing,
        searcher_iter=nothing,
        algorithm_iter=nothing,
        algorithm_iter_per_search=nothing,
        δ=nothing) where {P<:OptimizationProblem, S<:LocalizationMethod, A<:OptimizationAlgorithm{P}}

        M = new{P, A, S}(algorithm, searcher, objective)
        if objective !== nothing
            M.searcher.cmp = (a::OptimizationResult{P}, b::OptimizationResult{P}) ->
                cmp(a.result[objective], b.result[objective])
        end
        M.param_ranges = algorithm.localization.params
        M.params = Dict(:searcher_iter => [0.0, Inf], :δ => [0.0, 1.0])
        set!(M,
            param_ranges=param_ranges,
            searcher_iter=searcher_iter,
            algorithm_iter=algorithm_iter,
            algorithm_iter_per_search=algorithm_iter_per_search,
            δ=δ)
    end
end
function set!(M::WithParameterSearch{<:OptimizationProblem, <:OptimizationAlgorithm, <:LocalizationMethod};
    param_ranges=nothing,
    searcher_iter=nothing,
    algorithm_iter=nothing,
    algorithm_iter_per_search=nothing,
    δ=nothing)

    @some M.param_ranges = param_ranges
    @some M.searcher_iter = searcher_iter
    @some M.algorithm_iter = algorithm_iter
    @some M.algorithm_iter_per_search = algorithm_iter_per_search
    @some M.δ = δ
    M
end
function run!(
    M::WithParameterSearch{P, A, NelderMead},
    𝔓::P;
    memoranda=Set([])) where {P<:OptimizationProblem, A<:OptimizationAlgorithm}

    @init_memoria memoranda
    @unpack algorithm, searcher, param_ranges, searcher_iter,
            algorithm_iter, algorithm_iter_per_search, δ = M
    localization = algorithm.localization

    # Simplex for Nelder Mead
    n = length(param_ranges)
    S = []

    function set_params!(algorithm, params)
        localization = algorithm.localization
        for p in params
            set_param!(localization, p...)
        end
        algorithm
    end

    params = Dict([(param, r[1]) for (param, r) in param_ranges])
    push!(S, deepcopy(params))
    for (s, r) in params
        push!(S, deepcopy(params))
        S[end][s] = param_ranges[s][2]
    end

    function gen_f(algorithm, iter)
        function f(params)
            algorithm′ = deepcopy(algorithm)
            set_params!(algorithm′, params)
            algorithm′.max_iter = iter
            # TODO: memoria
            # TODO: verbosity
            run!(algorithm′, 𝔓, memoranda=memoranda)
        end
        f
    end

    iter_to_search = Int(floor(algorithm_iter_per_search / δ))
    n_searches = algorithm_iter ÷ iter_to_search
    iter_with_no_search = iter_to_search - algorithm_iter_per_search
    algorithm_best = algorithm
    result_best, params_best = nothing, nothing
    for i in 1:n_searches
        f = gen_f(algorithm_best, algorithm_iter_per_search)
        init!(searcher, f, S)
        params_best, result_best = searcher.S[1], searcher.y[1]
        for j in 1:searcher_iter
            step!(searcher, f) |>
                x -> begin
                    @memento params_best = x[1]
                    @memento result_best = x[2]
                end
        end
        S = searcher.S
        set_params!(algorithm_best, params_best)
        set!(algorithm_best, result_best)
        algorithm_best.max_iter = iter_with_no_search
        @memento result_best = run!(algorithm_best, 𝔓, memoranda)
        set!(algorithm_best, result_best)
    end
    hyper_result = @get_result params_best
    OptimizationResult{P}(
        memoria=merge(best_result.memoria, @get_memoria),
        result=merge(best_result.result, hyper_result))
end

end # end module Hyper
