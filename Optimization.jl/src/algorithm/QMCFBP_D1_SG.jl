# ------------------------- Subgradient Methods ------------------------- #
mutable struct QMCFBPAlgorithmD1SG <: OptimizationAlgorithm{QMCFBProblem}
    localization::SubgradientMethod
    verba               # verbosity utility
    max_iter            # max number of iterations
    ε
    ϵ
    μ₀                  # starting point
    heuristic_t         # heuristic struct type
    heuristic_each      #

    stopped             # if stopped do not initialise localization
    L̂                   # best upper bound
    x̂                   # ^ corresponding primal point
    memorabilia # set of the name of variables that can be recorded during execution
    QMCFBPAlgorithmD1SG(;
        localization=nothing,
        verbosity=nothing,
        my_verba=nothing,
        max_iter=nothing,
        ε=nothing,
        ϵ=nothing,
        μ₀=nothing,
        heuristic_t=Nothing,
        heuristic_each=1) = begin

        algorithm = new()
        algorithm.μ₀ = μ₀
        algorithm.stopped = false
        algorithm.L̂ = Inf
        algorithm.x̂ = nothing
        algorithm.memorabilia =
            Set([
                "x̅",           # → X(μ) see HarmonicErgodicPrimalStep ∈ Subgradient
                "x",           # primal coordinate
                "μ",           # constraint dual vector
                "L",           # Lagrangian
                "∂L",          # ∂L ∈ subgradient L(μ)
                "norm∂L",      # norm(∂L, Inf)
                "x_best",      # x for new best L
                "μ_best",      # μ for new best L
                "L_best",      # new best L
                "∂L_best",     # ∂L for each new best L
                "norm∂L_best", # norm(∂L, Inf) for each new best L
                "i_best",      # iteration counter for each new best L
                "L̂",           # upper bound by heuristic
                "x̂"])          # primal coor. in heuristic

        set!(
            algorithm,
            localization=localization,
            verbosity=verbosity,
            my_verba=my_verba,
            max_iter=max_iter,
            ε=ε,
            ϵ=ϵ,
            heuristic_t=heuristic_t,
            heuristic_each=heuristic_each)
    end
end
function set!(algorithm::QMCFBPAlgorithmD1SG;
    localization=nothing,
    verbosity=nothing,
    my_verba=nothing,
    max_iter=nothing,
    ε=nothing,
    ϵ=nothing,
    μ₀=nothing,
    heuristic_t=nothing,
    heuristic_each=nothing,
    x̂=nothing,
    L̂=nothing,
    stopped=nothing)

    @some algorithm.localization=localization
    if verbosity !== nothing
        algorithm.verba = ((level, message) -> verba(verbosity, level, message))
    end
    @some algorithm.verba=my_verba
    @some algorithm.max_iter=max_iter
    @some algorithm.ε=ε
    @some algorithm.ϵ=ϵ
    @some algorithm.μ₀=μ₀
    @some algorithm.heuristic_t = heuristic_t
    @some algorithm.heuristic_each = heuristic_each
    @some algorithm.x̂ = x̂
    @some algorithm.L̂ = L̂
    @some algorithm.stopped=stopped

    algorithm
end
function set!(algorithm::QMCFBPAlgorithmD1SG,
    result::OptimizationResult{QMCFBProblem})

    algorithm.μ₀ = result.result["μ_best"]
    if haskey(result.result, "localization")
        algorithm.localization = result.result["localization"]
        algorithm.stopped = true
    end
    if haskey(result.result, "x̂") && haskey(result.result, "L̂")
        if result.result["L̂"] < algorithm.L̂
            algorithm.L̂ = result.result["L̂"]
            algorithm.x̂ = result.result["x̂"]
        end
    end
    algorithm
end
function run!(
    algorithm::QMCFBPAlgorithmD1SG,
    problem::QMCFBProblem;
    memoranda=Set([]))

    @unpack Q, q, E, b, l, u = problem
    m, n = size(E)
    Ql, Qu = Q*l, Q*u

    view╲ = A -> view(A, [CartesianIndex(i, i) for i in 1:size(A, 1)])
    Q̂ = copy(Q)
    Q╲, Q̂╲ = view╲.([Q, Q̂])
    Q̂╲[:] = 1.0 ./ Q╲

    @unpack localization, verba, max_iter, ε, ϵ, μ₀, heuristic_t, heuristic_each, L̂, stopped = algorithm
    if μ₀ === nothing
        μ₀ = zeros(eltype(Q), m)
    end

    @init_memoria memoranda

    # Qx\tilde is Qx not projected in the box
    get_Qx̃ = μ -> -E'μ-q
    function in_box(Qx̃, Ql=Ql, Qu=Qu, ϵ=0.0)
        𝔅 = [Qx̃ .≤ Ql.+ϵ  zeros(Bool, length(Qx̃))  Qx̃ .≥ Qu.+ϵ]
        𝔅[:, 2] = .~(𝔅[:, 1] .| 𝔅[:, 3])
        𝔅
    end

    function get_an_xᵢ(Qx̃ᵢ, 𝔅ᵢ, Q̂╲ᵢ, lᵢ, uᵢ)
        if 𝔅ᵢ[2]
            Q̂╲ᵢ*Qx̃ᵢ
        elseif 𝔅ᵢ[1] != 𝔅ᵢ[3]
            lᵢ*𝔅ᵢ[1] + uᵢ*𝔅ᵢ[3]
        else
            rand() |> r -> r*lᵢ + (1.0-r)*uᵢ
        end
    end
    function get_an_x(Qx̃, 𝔅, Q̂╲=Q̂╲, l=l, u=u)
        get_an_xᵢ.(Qx̃, [𝔅ᵢ for 𝔅ᵢ in eachrow(𝔅)], Q̂╲, l, u)
    end
    function get_an_x(μ)
        Qx̃ = get_Qx̃(μ)
        𝔅 = in_box(Qx̃)
        get_an_x(Qx̃, 𝔅)
    end
    function get_L(x, μ)
        (0.5*Q╲.*x + q + E'μ)'x - μ'b
    end
    function get_L(μ)
        x = get_an_x(μ)
        get_L(x, μ)
    end
    function get_∂L(x, μ)
        E*x-b
    end
    # @return: a subgradient of L(μ)
    function get_a_∂L(μ)
        x = get_an_x(μ)
        get_∂L(x, μ)
    end
    function solve(μ₀)
        μ, x = copy(μ₀), get_an_x(μ₀)
        L, ∂L = get_L(x, μ), get_∂L(x, μ)

        # best solution up to now
        x_best, μ_best, L_best, ∂L_best, L̂ = copy(x), copy(μ), L, copy(∂L), algorithm.L̂

        wrapper = (func!, localization, x, μ, f_μ, f_xμ, ∂f_μ, ∂f_xμ) -> begin
            if typeof(localization) <: DualSubgradientMethod
                func!(localization, x, zeros(Bool, length(μ)), f_xμ, ∂f_xμ, μ)
            else
                func!(localization, f_μ, ∂f_μ, μ)
            end
        end
        get_x̅ = (typeof(localization) <: DualSubgradientMethod) ?
            () -> localization.x̅ :
            () -> nothing

        if stopped == false
            wrapper(init!,
                localization,
                μ -> get_an_x(μ),
                μ,
                μ -> -get_L(μ),
                (x, μ) -> get_L(x, μ),
                μ -> -get_a_∂L(μ),
                (x, μ) -> get_∂L(x, μ))
        end

        iₕ = 1
        if heuristic_t !== Nothing
            do_heuristic = (x) -> begin
                if iₕ ≥ heuristic_each
                    heuristic = heuristic_t(problem, x, ϵ=ϵ)
                    init!(heuristic)
                    x̂, Δ = run!(heuristic)
                    L̂ = 0.5*x̂⋅(Q╲.*x̂) + q⋅x̂
                    if algorithm.L̂ > L̂
                        algorithm.L̂ = L̂
                        algorithm.x̂ = x̂
                    end
                    iₕ = 1
                else
                    iₕ += 1
                end
                algorithm.L̂
            end
        else
            do_heuristic = (x) -> Inf
        end

        for i in 1:max_iter
            # TODO: develop stopping criteria
            (μ_t, α, sg) =
                wrapper(step!,
                    localization,
                    μ -> get_an_x(μ),
                    μ,
                    μ -> -get_L(μ),
                    (x, μ) -> get_L(x, μ),
                    μ -> -get_a_∂L(μ),
                    (x, μ) -> get_∂L(x, μ))

            @memento μ[:] = μ_t
            @memento x[:] = get_an_x(μ)
            @memento L = get_L(x, μ)
            @memento ∂L[:] = get_∂L(x, μ)
            @memento norm∂L = norm(∂L)
            @memento x̅ = get_x̅()
            @memento L̂ = do_heuristic(x̅!==nothing ? x̅ : x)
            if L > L_best
                @memento L_best=L
                @memento L̂_best = L̂
                @memento ∂L_best[:]=∂L
                @memento norm∂L_best=norm∂L
                @memento x_best[:]=x
                @memento μ_best[:]=μ
                @memento i_best=i
            end
            if L̂-L_best < ε
                break
            end
        end
        x̅ = get_x̅()
        x̂ = algorithm.x̂

        return @get_result x_best μ_best L_best ∂L_best x̅ localization L̂ x̂
    end

    solve(μ₀) |> result ->
        OptimizationResult{QMCFBProblem}(memoria=@get_memoria, result=result)
end

"""
**Example**
```julia
using Optimization
localization = Subgradient.FixedStepSize(0.1)
algorithm = QMCFBPAlgorithmD1SG(localization=localization, verbosity=0, max_iter=1000, ϵ=1e-8, ε=1e-8)
test = get_test(algorithm, m=10, n=20)
𝔓 = test.problem
Q, q, l, u, E, b = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b)
algorithm.μ₀ = zeros(eltype(Q), size(E, 1))
test.solver.options.memoranda = Set(["norm∂L"])
run!(test)

using Optimization; localization = Subgradient.FixedStepSize(0.1); algorithm = QMCFBPAlgorithmD1SG(localization=localization, verbosity=0, max_iter=1000, ϵ=1e-8, ε=1e-8); test = get_test(algorithm, m=10, n=20); 𝔓 = test.problem; Q, q, l, u, E, b = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b); algorithm.μ₀ = zeros(eltype(Q), size(E, 1)); test.solver.options.memoranda = Set(["norm∂L"]);
```
"""
