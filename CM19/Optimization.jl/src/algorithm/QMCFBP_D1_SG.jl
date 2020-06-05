# ------------------------- Subgradient Methods ------------------------- #
mutable struct QMCFBPAlgorithmD1SG <: OptimizationAlgorithm{QMCFBProblem}
    localization::SubgradientMethod
    verba               # verbosity utility
    max_iter            # max number of iterations
    ε
    ϵ
    μ₀                  # starting point

    memorabilia # set of the name of variables that can be recorded during execution
    QMCFBPAlgorithmD1SG(;
        localization=nothing,
        verbosity=nothing,
        my_verba=nothing,
        max_iter=nothing,
        ε=nothing,
        ϵ=nothing,
        μ₀=nothing) = begin

        algorithm = new()
        algorithm.memorabilia =
            Set([
                "x̅",        # → X(μ) see HarmonicErgodicPrimalStep ∈ Subgradient
                "x",        # primal coordinate
                "μ",        # constraint dual vector
                "L",        # Lagrangian
                "∂L",       # ∂L ∈ subgradient L(μ)
                "norm∂L",   # norm(∂L, Inf)
                "x′",       # x for new best L
                "μ′",       # μ for new best L
                "L′",       # new best L
                "∂L′",      # ∂L for each new best L
                "norm∂L′",  # norm(∂L, Inf) for each new best L
                "i′"])      # iteration counter for each new best L

        set!(
            algorithm,
            localization=localization,
            verbosity=verbosity,
            my_verba=my_verba,
            max_iter=max_iter,
            ε=ε,
            ϵ=ϵ,
            μ₀=μ₀)
    end
end
function set!(algorithm::QMCFBPAlgorithmD1SG;
    localization=nothing,
    verbosity=nothing,
    my_verba=nothing,
    max_iter=nothing,
    ε=nothing,
    ϵ=nothing,
    μ₀=nothing)

    @some algorithm.localization=localization
    if verbosity !== nothing
        algorithm.verba = ((level, message) -> verba(verbosity, level, message))
    end
    @some algorithm.verba=my_verba
    @some algorithm.max_iter=max_iter
    @some algorithm.ε=ε
    @some algorithm.ϵ=ϵ
    algorithm.μ₀=μ₀

    algorithm
end
function set!(algorithm::QMCFBPAlgorithmD1SG,
    result::OptimizationResult{QMCFBProblem})

    algorithm.μ₀ = result.result["μ′"]  # Try also with μ′
    algorithm
end
function run!(
    algorithm::QMCFBPAlgorithmD1SG,
    𝔓::QMCFBProblem;
    memoranda=Set([]))

    @unpack Q, q, E, b, l, u = 𝔓
    m, n = size(E)
    Ql, Qu = Q*l, Q*u

    view╲ = A -> view(A, [CartesianIndex(i, i) for i in 1:size(A, 1)])
    Q̂ = copy(Q)
    Q╲, Q̂╲ = view╲.([Q, Q̂])
    Q̂╲[:] = 1.0 ./ Q╲

    @unpack localization, verba, max_iter, ε, ϵ, μ₀ = algorithm
    if μ₀ === nothing
        μ₀ = zeros(eltype(Q), m)
    end

    @init_memoria memoranda

    # Qx̃ is Qx not projected in the box
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
    # @return: a subgradient of L(μ)
    function get_∂L(x, μ)
        E*x-b
    end
    function get_a_∂L(μ)
        x = get_an_x(μ)
        get_∂L(x, μ)
    end
    function solve(μ₀)
        μ, x = copy(μ₀), get_an_x(μ₀)
        L, ∂L = get_L(x, μ), get_∂L(x, μ)

        # best solution up to now
        x′, μ′, L′, ∂L′ = copy(x), copy(μ), -Inf, copy(∂L)

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

        wrapper(init!,
            localization,
            μ -> get_an_x(μ),
            μ,
            μ -> -get_L(μ),
            (x, μ) -> get_L(x, μ),
            μ -> -get_a_∂L(μ),
            (x, μ) -> get_∂L(x, μ))
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

            #step!(localization, μ->-get_L(μ), μ->-get_a_∂L(μ), μ)
            @memento μ[:] = μ_t
            @memento x[:] = get_an_x(μ)
            @memento L = get_L(x, μ)
            @memento ∂L[:] = get_∂L(x, μ)
            @memento norm∂L = norm(∂L)
            @memento x̅ = get_x̅()
            if L > L′
                @memento L′=L
                @memento ∂L′[:]=∂L
                @memento norm∂L′=norm∂L
                @memento x′[:]=x
                @memento μ′[:]=μ
                @memento i′=i
            end
        end
        x̅ = get_x̅()

        return @get_result x′ μ′ L′ ∂L′ x̅
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
