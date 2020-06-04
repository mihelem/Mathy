# TODO: any exact "descent" (not really, it's a saddle point) method
# --------------------- Primal Dual algorithm PD1 ------------------------- #
mutable struct QMCFBPAlgorithmPD1 <: OptimizationAlgorithm{QMCFBProblem}
    localization::DescentMethod
    verba       # verbosity utility
    max_iter    # max number of iterations
    ϵ₀          # error within which a point is on a boundary
    ε           # precision to which ∇L is considered null
    p₀          # starting point

    memorabilia # set of the name of variables that can be recorded during execution
    QMCFBPAlgorithmPD1(;
        localization=nothing,
        verbosity=nothing,
        my_verba=nothing,
        max_iter=nothing,
        ϵ₀=nothing,
        ε=nothing,
        p₀=nothing) = begin

        algorithm = new()
        algorithm.memorabilia = Set(["objective", "Π∇L", "∇L", "p", "normΠ∇L", "normΠ∇L_μ"])

        set!(algorithm,
            localization=localization,
            verbosity=verbosity,
            my_verba=my_verba,
            max_iter=max_iter,
            ϵ₀=ϵ₀,
            ε=ε,
            p₀=p₀)
    end
end
# about memorabilia
# names of the variables that can be set to be recorded during execution;
# by now it is a set; in the future it could become a dictionary, since
# to each variable in the mathematical domain we can have many different
# names in the program
function set!(algorithm::QMCFBPAlgorithmPD1;
    localization=nothing,
    verbosity=nothing,
    my_verba=nothing,
    max_iter=nothing,
    ϵ₀=nothing,
    ε=nothing,
    p₀=nothing)

    @some algorithm.localization=localization
    if verbosity !== nothing
        algorithm.verba = ((level, message) -> verba(verbosity, level, message))
    end
    @some algorithm.verba=my_verba
    @some algorithm.max_iter=max_iter
    @some algorithm.ϵ₀=ϵ₀
    @some algorithm.ε=ε
    algorithm.p₀=p₀
end
function set!(algorithm::QMCFBPAlgorithmPD1,
    result::OptimizationResult{QMCFBProblem})

    algorithm.p₀ = result["p"]  # Try also with μ′
    algorithm
end
function run!(algorithm::QMCFBPAlgorithmPD1, 𝔓::QMCFBProblem; memoranda=Set([]))
    @unpack Q, q, l, u, E, b = 𝔓
    @unpack descent, verba, max_iter, ϵ₀, ε, p₀ = algorithm
    @init_memoria memoranda

    m, n = size(E)
    p = p₀ === nothing ? [l + u .* rand(n); rand(m)] : p₀
    @views get_x, get_μ = p->p[1:n], p->p[n+1:n+m]
    x, μ = get_x(p), get_μ(p)

    a ≈ b = abs(a-b) ≤ ϵ₀
    a ⪎ b = a+ϵ₀ ≥ b
    a ⪍ b = a ≤ b+ϵ₀

    Π! = p -> (x = get_x(p); x[:] = min.(max.(x, l), u); p)
    Π∇! = (p, ∇L) -> begin
            x, ∇ₓL = get_x(p), get_x(∇L)
            𝔲, 𝔩 = (x .≥ u), (x .≤ l)
            ∇ₓL[𝔲] = (a -> max(0., a)).(∇ₓL[𝔲])
            ∇ₓL[𝔩] = (a -> min(0., a)).(∇ₓL[𝔩])
            ∇L
        end

    # using ∇L = (∇_x, -∇_μ) to have descent direction for everyone
    get_∇L = p -> (x=get_x(p); μ=get_μ(p); [Q*x+q+E'μ; -E*x+b])
    get_Π∇L = p -> Π∇!(p, get_∇L(p))

    init!(localization, nothing, get_Π∇L, p)
    @memento Π∇L = get_Π∇L(p)
    for i=1:max_iter
        @memento normΠ∇L = norm(Π∇L, Inf)
        verba(1, "||Π∇L|| = $(normΠ∇L)")
        if normΠ∇L < ε
            verba(0, "\n$i iterazioni\n")
            break
        end
        @memento p[:] = Π!(step!(descent, nothing, get_Π∇L, p))
        @memento Π∇L = get_Π∇L(p)
        @memento normΠ∇L_μ = norm(get_μ(Π∇L), Inf)
        @memento objective = 0.5*x'Q*x+q'x
    end

    normΠ∇L = norm(Π∇L, 2); verba(0, "||Π∇L|| = $(normΠ∇L)")
    verba(0, "||Ex-b|| = $(norm(get_μ(Π∇L), Inf))")
    L = 0.5*x'Q*x+q'x; verba(0, "L = $L")

    # Need a deep copy?
    result = @get_result p Π∇L normΠ∇L L
    OptimizationResult{QMCFBProblem}(memoria=@get_memoria, result=result)
end
