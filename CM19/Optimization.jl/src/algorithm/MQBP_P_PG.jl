# Dummy struct, since for projected methods the step! should need a
# signature different from the other DescentMethod s
mutable struct QuadraticBoxPCGDescent <: DescentMethod end
mutable struct MQBPAlgorithmPG1 <: OptimizationAlgorithm{MQBProblem}
    localization::DescentMethod      #
    verba                       # verbosity utility
    max_iter                    #
    ε                           # required: norm(∇f, ?) < ε
    ϵ₀                          # abs error to which inequalities are satisfied
    x₀                          # starting point

    memorabilia
    MQBPAlgorithmPG1(;
        localization=nothing,
        verbosity=nothing,
        my_verba=nothing,
        max_iter=nothing,
        ε=nothing,
        ϵ₀=nothing,
        x₀=nothing) = begin

        algorithm = new()
        algorithm.memorabilia = Set(["normΠ∇f", "Π∇f", "x", "f", "d"])
        set!(algorithm,
            localization=localization,
            verbosity=verbosity,
            my_verba=my_verba,
            max_iter=max_iter,
            ε=ε,
            ϵ₀=ϵ₀,
            x₀=x₀)
    end
end
function set!(algorithm::MQBPAlgorithmPG1;
    localization=nothing,
    verbosity=nothing,
    my_verba=nothing,
    max_iter=nothing,
    ε=nothing,
    ϵ₀=nothing,
    x₀=nothing)

    @some algorithm.localization = localization
    if verbosity !== nothing
        algorithm.verba = ((level, message) -> verba(verbosity, level, message))
    end
    @some algorithm.verba = my_verba
    @some algorithm.max_iter = max_iter
    @some algorithm.ε = ε
    @some algorithm.ϵ₀ = ϵ₀
    algorithm.x₀ = x₀

    algorithm
end
function set!(algorithm::MQBPAlgorithmPG1,
    result::OptimizationResult{MQBProblem})

    algorithm.x₀ = result.result["x"]
    if haskey(result.result, "localization")
        algorithm.localization = result.result["localization"]
    end
    algorithm
end
function run!(algorithm::MQBPAlgorithmPG1, 𝔓::MQBProblem; memoranda=Set([]))
    @unpack Q, q, l, u = 𝔓
    @unpack localization, max_iter, verba, ε, ϵ₀, x₀ = algorithm
    @init_memoria memoranda

    x = (x₀ === nothing) ? 0.5*(l+u) : x₀
    a::AbstractFloat ⪝ b::AbstractFloat = a ≤ b + ϵ₀        # \simless
    a::AbstractFloat ≃ b::AbstractFloat = abs(a-b) ≤ ϵ₀     # \simeq
    to0 = (x::AbstractFloat -> (x ≃ 0.0) ? 0.0 : x)

    # Box Projectors
    # Coordinate Space
    Π = (x, l, u) -> ((u .⪝ x) .| (x .⪝ l))
    Π! = (x, l, u) -> (x[:] = min.(max.(x, l), u))
    # Tangent Space
    ΠᶜT = (d, x, l, u) -> begin                             # \Pi \^c T
        𝔲, dec = (u .⪝ x), (d .> 0.0)
        𝔩, inc = (x .⪝ l), (d .< 0.0)
        ((𝔲 .& dec) .| (𝔩 .& inc))
    end
    ΠT = (d, x, l, u) -> (.~ΠᶜT(d, x, l, u))
    ΠT! = (d, x, l, u) -> begin
        d[ΠᶜT(d, x, l, u)] .= 0.0
        d
    end

    #
    get_Πx = (x, l, u) -> min.(max.(x, l), u)
    get_f = (Πx, Q, q) -> 0.5*Πx'Q*Πx + q'Πx
    get_Πf = (x, Q, q, l, u) -> get_f(get_Πx(x, l, u), Q, q)
    get_∇f = (Πx, Q, q) -> Q*Πx+q

    get_Π∇f = (x, Q, q, l, u) -> begin
        Πx = get_Πx(x, l, u)
        ∇f = get_∇f(Πx, Q, q)
        -ΠT!(-∇f, x, l, u)
    end

    # ᾱ is an α corresponding to the line crossing a side of the box
    # assuming a valid  l .≤ x .≤ u

    # ----------- Simpler Approach ----------- #
    function get_Δx′(x, d, l, u)
        Δx = -x
        (d .> 0.0) |> (inc -> Δx[inc] += u[inc])
        (d .< 0.0) |> (dec -> Δx[dec] += l[dec])
        Δx
    end
    function line_search′(pq::PriorityQueue, x, Δx, d, Q, q, Qx)
        𝔐 = .~zeros(Bool, length(x))                             # 𝔐 :: \frakM : moving coordinates
        Δ1 = d.*q + d.*Qx
        dQ = d.*Q
        Δα = [sum(Δ1), sum(dQ*d)]

        x′ = copy(x)
        count = 0
        while length(pq) > 0
            if Δα[1] ≥ 0.0
                break
            end

            α = - Δα[1] / max(Δα[2], +0.0)
            i, ᾱ = peek(pq); dequeue!(pq)
            if α ≤ ᾱ
                x′[𝔐] += α*d[𝔐]
                break
            else
                Δα[2] += dQ[i, i]*d[i] - 2.0*sum(dQ[𝔐,i]*d[i])
                𝔐[i] = false
                dQ[𝔐, i]*Δx[i] |>
                    Δ -> (Δ1[𝔐] += Δ; Δα[1] += sum(Δ) - Δ1[i])
                x′[i] += Δx[i]
            end
            count += 1
        end

        return (x′, 𝔐)
    end
    function local_search′(x, Q, q, l, u, max_iter, stop_on_cross=true)
        x′ = copy(x)

        g′, g = Q*x+q, zeros(eltype(x), length(x)) .+ Inf
        d = -g′
        for i in 1:max_iter
            𝔐 = (ΠT(d, x′, l, u) .& .~((d / norm(d, Inf)) .≃ 0.0))

            norm_Πg′ = norm(g[𝔐], Inf)
            @memento local_normΠ∇f = norm_Πg′
            verba(2, "local_search : norm_Πg′ = $(norm_Πg′)")
            if norm_Πg′ < ε || count(𝔐) == 0
                break
            end

            ᾱ = minimum(get_Δx′(x′[𝔐], d[𝔐], l[𝔐], u[𝔐]) ./ d[𝔐])
            Δα = (d[𝔐]'q[𝔐] + d[𝔐]'Q[𝔐, :]*x′, d'Q*d)
            α = - Δα[1] / Δα[2]
            if Δα[1] ≥ 0.0
                verba(1, "local_search′ : something went wrong, I feel stiff")
            elseif α ≤ ᾱ
                x′[𝔐] += α*d[𝔐]
            else
                x′[𝔐] += ᾱ*d[𝔐]
                if stop_on_cross
                    break
                end
            end

            g′[:], g[:] = Q*x′+q, g′
            β = max(0.0, g′⋅(g′-g) / g⋅g)
            d[:] = -g′ + β*d
        end

        return x′
    end
    function step′(x, d, Q, q, l, u)
        𝔐 = (ΠT(d, x, l, u) .& .~((d / norm(d, Inf)) .≃ 0.0))      # 𝔐 :: \frakM : moving coordinates
        x′, d′, l′, u′, Q′, q′ = x[𝔐], d[𝔐], l[𝔐], u[𝔐], Q[𝔐, 𝔐], q[𝔐]

        Δx′ = get_Δx′(x′, d′, l′, u′)
        ᾱs = Δx′ ./ d′
        pq = PriorityQueue(zip([1:length(𝔐)+1;], [ᾱs; Inf]))

        x′, 𝔐′ = line_search′(pq, x′, Δx′, d′, Q′, q′, Q[𝔐, :]*x)
        x[𝔐] = x′
        if any(𝔐′)
            𝔐′ = begin
                temp = copy(𝔐)
                temp[𝔐][.~𝔐′] .= false
                temp
            end
            x[𝔐′] = local_search′(x[𝔐′], Q[𝔐′, 𝔐′], q[𝔐′] + Q[𝔐′, .~𝔐′]*x[.~𝔐′], l[𝔐′], u[𝔐′], 100, false)
        end
        return x
    end

    function solve(localization, x, Q, q, l, u)
        if typeof(localization) !== QuadraticBoxPCGDescent
            init!(localization, x -> get_Πf(x, l, u), x -> get_Π∇f(x, Q, q, l, u), x)
        end
        x[:] = get_Πx(x, l, u)
        g = get_∇f(x, Q, q)
        @memento Π∇f = -ΠT!(-g, x, l, u)
        @memento normΠ∇f = norm(Π∇f, Inf)
        verba(1, "||Π∇f|| : $normΠ∇f")
        @memento d = -g
        @memento Πd = ΠT
        for i in 1:max_iter
            if normΠ∇f < ε
                verba(0, "\nIterations: $i\n")
                break
            end

            if typeof(localization) !== QuadraticBoxPCGDescent
                @memento x[:] = get_Πx(step!(localization, x -> get_Πf(x, l, u), x -> get_Π∇f(x, Q, q, l, u), x), l, u)
                @memento Π∇f[:] = get_Π∇f(x, Q, q, l, u)
            else
                @memento x[:] = get_Πx(step′(x, d, Q, q, l, u), l, u)
                g′ = get_∇f(x, Q, q)
                @memento Π∇f[:] = -ΠT!(-g′, x, l, u)
                @memento β = g′⋅(g′-g) / g⋅g
                β = max(0.0 , isnan(β) ? 0.0 : β)
                @memento d[:] = -g′ + β*d
                g[:] = g′
            end

            verba(2, "x : $x")
            verba(2, "Π∇f : $Π∇f")
            @memento normΠ∇f = norm(Π∇f, Inf)
            verba(1, "||Π∇f|| : $normΠ∇f")
        end

        @memento f = get_f(x, Q, q)
        verba(0, "f = $f")
        result = @get_result x Π∇f normΠ∇f f localization
        OptimizationResult{MQBProblem}(memoria=@get_memoria, result=result)
    end

    solve(localization, x, Q, q, l, u)
    # x = local_search′(x, Q, q, l, u, max_iter, false)
    # result = @get_result x
    # OptimizationResult{MQBProblem}(memoria=@get_memoria, result=result)
end
