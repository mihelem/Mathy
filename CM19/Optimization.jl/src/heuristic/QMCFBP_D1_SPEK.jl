using DataStructures
"""
Min-Cost Maximum-Flow algorithm
Edmond-Karp + Shortest Path
worst case O(m²n²)

"""
mutable struct SPEKHeuristic <: Heuristic
    E::SparseMatrixCSC  # node-arc incidence matrix
    Eᵀ::SparseMatrixCSC # ^ transpose
    c                   # arc cost
    b                   # requested residual node net inward flux
    l                   # residual arc minimum flow
    u                   # residual arc maximum flow
    x₀                  # starting flow distribution
    ϵ                   # threshold under which constraints are satisfied
    ϵₚ                  # a distance is better if less than ...
    strict              # true ⟹ min-cost flow, o.w. faster heuristic

    x
    SPEKHeuristic(problem::QMCFBProblem, c, x; ϵ=0.0, ϵₚ=0.0, strict=true) =
        new(problem.E,
            SparseMatrixCSC(problem.E'),
            c,
            problem.b-problem.E*x,
            problem.l-x,
            problem.u-x,
            x,
            ϵ,
            ϵₚ,
            strict)
end
function init!(H::SPEKHeuristic)
    H.x = zeros(eltype(H.x₀), length(H.x₀))
    H
end
function run!(H::SPEKHeuristic)
    @unpack E, Eᵀ, c, b, l, u, x₀, x, ϵ, ϵₚ, strict = H
    m, n = size(E)
    b′ = copy(b)                # we'll move b′toward 0
    Ti = eltype(E.rowval)       # index type

    # fan = rowvals(Eᵀ)           # incident arcs
    # io = nonzeros(Eᵀ)           # 1, -1 = in, out
    function get_fanio(node)       # input: node index
        fanio = Tuple{Ti, eltype(E), Ti}[]
        function get_other_node(node, edge)
            for j in nzrange(E, edge)
                if rowvals(E)[j] != node
                    return rowvals(E)[j]
                end
            end
        end
        for j in nzrange(Eᵀ, node)
            node′ = get_other_node(node, rowvals(Eᵀ)[j])
            fan_j = rowvals(Eᵀ)[j]
            io_j = nonzeros(Eᵀ)[j]
            tupla = Tuple{Ti, eltype(E), Ti}((fan_j, io_j, node′))
            push!(fanio, tupla)
        end
        fanio
    end
    function get_max_flux(edge, io)
        if io == 1
            x[edge]-l[edge]
        elseif io == -1
            u[edge]-x[edge]
        end
    end

    parent = Array{Tuple{Ti, eltype(E), Ti}, 1}(undef, size(b′))
    function get_max_flux(sink)
        node = sink
        max_flux = b′[sink]
        while parent[node][1] != 0
            edge, io, node = parent[node]
            max_flux = min(max_flux, get_max_flux(edge, io))
        end
        min(max_flux, -b′[node])
    end
    function flow_flux!(sink, flux)
        b′[sink] -= flux
        node = sink
        while parent[node][1] != 0
            edge, io, node = parent[node]
            x[edge] -= io*flux
        end
        b′[node] += flux
        # println("flux: $flux => b′[$sink] -= $flux,  b′[$node] += $flux")
    end
    function get_dist!(distances, sources, visit_d, in_queue, parent, visit_count)
        dist_sum = 0
        function pop_node()
            node = popfirst!(visit_d)
            in_queue[node] = false
            dist_sum -= distances[node]
            node
        end
        function add_node(node)
            if length(visit_d)>0 && distances[node] ≤ distances[first(visit_d)]
                pushfirst!(visit_d, node)
            else
                push!(visit_d, node)
            end
            dist_sum += distances[node]
            in_queue[node] = true
        end
        function update_node(node, distance)
            dist_sum += distance-distances[node]
            distances[node] = distance
        end
        function greater_last()
            if length(visit_d) == 0
                return
            end
            dist_avg = dist_sum / length(visit_d)
            while first(visit_d) < dist_avg
                push!(visit_d, popfirst!(visit_d))
            end
        end

        fill!(in_queue, false)
        fill!(distances, typemax(eltype(distances)))
        in_queue[sources] .= true
        visit_count[sources] .= 1
        push!(visit_d, sources...)
        distances[sources] .= 0
        parent[sources] .= (source->(0, 0, source)).(sources)
        while length(visit_d)>0
            node = pop_node()
            fanio = get_fanio(node)
            distance = distances[node]
            for (edge, io, node′) in fanio
                if get_max_flux(edge, io) > 0.0
                    distance′ = distance - io*c[edge]
                    if distance′+ϵₚ < distances[node′]
                        parent[node′] = (edge, io, node)
                        if in_queue[node′]
                            update_node(node′, distance′)
                        else
                            if (visit_count[node′] += 1) > m
                                error("Negative Cycle: $parent\n$distances")
                            end
                            distances[node′] = distance′
                            add_node(node′)
                        end
                        greater_last()
                    end
                end
            end
        end
        visit_count .= 0
    end

    unreachable = typemax(eltype(b))
    distances = fill(unreachable, length(b))
    visit_d = Deque{Ti}()
    visit_count = zeros(Int64, length(b))
    in_queue = zeros(Bool, length(b))

    sources = findall(b′ .< -ϵ)
    flown = true
    while flown && length(sources)>0
        flown = false

        get_dist!(distances, sources, visit_d, in_queue, parent, visit_count)

        sinks = findall(b′ .> ϵ)
        sinks = sinks[distances[sinks] .< unreachable]
        if length(sinks) > 0
            if strict
                sink = sinks[argmin(distances[sinks])]
                flow_flux!(sink, get_max_flux(sink))
            else
                (sink->flow_flux!(sink, get_max_flux(sink))).(sinks)
            end
            flown = true
        end

        sources = findall(b′ .< -ϵ)
    end

    (x₀+x, b′)
end
