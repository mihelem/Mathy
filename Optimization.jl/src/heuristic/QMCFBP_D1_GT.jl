"""
Goldberg-Tarjan max-flow
           WIP
O(m²n)

"""
mutable struct GTHeuristic <: Heuristic
    E::SparseMatrixCSC
    Eᵀ::SparseMatrixCSC
    b
    l
    u
    x₀
    ϵ

    x
    GTHeuristic(problem::QMCFBProblem, x; ϵ=0.0) =
        new(problem.E,
            SparseMatrixCSC(problem.E'),
            problem.b-problem.E*x,
            problem.l-x,
            problem.u-x,
            x,
            ϵ)
end
function init!(H::GTHeuristic)
    H.x = zeros(eltype(H.x₀), length(H.x₀))
    H
end
function run!(H::GTHeuristic)
    @unpack E, Eᵀ, b, l, u, x₀, x, ϵ = H
    b′ = copy(b)                # we'll bring b to 0
    Ti = eltype(E.rowval)       # index type
    m, n = size(E)

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

    label = Array{Ti, 1}(undef, size(b′))
    function get_max_flux(source, edge, io)
        min(-b′[source], get_max_flux(edge, io))
    end
    function push_flux!(source, edge, io, sink, flux)
        if io == 1
            x[edge] -= flux
        elseif io == -1
            x[edge] += flux
        end
        b′[source] += flux
        b′[sink] -= flux
    end
    function relabel!(source, sink)
        label[source] = label[sink]+1
    end
    function init_all()
        label[]
    end

    sources = findall(b′ .< -ϵ)
    flown = true
    bfs_q1, bfs_q2 = Queue{Ti}(), Queue{Ti}()
    visited = zeros(Bool, length(b′))
    sinks = Ti[]
    for i in 1:m*n
        if !(flown && length(sources)>0)
            break
        end
        flown = false
        visited[sources] .= true
        for source in sources
            parent[source] = (0, 0, source)
        end

        (x->enqueue!(bfs_q1, x)).(sources)
        while !flown && length(bfs_q1)>0
            while length(bfs_q1)>0
                node = dequeue!(bfs_q1)
                fanio = get_fanio(node)
                for (edge, io, node′) in fanio
                    if (visited[node′] == false) && (get_max_flux(edge, io) > 0.0)
                        visited[node′] = true
                        enqueue!(bfs_q2, node′)
                        parent[node′] = (edge, io, node)
                        if b′[node′] > 0.0
                            flown = true
                            push!(sinks, node′)
                        end
                    end
                end
            end
            empty!(bfs_q1)
            bfs_q1, bfs_q2 = bfs_q2, bfs_q1
        end
        for sink in sinks
            flow_flux!(sink, get_max_flux(sink))
        end
        empty!(sinks), empty!(bfs_q1), empty!(bfs_q2)
        visited .= false
        sources = findall(b′ .< -ϵ)
    end

    (x₀+x, b′)
end
