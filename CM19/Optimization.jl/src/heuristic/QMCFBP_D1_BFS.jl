"""
Example
```julia
using Optimization
subgradient = Subgradient.FilteredPolyakStepSize(gen_γ=i -> 10.0/i, β=0.4)
algorithm = QMCFBPAlgorithmD1SG(
          localization=subgradient,
          verbosity=1,
          max_iter=10, # not useful
          ε=1e-6,
          ϵ=1e-12);
test = get_test(algorithm, m=15, n=30, singular=8);
test.solver.options.memoranda = Set(["norm∂L′", "L′","i′"])
run!(test)
x = test.result.result["x′"]
𝔓 = test.problem; Q, q, l, u, E, b = (𝔓.Q, 𝔓.q, 𝔓.l, 𝔓.u, 𝔓.E, 𝔓.b);
heu = BFSHeuristic(𝔓, x)
init!(heu)
x′, b′ = run!(heu)
A = Optimization.MinCostFlow.incidence_to_adjacency(E)
using GraphRecipes, Plots
graphplot(A, curvature_scalar=0.01, names=1:5, markersize=0.2, arrow=arrow(:closed, :head, 1, 1))

```

"""
mutable struct BFSHeuristic <: Heuristic
    E::SparseMatrixCSC
    Eᵀ::SparseMatrixCSC
    b
    l
    u
    x₀
    ϵ

    x
    BFSHeuristic(𝔓::QMCFBProblem, x; ϵ=0.0) =
        new(𝔓.E,
            sparse(Array(𝔓.E')),
            𝔓.b-𝔓.E*x,
            𝔓.l-x,
            𝔓.u-x,
            x,
            ϵ)
end
function init!(H::BFSHeuristic)
    H.x = zeros(eltype(H.x₀), length(H.x₀))
    H
end
function run!(H::BFSHeuristic)
    @unpack E, Eᵀ, b, l, u, x₀, x, ϵ = H
    b′ = copy(b)                # we'll bring b to 0
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
    for i::Ti in 1:length(b)
        while b′[i] <  -ϵ
            parent[i] = (0, 0, i)
            sinks = Ti[]
            visited = zeros(Bool, length(b′))
            bfs_queue = Queue{Ti}()         # BFS queue
            enqueue!(bfs_queue, i)          # start BFS from given node
            visited[i] = true
            while length(bfs_queue)>0
                node = dequeue!(bfs_queue)
                fanio = get_fanio(node)
                for (edge, io, node′) in fanio
                    if (visited[node′] == false) && (get_max_flux(edge, io) > ϵ)
                        visited[node′] = true
                        enqueue!(bfs_queue, node′)
                        parent[node′] = (edge, io, node)
                        if b′[node′] > ϵ
                            push!(sinks, node′)     # Or just augment the path here...
                        end
                    end
                end
            end

            for sink in sinks
                flow_flux!(sink, get_max_flux(sink))
                if b′[i] ≥ -ϵ
                    break
                end
            end
        end
    end

    (x₀+x, b′)
end
