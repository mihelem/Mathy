#= Utils to parse QMCFBProblem expressed in standard DIMACS format
   as the instances provided at
       http://groups.di.unipi.it/optimize/Data/MCF.html          =#

using SparseArrays, Optimization, Parameters

struct PargenParams
    n::Int64       # arcs
    ρ::Int64       # graph density ∈ [1, 2, 3]
    k::Int64       # instance number
    cf::Char       # linear to fixed cost, 'a': high, 'b': low (CHECK)
    cq::Char       # quadratic to fixed cost, 'a': high, 'b': low (CHECK)
    scale::String  # scale arc capacity by 0.7 if "s", no action if "ns"
end

mutable struct TestgenParams
    n::Int64       # arcs
    ρ::Int64       # graph density ∈ [1, 2, 3]
    k::Int64       # instance number
    cf::Char       # linear to fixed cost, 'a': high, 'b': low (CHECK)
    cq::Char       # quadratic to fixed cost, 'a': high, 'b': low (CHECK)
    scale::String  # scale arc capacity by 0.7 if "s", no action if "ns"
    singular::Int64

    TestgenParams(par::PargenParams) =
        new(par.n, par.ρ, par.k, par.cf, par.cq, par.scale)
    TestgenParams(par::PargenParams, singular::Int64) = begin
        M = TestgenParams(par)
        M.singular = singular
        M
    end
end

struct NetgenDMX
end

struct NetgenQFC
end

struct NetgenDIMACS
end

function Base.parse(::Type{PargenParams}, name::String)
    token = split(name, '-')
    PargenParams(
        parse(Int64, token[2]),
        parse(Int64, token[3]),
        parse(Int64, token[4]),
        token[5][1],
        token[6][1],
        split(token[7], '.')[1])
end

function Base.parse(::Type{TestgenParams}, name::String)
    token = split(name, '-')
    TestgenParams(
        PargenParams(
            parse(Int64, token[2]),
            parse(Int64, token[3]),
            parse(Int64, token[4]),
            token[5][1],
            token[6][1],
            token[7]),
        parse(Int64, split(token[8], '.')[1]))
end

function Base.string(params::TestgenParams)
    "netgen-"*string(params.n)*"-"*string(params.ρ)*"-"*string(params.k)*"-"*params.cf*"-"*params.cq*
        "-"*params.scale*"-"*lpad(string(params.singular), Int64(ceil(log10(params.n)))+1, '0')
end

# qfc -> Q, 𝔮
# Q : diagonal quadratic cost matrix
# 𝔮 : fixed cost vector
# such that the total cost, given the flux x, is
#  ½x'Qx + q'x + 𝔮.*(x used)

function Base.parse(::Type{NetgenQFC}, filename::String; type=Float64)
    open(filename, "r") do io
        parse(NetgenQFC, io; type=type)
    end
end
function Base.parse(::Type{NetgenQFC}, io::IOStream; type=Float64)
    n = parse(Int64, readline(io))
    parse_line = () -> [parse(type, val) for val in split(readline(io), " ")[1:n]]
    𝔮 = parse_line()
    Q = spdiagm(0 => parse_line())
    Q, 𝔮
end

# dmx -> E, b, l, u, q
# E : node-arc incidence matrix
# b : flux conservation constraint
# q : linear cost
# such that the flux x should satisfies
#  Ex = b  (flux conservation)   &&
#  l .≤ x .≤ u  (capacity constraints)

function Base.parse(::Type{NetgenDMX}, filename::String; type=Float64)
    open(filename, "r") do io
        parse(NetgenDMX, io; type=type)
    end
end
function Base.parse(::Type{NetgenDMX}, io::IOStream; type=Float64)
    function get_problem_size()
        while !eof(io)
            line = readline(io)
            if length(line)>0 && line[1]=='p'
                seekstart(io)
                return [parse(Int64, val) for val in split(line, " ")[3:end]]
            end
        end
    end
    m, n = get_problem_size()
    E, b, l, u, q = spzeros(Int8, m, n), zeros(type, m), zeros(type, n), zeros(type, n), zeros(type, n)

    edge_cnt = 0
    while !eof(io)
        line = readline(io)
        if length(line) > 0
            if line[1] == 'n'
                vars = split(line, " ")[2:end]
                b[parse(Int64, vars[1])] = parse(type, vars[2])
            elseif line[1] == 'a'
                edge_cnt += 1
                vars = split(line, " ")[2:end]
                parse(Int64, vars[1]) |>
                no -> begin
                    if !(1≤no≤m)
                        println(line)
                    end
                    E[no, edge_cnt] = -1
                end
                parse(Int64, vars[2]) |>
                no -> begin
                    if !(1≤no≤m)
                        println(line)
                    end
                    E[no, edge_cnt] = 1
                end
                l[edge_cnt] = parse(type, vars[3])
                u[edge_cnt] = parse(type, vars[4])
                q[edge_cnt] = parse(type, vars[5])
            end
        end
    end
    E, -b, l, u, q
end

# filename: name of generated instance, without file extension [.dmx, .qfc]
function Base.parse(::Type{NetgenDIMACS}, filename::String; type=Float64)
    E, b, l, u, q = parse(NetgenDMX, filename*".dmx"; type=type)
    Q, 𝔮 = parse(NetgenQFC, filename*".qfc"; type=type)
    QMCFBProblem(Q, q, l, u, E, b, false)
end

function Base.write(::Type{NetgenDMX}, filename::String, problem::QMCFBProblem)
    @unpack E, b, l, u, q = problem
    m, n = size(E)

    supply = -sum(b[b .< 0.0])
    sources, sinks = count(b .< 0.0), count(b .> 0.0)
    min_arc_cost, max_arc_cost = minimum(q), maximum(q)
    min_arc_cap, max_arc_cap = minimum(u), maximum(u)
    open(normalize_suffix(filename, ".dmx"), "w") do io
        # Problem overall description
        print(io,
            "c Optimization.jl\n",
            "c  Problem  1 input parameters\n",
            "c  ---------------------------\n",
            "c   Number of nodes:            $m\n",
            "c   Source nodes:               $sources\n",
            "c   Sink nodes:                 $sinks\n",
            "c   Number of arcs:             $n\n",
            "c   Minimum arc cost:           $min_arc_cost\n",
            "c   Maximum arc cost:           $max_arc_cost\n",
            "c   Total supply:               $supply\n",
            "c   Minimum arc capacity:       $min_arc_cap\n",
            "c   Maximum arc capacity:       $max_arc_cap\n",
            "c\n",
            "c  *** Minimum cost flow ***\n",
            "c\n",
            "p min $m $n\n")

        # Nodes
        for source in findall(b .< 0)
            println(io, "n $source $(-b[source])")
        end
        for sink in findall(b .> 0)
            println(io, "n $sink $(-b[sink])")
        end

        # Arcs
        for edge in 1:n
            inout = nzrange(E, edge) |>
                ij -> nonzeros(E)[ij[1]] == -1 ? ij : [ij[2], ij[1]]
            println(io, "a $(rowvals(E)[inout[1]]) $(rowvals(E)[inout[2]]) $(l[edge]) $(u[edge]) $(q[edge])")
        end
    end
end

function Base.write(::Type{NetgenQFC}, filename::String, problem::QMCFBProblem)
    @unpack Q = problem
    n = size(Q, 1)
    Q╲ = view(Q, [CartesianIndex(i, i) for i in 1:n])
    open(normalize_suffix(filename, ".qfc"), "w") do io
        println(io, n)
        for v in Q╲[1:end-1]
            print(io, 0, " ")
        end
        println(io, 0)
        for v in Q╲[1:end-1]
            print(io, v, " ")
        end
        println(io, Q╲[end])
    end
end

function Base.write(::Type{NetgenDIMACS}, filename::String, problem::QMCFBProblem)
    write(NetgenDMX, filename, problem)
    write(NetgenQFC, filename, problem)
end

function normalize_path(path)
    if length(path) > 0 && !endswith(path, "/")
        return path*"/"
    end
    return path
end

function normalize_suffix(name::String, suffix::String)
    if endswith(name, suffix)
        name
    else
        name*suffix
    end
end

function parse_dir(::Type{NetgenDIMACS}, path::String; type=Float64, filter=_->true)
    problems = Dict{String, QMCFBProblem}()
    dmx = Set{String}()
    qfc = Set{String}()
    files = Set{String}()
    for name in readdir(path)
        prename = name[1:end-4]
        if endswith(name, ".dmx")
            if !(prename in dmx) && (prename[1:end-4] in qfc)
                push!(files, prename)
            end
            push!(dmx, prename)
        elseif endswith(name, ".qfc")
            if !(prename in qfc) && (prename in dmx)
                push!(files, prename)
            end
            push!(qfc, prename)
        end
    end
    path = normalize_path(path)
    for filename in files
        if filter(filename)
            problems[filename] = parse(NetgenDIMACS, path*filename; type=type)
        end
    end
    problems
end
