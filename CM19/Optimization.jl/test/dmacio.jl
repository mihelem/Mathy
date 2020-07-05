#= Utils to parse QMCFBProblem expressed in standard DIMACS format
   as the instances provided at
       http://groups.di.unipi.it/optimize/Data/MCF.html          =#

using SparseArrays

# qfc -> Q, 𝔮
# Q : diagonal quadratic cost matrix
# 𝔮 : fixed cost vector
# such that the total cost, given the flux x, is
#  ½x'Qx + q'x + 𝔮.*(x used)

function parse_qfc(filename::String, type=Float64)
    open(filename, "r") do io
        parse_qfc(io, type)
    end
end
function parse_qfc(io::IOStream, type=Float64)
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

function parse_dmx(filename::String, type=Float64)
    open(filename, "r") do io
        parse_dmx(io, type)
    end
end
function parse_dmx(io::IOStream, type=Float64)
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
                E[parse(Int64, vars[1]), edge_cnt] = -1
                E[parse(Int64, vars[2]), edge_cnt] = 1
                l[edge_cnt] = parse(type, vars[3])
                u[edge_cnt] = parse(type, vars[4])
                q[edge_cnt] = parse(type, vars[5])
            end
        end
    end
    E, -b, l, u, q
end
