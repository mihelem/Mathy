# Generate instances with PARGEN+NETGEN+QFCGEN

pargen_path = "./pargen"

# generate the parameter files
function doit(K)
    for n in 1000
        for ρ in 1:3
            for k in 1:K
                for cf in ['a', 'b']
                    for cq in ['a', 'b']
                        for scale in ["ns"]
                            run(`$pargen_path $n $ρ $k $cf $cq $scale`)
                        end
                    end
                end
            end
        end
    end
end

n, ρ, k, cf, cq, scale = 15, 1, 1, "a", "a", "ns"
run(`$pargen_path $n $ρ $k $cf $cq $scale`)

# TEMPORARY CODE TO GENERATE SCALING-TESTING TESTSET
for n in [2^i for i in 10:20]
    for ρ in 1:3
        run(`$pargen_path $n $ρ $k $cf $cq $scale`)
    end
end


doit(1)
# generate with netgen the network file
path = "./CM19/Optimization.jl/src/cpp/bin/scaling/"
for filename in readdir(path)
    if endswith(filename, ".par") == false
        continue
    end
    dmx_file = filename[1:end-3]*"dmx"
    write(
        path*dmx_file,
        read(
            pipeline(
                path*filename,
                `./CM19/Optimization.jl/test/gen/netgen`),
            String))
end
# Generate with qfcgen the quadratic files
for filename in readdir(path)
    if endswith(filename, ".dmx") == false
        continue
    end
    qfc_file = filename[1:end-3]*"qfc"
    run(`./CM19/Optimization.jl/test/gen/qfcgen $path$filename`)
end

function add_singular(problems::Dict{String, QMCFBProblem},
    singulars::Array{Float64, 1})

    result = Dict{String, QMCFBProblem}()
    for (name, problem) in problems
        m, n = size(problem.E)
        ss = convert.(Int64, ceil.(singulars.*n))
        digits = length(string(maximum(ss)))
        for s in ss
            newname = name*"-"*lpad(s, digits, '0')
            result[newname] = deepcopy(problem)
            Q╲ = view(result[newname].Q, [CartesianIndex(i, i) for i in 1:n])
            Q╲[end-s+1:end] .= 0
        end
    end
    result
end

sproblems = add_singular(problems, [0.0, 0.33, 0.66, 1.0])
for (name, problem) in sproblems
    fullname = path*name
    write(NetgenDIMACS, fullname, problem)
end


include("./dmacio.jl")
problems = parse_dir(NetgenDIMACS, path)


mypath = "CM19/Optimization.jl/test/gen/set/"
for filename in readdir(path)
    if length(split(filename, '-')) == 7
        rm(path*filename)
    end
end
