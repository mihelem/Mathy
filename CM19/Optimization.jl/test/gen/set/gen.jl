# Generate instances with PARGEN+NETGEN+QFCGEN

# generate the parameter files
function doit(k)
    for n in 1000
        for ρ in 1:3
            for k in 1:5
                for cf in ['a', 'b']
                    for cq in ['a', 'b']
                        for scale in ["ns"]
                            run(`./Optimization.jl/test/gen/pargen $n $ρ $k $cf $cq $scale`)
                        end
                    end
                end
            end
        end
    end
end

doit(1)
# generate with netgen the network file
path = "./Optimization.jl/test/gen/set/"
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
                `./Optimization.jl/test/gen/netgen`),
            String))
end
# Generate with qfcgen the quadratic files
for filename in readdir(path)
    if endswith(filename, ".dmx") == false
        continue
    end
    qfc_file = filename[1:end-3]*"qfc"
    run(`./Optimization.jl/test/gen/qfcgen $path$filename`)
end

include("../../dmacio.jl")
problems = parse_dir(NetgenDIMACS, "Optimization.jl/test/gen/set/")

problem = parse(NetgenDIMACS, "Optimization.jl/test/gen/set/netgen-1000-1-1-b-b-ns")
parames = pargen_name_to_params("netgen-1000-1-1-a-a-ns.dmx")
