module Utils

export @some, verba, @init_memoria, @memento, @get_memoria, @get_result, mid

# TODO: here is just for a single assignment, 
# need to be extended to handle general expressions
macro some(arg)
    if typeof(arg) === Expr
        if arg.head === :(=)
            quote
                if $(arg.args[2]) !== nothing
                    $(arg.args[1]) = $(arg.args[2])
                end
            end |> esc
        end
    end
end

# verbosity utility
function verba(verbosity, level, message)
    if level ≤ verbosity
        println(message)
    end
end
# iteration recorder
# against macro hygiene, it is creating a dictionary called memoranda
macro init_memoria(expr)
    quote
        memoria = Dict{String, AbstractArray}()
        for meme in $expr
            memoria[meme] = []
        end
    end |> esc
end
macro nomino_memoria(nomen, expr)

end
macro memento(expr)
    if (typeof(expr) === Expr) && (expr.head === :(=))
        l_symbol = expr.args[1]
        while (typeof(l_symbol) === Expr) && (l_symbol.head === :ref)
            l_symbol = l_symbol.args[1]
        end
        quote
            $(expr)
            let meme = string($(Meta.quot(l_symbol)))
                if haskey(memoria, meme)
                    push!(memoria[meme], deepcopy($(l_symbol)))
                end
            end
        end |> esc
    end
end
macro get_memoria()
    :(memoria) |> esc
end
macro get_result(args...)
    result_expr = :(Dict{String, Any}())
    for var in args
        push!(result_expr.args, :($(string(var)) => $var) |> esc)
    end
    result_expr
end

function mid(a, b, c)
    a, b = a ≤ b ? (a, b) : (b, a)
    min(max(a, c), b)
end

end     # end module Utils
