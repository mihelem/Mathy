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
