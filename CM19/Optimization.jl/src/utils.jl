"""
```julia
module Utils
```

Submodule easying:
* conditional initialization (`@some`)
* verbosity (`verba`)
* tracking of data in iterative algorithms (`@init_memoria`, `@memento`, `@get_memoria`)
"""
module Utils

export @some, verba, @init_memoria, @memento, @get_memoria, @get_result, mid

"""
```julia
@some
```

Conditional initialization

**Example**
```julia
# assign b to a only if b is not nothing
@some a = b
``
**TODO **
* Here is just for a single assignment, need to be extended to handle general expressions

"""
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

"""
```julia
verba(verbosity, level, message)
```

Verbosity utility: print message only if level ≤ verbosity. Usually in each `OptimizationAlgorithm` it is bound to a chosen verbosity level.

**Note**
* Maybe will be replaced by a variable specific tool, since verbosity levels are somehow arbitrary.
"""
function verba(verbosity, level, message)
    if level ≤ verbosity
        println(message)
    end
end

"""
```
@init_memoria(expr)
```julia

**Iteration recorder**.
Against macro hygiene, it is creating a dictionary called `memoria`, where
the requested intermediate results of computations are saved.

**Example**
```julia
# By hand
@memoria ["∇f", "norm∇f"]
# usually the summentioned set is specified in memoranda:
run!(algorithm, problem, memoranda=...)
# which is set to solver.options.memoranda when is called
run!(solver, problem)
```

"""
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

"""
```julia
@memento
```

**Example**
```julia
# Pushing 10. to memoria["x"] if "x" is a key in the Dict memoria
@memento x = 10.
```

"""
macro memento(expr)
    if typeof(expr) === Expr
        if expr.head === :(=)
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
        elseif expr.head === :function

        end
    end
end

"""
```julia
macro get_memoria()
```

Return the dictionary memoria

**Example**
```julia
OptimizationResult{QMCFBProblem}(memoria=@get_memoria, result=result)
```

"""
macro get_memoria()
    :(memoria) |> esc
end

"""
```julia
@get_result
```

Create a dictionary of the passed variables with pairs `(name => value)`

**Example**
```julia
result = @get_result p Π∇L normΠ∇L L
OptimizationResult{QMCFBProblem}(memoria=@get_memoria, result=result)
```

"""
macro get_result(args...)
    result_expr = :(Dict{String, Any}())
    for var in args
        push!(result_expr.args, :($(string(var)) => $var) |> esc)
    end
    result_expr
end

"""
```julia
mid(a, b, c)
```

Returns the one between `a`, `b`, `c` with value in the middle

"""
function mid(a, b, c)
    a, b = a ≤ b ? (a, b) : (b, a)
    min(max(a, c), b)
end

end     # end module Utils
