# README
`Optimization.jl` is a WIP project collecting some experiments in mathematical
programming sprout from the project for **Computational Mathematics for Machine Learning and Data Analysis**
held by Antonio Frangioni and Federico Poloni at University of Pisa.
It is mainly a testing bed for the theory conveyed at the lectures.
## How to use the package
Download (clone) the repository in a `path` which you can then add to the `LOAD_PATH`
variable in `julia` with the command
```julia
push!(LOAD_PATH, /path/to/the/repo)
```
To have it permanently added you can add such line of code in `~/.julia/config/startup.jl` - create the
file if it still does not exist.

The next step is to add the package to your REPL with
```julia
using Optimization
```

Then you are ready to go.

## Where?

**Structure**:

Here are some examples, which may give you an idea of where you can go with this package.

### QMCFBP : Quadratic Min Cost Flow Boxed Problem (...separable convex...)
__→ mincostflow.jl__

`minₓ { ½xᵀQx + qᵀx  with  x s.t.  Ex = b  &  l ≤ x ≤ u }`
with
* `Q` ∈ { diag ≥ 0 }
* `E` : node-arc incidence matrix of directed graph, `rank ≡ nodes - connected components` if unreduced

**Deflected Subgradient**

__→ QMCFBP_*_SG.jl__

Here we are instantiating and running an instance of a QMCFB problem randomly generated
where the algorithm of choice has been one of the deflected subgradient methods, `RMSProp`.
```julia
using Optimization
subgradient =
    Subgradient.RMSProp(
        α=1e-7,
        γ=1-1e-11)
algorithm =
    QMCFBPAlgorithmD1SG(
        subgradient=subgradient,
        verbosity=0,
        max_iter=1000000,
        ϵ=1e-8,
        ε=1e-8)
test =
    get_test(
        algorithm,
        m=1000,     # number of nodes
        n=2000,     # number of arcs
        singular=0, # dim(ker(Q))
        active=0)
test.solver.options.memoranda =
    Set([
        "norm∂L",
        "L",
        "norm∂L′",
        "L′",
        "i′"]);
run!(test)
p =
    plot(
        test.result,
        "norm∂L′",
        "i′",
        mapping=x->log.(x))
```

### MQBP : Min Quadratic Boxed Problem (convex)
__→ minquadratic.jl__

`minₓ ½ x'Qx + q'x` where  `l .≤ x .≤ u` and `Q ≽ 0`

```julia

```
