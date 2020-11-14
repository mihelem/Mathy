# README
`Optimization.jl` is collecting some experiments in mathematical
programming sprout from the project for **Computational Mathematics for Machine Learning and Data Analysis**
held by Antonio Frangioni and Federico Poloni (University of Pisa).
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

For some examples, look at the [jupyter notebook](report/noML09_tutorial.ipynb).

### QMCFBP : Quadratic Min Cost Flow Boxed Problem (...separable convex...)
`→ mincostflow.jl`

`minₓ { ½xᵀQx + qᵀx  with  x s.t.  Ex = b  &  l ≤ x ≤ u }`
with
* `Q` ∈ { diag ≥ 0 }
* `E` : node-arc incidence matrix of directed graph, `rank ≡ nodes - connected components` if unreduced

For a sketch of the theory behind the code, take a look at the [report](report/CM19_20___noML09.pdf).


