** Testing and Benchmarking **

- - Warning - -
Paths, files etc... are to customized

- - Files Description - -

disposable/
 exp.jl : compare two subgradient methods
 runnertemplate.jl :
 tableA3.jl :
 testsetrun.jl :
 testsetrunexp.jl : most of the experiments in the report

gen/
 * :  source code of Pargen+Netgen+Qfcgen

utils/
 benchmark.jl : benchmark all the details of the QMCFBProblem solvers
 dmacio.jl : read problem files (generated with PARGEN+NETGEN+QFCGEN)
 gen.jl : drive PARGEN+NETGEN+QFCGEN generators
 grb.jl : solve QMCFBProblems with Gurobi via JuMP
 nljumpy.jl : solve with Ipopt
