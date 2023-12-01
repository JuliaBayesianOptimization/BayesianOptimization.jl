using Pkg
Pkg.activate("examples/")

# add it by running `dev ../BayesianOptimization.jl/`
# -> need to manually add before (they have no versions): Note the branch names!
# add https://github.com/samuelbelko/SurrogatesBase.jl.git#finite_posterior
# add https://github.com/JuliaBayesianOptimization/SurrogatesAbstractGPs.jl.git
# add https://github.com/JuliaBayesianOptimization/AbstractBayesianOptimization.jl.git
using BayesianOptimization

# -- for plotting --
using Plots
using LinearAlgebra
# plotlyjs()
gr()
# ----

# lb = left bottom point in domain, ub = top right point in domain
lb, ub = [-2.0, -1.0], [2.0, 2.0]

rosenbrock(x::Vector; kwargs...) = rosenbrock(x[1], x[2]; kwargs...)
# non-convex function
rosenbrock(x1, x2) = 100 * (x2 - x1^2)^2 + (1 - x1)^2

minima(::typeof(rosenbrock)) = [[1, 1]], 0
r_mins, r_fmin = minima(rosenbrock)

function p(r_mins)
    plt = contour(-2:0.1:2,
        -1:0.1:2,
        (x, y) -> -rosenbrock([x, y]),
        levels = 500,
        fill = true)
    plt = scatter!((x -> x[1]).(history(oh)[1]),
        (x -> x[2]).(history(oh)[1]),
        label = "eval. hist")
    plt = scatter!((x -> x[1]).(r_mins),
        (y -> y[2]).(r_mins),
        label = "true minima",
        markersize = 10,
        shape = :diamond)
    plt = scatter!([solution(oh)[1][1]],
        [solution(oh)[1][2]],
        label = "observed min.",
        shape = :rect)
    plt
end

# g, sense::Sense, lb, ub, max_evaluations
oh = OptimizationHelper(rosenbrock, Min, lb, ub, 200)
# oh, n_init; optimize_θ_every = 10, ...
dsm_spec = BasicGPSpecification(oh, 10, optimize_θ_every = 10)
policy = ExpectedImprovementPolicy(optimizer_options = (restarts = 50,))
#policy = MaxMeanPolicy()
# policy = MutualInformationPolicy()
# policy = ProbabilityOfImprovementPolicy()
#policy = ThompsonSamplingPolicy(oh)
# policy = UpperConfidenceBoundPolicy()

# run initial sampling, create initial trust regions and local models
dsm = initialize(dsm_spec, oh)

# savefig(p(), "plot_before_optimization.png")
#display(p())

# Optimize
optimize!(dsm, policy, oh)

# savefig(p(), "plot_after_optimization.png")
display(p(r_mins))

observed_dist = minimum((m -> norm(solution(oh)[1] .- m)).(r_mins))
observed_regret = abs(solution(oh)[2] - r_fmin)
