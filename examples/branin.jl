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
lb, ub = [-15.0, -15.0], [15.0, 15.0]

# copied from BaysianOptimization.jl
# shift x1, x2 by Δ, b/c inital sampling is randomly almost hitting an optimum
Δ = 2.5
branin(x::Vector; kwargs...) = branin(x[1], x[2]; kwargs...)
function branin(x1,
    x2;
    a = 1,
    b = 5.1 / (4π^2),
    c = 5 / π,
    r = 6,
    s = 10,
    t = 1 / (8π),
    noiselevel = 0)
    x1 += Δ
    x2 += Δ
    a * (x2 - b * x1^2 + c * x1 - r)^2 + s * (1 - t) * cos(x1) + s + noiselevel * randn()
end
function minima(::typeof(branin))
    [[-π - Δ, 12.275 - Δ], [π - Δ, 2.275 - Δ], [9.42478 - Δ, 2.475 - Δ]], 0.397887
end
b_mins, b_fmin = minima(branin)

function p(mins)
    plt = contour(-15:0.1:15,
        -15:0.1:15,
        (x, y) -> -branin([x, y]),
        levels = 80,
        fill = true)
    plt = scatter!((x -> x[1]).(history(oh)[1]),
        (x -> x[2]).(history(oh)[1]),
        label = "eval. hist")
    plt = scatter!((x -> x[1]).(mins),
        (y -> y[2]).(mins),
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
oh = OptimizationHelper(branin, Min, lb, ub, 200)
# oh, n_init; optimize_θ_every = 10, ...
dsm_spec = BasicGPSpecification(oh, 10; optimize_θ_every = 10)
policy = ExpectedImprovementPolicy(optimizer_options = (restarts = 50,))
# policy = MaxMeanPolicy()

# MutualInformationPolicy does not work that well, is it correct??
# policy = MutualInformationPolicy(optimizer_options = (restarts = 50,))

# PI does not work that well, is it correct??
# policy = ProbabilityOfImprovementPolicy()

#policy = ThompsonSamplingPolicy(oh)
#policy = UpperConfidenceBoundPolicy()

# run initial sampling, create initial trust regions and local models
dsm = initialize(dsm_spec, oh)

# savefig(p(), "plot_before_optimization.png")
# display(p())

# Optimize
optimize!(dsm, policy, oh)

# profile in VSCode Optimize
# @profview optimize!(dsm, policy, oh)

# savefig(p(), "plot_after_optimization.png")
display(p(b_mins))

observed_dist = minimum((m -> norm(solution(oh)[1] .- m)).(b_mins))
observed_regret = abs(solution(oh)[2] - b_fmin)
