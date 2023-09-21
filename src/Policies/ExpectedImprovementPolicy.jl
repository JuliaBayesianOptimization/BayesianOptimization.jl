"""
The expected improvement measures the expected improvement `x - τ` of a point `x`
upon an incumbent target `τ`. For Gaussian distributions it is given by

`(μ(x) - τ) * ϕ[(μ(x) - τ)/σ(x)] + σ(x) * Φ[(μ(x) - τ)/σ(x)]`,

where `ϕ` is the standard normal distribution function and `Φ` is the standard
normal cumulative function, and `μ(x)`, `σ(x)` are mean and standard deviation
of the distribution at point `x`.
"""
function ExpectedImprovement(μ, σ², τ = -Inf)
    σ² == 0 && return μ > τ ? μ - τ : 0.0
    return (μ - τ) * normal_cdf(μ - τ, σ²) + sqrt(σ²) * normal_pdf(μ - τ, σ²)
end

struct ExpectedImprovementPolicy <: AbstractPolicy
    optimizer_options::NamedTuple
end

function ExpectedImprovementPolicy(; optimizer_options = (;))
    merged_options = merge((method = :LD_LBFGS, restarts = 10, maxeval = 2000),
        optimizer_options)
    return ExpectedImprovementPolicy(merged_options)
end

function AbstractBayesianOptimization.next_batch!(ac_policy::ExpectedImprovementPolicy,
    dsm::BasicGP,
    oh::OptimizationHelper)
    objective = x -> begin
        # possibly faster implementation when computing both at once
        μ, σ² = mean_and_var(dsm.surrogate, x)
        return ExpectedImprovement(μ, σ², norm_observed_maximum(oh))
    end
    maximizer, maximum = maximize_acquisition(objective,
        dimension(oh),
        ac_policy.optimizer_options)
    # TODO: log maximum ?
    return [maximizer]
end
