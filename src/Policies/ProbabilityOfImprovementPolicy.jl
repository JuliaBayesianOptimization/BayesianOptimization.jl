"""
The probability of improvement measures the probability that a point `x` leads
to an improvement upon an incumbent target `τ`. For Gaussian distributions it is
given by

`Φ[(μ(x) - τ)/σ(x)]`,

where `Φ` is the standard normal cumulative distribution function and `μ(x)`, `σ(x)`
are mean and standard deviation of the distribution at point `x`.
"""
function ProbabilityOfImprovement(μ, σ², τ = -Inf)
    σ² == 0 && return float(μ > τ)
    return normal_cdf(μ - τ, sqrt(σ²))
end

struct ProbabilityOfImprovementPolicy <: AbstractPolicy
    optimizer_options::NamedTuple
end

function ProbabilityOfImprovementPolicy(; optimizer_options = (;))
    merged_options = merge((method = :LD_LBFGS, restarts = 10, maxeval = 2000),
        optimizer_options)
    return ProbabilityOfImprovementPolicy(merged_options)
end

function AbstractBayesianOptimization.next_batch!(ac_policy::ProbabilityOfImprovementPolicy,
    dsm::BasicGP,
    oh::OptimizationHelper)
    # τ is set to the observed maximum sofar
    # TODO: add ξ > 0 (with a decreasing schedule to 0) to τ to drive more exploration,
    # see P.11 of A tutorial on bayesian optimizationof expensive const functions,
    # with application to active user modeling and hierarchical reinforcement learning,
    # by E.Brochu, V.M.Cora, N. Freitas
    objective = x -> begin
        # possibly faster implementation when computing both at once
        μ, σ² = mean_and_var_at_point(dsm.surrogate, x)
        return ProbabilityOfImprovement(μ, σ², norm_observed_maximum(oh))
    end

    maximizer, maximum = maximize_acquisition(objective,
        dimension(oh),
        ac_policy.optimizer_options)
    # TODO: log maximum ?
    return [maximizer]
end
