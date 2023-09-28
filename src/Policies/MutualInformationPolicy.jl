
"""
The mutual information measures the amount of information gained by querying at
`x`. The parameter `γ̂` gives a lower bound for the information on `f` from the queries
{x}. For a Gaussian this is `γ̂ = ∑σ²(x)` and the mutual information at `x` is

`μ(x) + √(α)*(√(σ²(x)+γ̂) - √(γ̂))`,

where `μ(x)`, `σ(x)` are mean and standard deviation
of the distribution at point `x`.

See Contal E., Perchet V., Vayatis N. (2014), "Gaussian Process Optimization
with Mutual Information" http://proceedings.mlr.press/v32/contal14.pdf
"""
MutualInformation(μ, σ², sqrt_α, γ̂) = μ + sqrt_α * (sqrt(σ² + γ̂) - sqrt(γ̂))

mutable struct MutualInformationPolicy <: AbstractPolicy
    optimizer_options::NamedTuple
    sqrt_α::Float64
    γ̂::Float64
end

function MutualInformationPolicy(; optimizer_options = (;), α = 1.0, γ̂ = 0.0)
    merged_options = merge((method = :LD_LBFGS, restarts = 10, maxeval = 2000),
        optimizer_options)
    return MutualInformationPolicy(merged_options, sqrt(α), γ̂)
end

#  requires no_history falg in OptimizationHelper set to true
function AbstractBayesianOptimization.next_batch!(ac_policy::MutualInformationPolicy,
    dsm,
    oh)
    if iszero(evaluation_counter(oh))
        ac_policy.γ̂ = 0.0
    else
        ac_policy.γ̂ += var_at_point(dsm, norm_last_x(oh))
    end
    maximizer = first(maximize_acquisition(dimension(oh), ac_policy.optimizer_options) do x
        # possibly faster implementation when computing both at once
        μ, σ² = mean_and_var_at_point(dsm, x)
        return MutualInformation(μ, σ², ac_policy.sqrt_α, ac_policy.γ̂)
    end)
    return [maximizer]
end
