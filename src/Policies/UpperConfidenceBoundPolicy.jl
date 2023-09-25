abstract type BetaScaling end
"""
Sets `βt` of `UpperConfidenceBound` as

`βt = √(2 * log(t^(D/2 + 2) * π^2/(3δ)))`,

where `t` is the number of observations, `D` is the dimensionality of the input
data points and `δ` is a small constant (default `δ = 0.1`).

See Brochu E., Cora V. M., de Freitas N. (2010), "A Tutorial on Bayesian
Optimization of Expensive Cost Functions, with Application to Active User
Modeling and Hierarchical Reinforcement Learning", https://arxiv.org/abs/1012.2599v1
page 16.
"""
struct BrochuBetaScaling <: BetaScaling
    δ::Float64
end
"""
Causes no adjustment of `βt` in `UpperConfidenceBound`, `βt` is a fixed parameter,
by default equal to `0.1`.
"""
struct NoBetaScaling <: BetaScaling
    βt::Float64
end

NoBetaScaling() = NoBetaScaling(; βt = 0.1)

"""
For Gaussian distributions the upper confidence bound at `x` is given by

`μ(x) + βt * σ(x)`

where `βt` is a fixed parameter in the case of `NoBetaScaling` or an observation
size dependent parameter in the case of e.g. `BrochuBetaScaling`.
"""
UpperConfidenceBound(μ, σ², βt) = μ + βt * sqrt(σ²)

struct UpperConfidenceBoundPolicy{S <: BetaScaling} <: AbstractPolicy
    scaling::S
    optimizer_options::NamedTuple
end

function UpperConfidenceBoundPolicy(;
    scaling = BrochuBetaScaling(0.1),
    optimizer_options = (;))
    merged_options = merge((method = :LD_LBFGS, restarts = 10, maxeval = 2000),
        optimizer_options)
    return UpperConfidenceBoundPolicy(scaling, merged_options)
end

function AbstractBayesianOptimization.next_batch!(ac_policy::UpperConfidenceBoundPolicy{
        NoBetaScaling,
    },
    dsm, oh)
    objective = x -> begin
        # possibly faster implementation when computing both at once
        μ, σ² = mean_and_var_at_point(dsm, x)
        return UpperConfidenceBound(μ, σ², ac_policy.scaling.βt)
    end
    maximizer, maximum = maximize_acquisition(objective,
        dimension(oh),
        ac_policy.optimizer_options)
    # TODO: log maximum ?
    return [maximizer]
end

function AbstractBayesianOptimization.next_batch!(ac_policy::UpperConfidenceBoundPolicy{
        BrochuBetaScaling,
    }, dsm, oh)
    # number of observations
    nobs = evaluation_counter(oh)
    nobs == 0 && (nobs = 1)
    βt = sqrt(2 * log(nobs^(dimension(oh) / 2 + 2) * π^2 / (3 * ac_policy.scaling.δ)))
    maximizer = first(maximize_acquisition(dimension(oh), ac_policy.optimizer_options) do x
        # possibly faster implementation when computing both at once
        μ, σ² = mean_and_var_at_point(dsm, x)
        return UpperConfidenceBound(μ, σ², βt)
    end)
    return [maximizer]
end
