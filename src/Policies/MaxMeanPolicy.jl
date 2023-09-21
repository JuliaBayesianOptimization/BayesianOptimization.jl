struct MaxMeanPolicy <: AbstractPolicy
    optimizer_options::NamedTuple
end

function MaxMeanPolicy(; optimizer_options = (;))
    merged_options = merge((method = :LD_LBFGS, restarts = 10, maxeval = 2000),
        optimizer_options)
    return MaxMeanPolicy(merged_options)
end

function AbstractBayesianOptimization.next_batch!(ac_policy::MaxMeanPolicy,
    dsm::BasicGP,
    oh::OptimizationHelper)
    # optimize posterior mean
    objective = x -> mean(dsm.surrogate, x)
    maximizer, maximum = maximize_acquisition(objective,
        dimension(oh),
        ac_policy.optimizer_options)
    # TODO: log maximum ?
    return [maximizer]
end
