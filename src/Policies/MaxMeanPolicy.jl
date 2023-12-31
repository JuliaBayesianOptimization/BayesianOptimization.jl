struct MaxMeanPolicy <: AbstractPolicy
    optimizer_options::NamedTuple
end

function MaxMeanPolicy(; optimizer_options = (;))
    merged_options = merge((method = :LD_LBFGS, restarts = 10, maxeval = 2000),
        optimizer_options)
    return MaxMeanPolicy(merged_options)
end

function AbstractBayesianOptimization.next_batch!(ac_policy::MaxMeanPolicy,
    dsm,
    oh)
    # optimize posterior mean
    maximizer = first(maximize_acquisition(dimension(oh), ac_policy.optimizer_options) do x
        only(mean(finite_posterior(dsm, [x])))
    end)
    return [maximizer]
end
