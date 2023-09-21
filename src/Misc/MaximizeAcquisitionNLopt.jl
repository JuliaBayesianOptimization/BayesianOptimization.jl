# adopted from https://github.com/jbrea/BayesianOptimization.jl
module MaximizeAcquisitionNLopt

import NLopt
using ForwardDiff
using Random

export maximize_acquisition

include("utilities.jl")

function wrap_gradient(f)
    return (x, g) -> begin
        res = DiffResults.DiffResult(0.0, g)
        ForwardDiff.gradient!(res, f, x)
        res.value
    end
end
wrap_dummygradient(f) = (x, g) -> f(x)

# TODO: stochastic acquisition like Gutmann and Corander (2016), p. 20
function maximize_acquisition(objective, dimension, options)
    lowerbounds = zeros(dimension)
    upperbounds = ones(dimension)
    opt = NLopt.Opt(options.method, dimension)

    for (option, value) in pairs(options)
        (option == :method || option == :restarts) && continue
        setproperty!(opt, option, value)
    end

    NLopt.lower_bounds!(opt, lowerbounds)
    NLopt.upper_bounds!(opt, upperbounds)

    if string(options.method)[2] == 'D'
        f = wrap_gradient(objective)
        NLopt.max_objective!(opt, f)
    else
        NLopt.max_objective!(opt, wrap_dummygradient(objective))
    end
    return acquire_max(opt, dimension, options.restarts)
end

function acquire_max(opt, dimension, restarts)
    lowerbounds = zeros(dimension)
    upperbounds = ones(dimension)
    current_maximum = -Inf
    current_maximizer = lowerbounds
    seq = ScaledLHSIterator(lowerbounds, upperbounds, restarts)
    for x0 in seq
        f, x, ret = NLopt.optimize(opt, x0)
        ret == NLopt.FORCED_STOP &&
            @warn("NLopt returned FORCED_STOP while optimizing the acquisition function.")
        if f > current_maximum
            current_maximizer = x
            current_maximum = f
        end
    end
    return current_maximizer, current_maximum
end

end
