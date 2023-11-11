mutable struct BasicGPState
    isdone::Bool
end

"""
TODO: add docs for "Pre decision support model", fix interface in AbstractBayesianOptimization
"""
struct PreBasicGP{ H <: NamedTuple, J} <: AbstractDecisionSupportModel
    oh::OptimizationHelper
    # number of initial samples
    n_init::Int
    # use kernel_creator for constructing surrogate at initialization
    kernel_creator::Function
    # for hyperparameter optimization
    θ_initial::H
    # in terms of number of objective evaluations
    optimize_θ_every::Int
    initializer::J
    state::BasicGPState
    # todo verbosity levels
    verbose::Bool
end

function PreBasicGP(oh,
        n_init;
        optimize_θ_every = 10,
        kernel_creator = kernel_creator,
        θ_initial = compute_θ_initial(dimension(oh)),
        initializer = nothing,
        verbose = true)
    if isnothing(initializer)
        # TODO: how many samples do we need to skip for Sobol for better uniformity?
        initializer = SobolSeq(dimension(oh))
        # skip first 2^10 -1 samples
        skip(initializer, 10)
    end
    return PreBasicGP(oh,
        n_init,
        kernel_creator,
        θ_initial,
        optimize_θ_every,
        initializer,
        BasicGPState(false),
        verbose)
end

"""
A basic Gaussian process surrogate with hyperparameter optimization, each hyperparmeter
needs to have a specified interval of valid values.

We assume that the domain is `[0,1]^dim` and we are maximizing.
"""
struct BasicGP{S <: GPSurrogate, H <: NamedTuple, J} <: AbstractDecisionSupportModel
    oh::OptimizationHelper
    # number of initial samples
    n_init::Int
    # use kernel_creator for constructing surrogate at initialization
    kernel_creator::Function
    # for hyperparameter optimization
    θ_initial::H
    # in terms of number of objective evaluations
    optimize_θ_every::Int
    initializer::J
    state::BasicGPState
    surrogate::S
    # todo verbosity levels
    verbose::Bool
end

# TODO: change interface in AbstractBayesianOptimization (and remove ! from initialize)
function initialize(pre_dsm::PreBasicGP, oh)
    # check if there is budget before evaluating the objective
    if evaluation_budget(oh) < pre_dsm.n_init
        throw(ErrorException("Cannot initialize model, no evaluation budget left."))
    end

    init_xs = [next!(pre_dsm.initializer) for _ in 1:(pre_dsm.n_init)]
    init_ys = evaluate_objective!(oh, init_xs)
    # update of observed maximum & maximizer is done automatically by OptimizationHelper

    surrogate = GPSurrogate(init_xs,
        init_ys;
        kernel_creator = pre_dsm.kernel_creator,
        hyperparameters = ParameterHandling.value(pre_dsm.θ_initial))
    update_hyperparameters!(surrogate, BoundedHyperparameters([pre_dsm.θ_initial]))

    return BasicGP(pre_dsm.oh,
        pre_dsm.n_init,
        pre_dsm.kernel_creator,
        pre_dsm.θ_initial,
        pre_dsm.optimize_θ_every,
        pre_dsm.initializer,
        pre_dsm.state,
        surrogate,
        pre_dsm.verbose)
end

function AbstractBayesianOptimization.update!(dsm::BasicGP, oh, xs, ys)
    dsm.verbose || @info @sprintf "#eval %3i: update! run" evaluation_counter(oh)
    @assert length(xs) == length(ys)
    add_points!(dsm.surrogate, xs, ys)
    if evaluation_counter(oh) % dsm.optimize_θ_every == 0
        update_hyperparameters!(dsm.surrogate, BoundedHyperparameters([dsm.θ_initial]))
        dsm.verbose ||
            @info @sprintf "#eval %3i: hyperparmeter optimization run" evaluation_counter(oh)
    end
    # update of observed maximum & maximizer is done automatically by OptimizationHelper
    return nothing
end

AbstractBayesianOptimization.isdone(dsm::BasicGP) = dsm.state.isdone

# forwarding pattern for SurrogatesBase interface used in policies
SurrogatesBase.finite_posterior(dsm::BasicGP, xs) = finite_posterior(dsm.surrogate, xs)
