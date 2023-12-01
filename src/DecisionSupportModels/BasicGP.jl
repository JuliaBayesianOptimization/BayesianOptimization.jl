mutable struct BasicGPState
    isdone::Bool
end

"""
TODO: add docs for "decision support model specification", fix interface definition
 in AbstractBayesianOptimization
"""
struct BasicGPSpecification{ H <: NamedTuple, J} <: AbstractDecisionSupportModel
    # number of initial samples
    n_init::Int
    # use kernel_creator for constructing surrogate at initialization
    kernel_creator::Function
    # for hyperparameter optimization
    θ_initial::H
    # in terms of number of objective evaluations
    optimize_θ_every::Int
    initializer::J
    # todo verbosity levels
    verbose::Bool
end

function BasicGPSpecification(
        oh,
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
    return BasicGPSpecification(
        n_init,
        kernel_creator,
        θ_initial,
        optimize_θ_every,
        initializer,
        verbose)
end

"""
A basic Gaussian process surrogate with hyperparameter optimization, each hyperparmeter
needs to have a specified interval of valid values.

We assume that the domain is `[0,1]^dim` and we are maximizing.
"""
struct BasicGP{S <: GPSurrogate} <: AbstractDecisionSupportModel
    oh::OptimizationHelper
    spec::BasicGPSpecification
    state::BasicGPState
    surrogate::S
end

# TODO: change interface in AbstractBayesianOptimization (and remove ! from initialize)
function initialize(dsm_spec::BasicGPSpecification, oh)
    # check if there is budget before evaluating the objective
    if evaluation_budget(oh) < dsm_spec.n_init
        throw(ErrorException("Cannot initialize model, no evaluation budget left."))
    end

    init_xs = [next!(dsm_spec.initializer) for _ in 1:(dsm_spec.n_init)]
    init_ys = evaluate_objective!(oh, init_xs)
    # update of observed maximum & maximizer is done automatically by OptimizationHelper

    surrogate = GPSurrogate(init_xs,
        init_ys;
        kernel_creator = dsm_spec.kernel_creator,
        hyperparameters = ParameterHandling.value(dsm_spec.θ_initial))
    update_hyperparameters!(surrogate, BoundedHyperparameters([dsm_spec.θ_initial]))

    return BasicGP(oh, dsm_spec, BasicGPState(false), surrogate)
end

function AbstractBayesianOptimization.update!(dsm::BasicGP, oh, xs, ys)
    dsm.spec.verbose || @info @sprintf "#eval %3i: update! run" evaluation_counter(oh)
    @assert length(xs) == length(ys)
    add_points!(dsm.surrogate, xs, ys)
    if evaluation_counter(oh) % dsm.spec.optimize_θ_every == 0
        update_hyperparameters!(dsm.surrogate, BoundedHyperparameters([dsm.spec.θ_initial]))
        dsm.spec.verbose ||
            @info @sprintf "#eval %3i: hyperparmeter optimization run" evaluation_counter(oh)
    end
    # update of observed maximum & maximizer is done automatically by OptimizationHelper
    return nothing
end

AbstractBayesianOptimization.isdone(dsm::BasicGP) = dsm.state.isdone

# forwarding pattern for SurrogatesBase interface used in policies
SurrogatesBase.finite_posterior(dsm::BasicGP, xs) = finite_posterior(dsm.surrogate, xs)
