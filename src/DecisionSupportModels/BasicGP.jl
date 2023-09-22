mutable struct BasicGPState
    isdone::Bool
end

"""
A basic Gaussian process surrogate with hyperparameter optimization, each hyperparmeter
needs to have a specified interval of valid values.

We assume that the domain is `[0,1]^dim` and we are maximizing.
"""
struct BasicGP{D <: Real, R <: Real, J} <: AbstractDecisionSupportModel
    oh::OptimizationHelper
    # number of initial samples
    n_init::Int
    # use kernel_creator for constructing surrogate at initialization
    kernel_creator::Function
    # for hyperparameter optimization
    θ_initial::NamedTuple
    # in terms of number of objective evaluations
    optimize_θ_every::Int
    initializer::J
    state::BasicGPState
    surrogate::GPSurrogate{Vector{D}, R}
    # todo verbosity levels
    verbose::Bool
end

function BasicGP(oh::OptimizationHelper,
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
    return BasicGP(oh,
        n_init,
        kernel_creator,
        θ_initial,
        optimize_θ_every,
        initializer,
        BasicGPState(false),
        GPSurrogate(Vector{Vector{domain_eltype(oh)}}(),
            Vector{range_type(oh)}();
            kernel_creator = kernel_creator,
            hyperparameters = ParameterHandling.value(θ_initial)),
        verbose)
end

function AbstractBayesianOptimization.initialize!(dsm::BasicGP, oh::OptimizationHelper)
    # check if there is budget before evaluating the objective
    if evaluation_budget(oh) < dsm.n_init
        dsm.state.isdone = true
        dsm.verbose || @info "Cannot initialize model, no evaluation budget left."
        return nothing
    end

    init_xs = [next!(dsm.initializer) for _ in 1:(dsm.n_init)]
    init_ys = evaluate_objective!(oh, init_xs)
    # update of observed maximum & maximizer is done automatically by OptimizationHelper
    add_points!(dsm.surrogate, init_xs, init_ys)
    update_hyperparameters!(dsm.surrogate, BoundedHyperparameters(dsm.θ_initial))
    return nothing
end

function AbstractBayesianOptimization.update!(dsm::BasicGP, oh::OptimizationHelper, xs, ys)
    dsm.verbose || @info @sprintf "#eval %3i: update! run" evaluation_counter(oh)
    @assert length(xs) == length(ys)
    add_points!(dsm.surrogate, xs, ys)
    if evaluation_counter(oh) % dsm.optimize_θ_every == 0
        update_hyperparameters!(dsm.surrogate, BoundedHyperparameters(dsm.θ_initial))
        dsm.verbose ||
            @info @sprintf "#eval %3i: hyperparmeter optimization run" evaluation_counter(oh)
    end
    # update of observed maximum & maximizer is done automatically by OptimizationHelper
    return nothing
end

AbstractBayesianOptimization.isdone(dsm::BasicGP) = dsm.state.isdone
