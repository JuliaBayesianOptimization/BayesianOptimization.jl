struct ThompsonSamplingPolicy{J} <: AbstractPolicy
    n_samples::Int
    # sampling_algorithm::SamplingAlgorithm
    sobol_generator::J
end

function ThompsonSamplingPolicy(oh::OptimizationHelper;
    n_samples = min(100 * dimension(oh), 5000),
    sampling_algorithm = SobolSample())
    # return ThompsonSamplingPolicy(n_samples, sampling_algorithm)

    sobol_gen = SobolSeq(dimension(oh))
    # skip first 2^10 -1 samples
    skip(sobol_gen, 10)
    return ThompsonSamplingPolicy(n_samples, sobol_gen)
end

function AbstractBayesianOptimization.next_batch!(ac_policy::ThompsonSamplingPolicy,
    dsm::BasicGP,
    oh::OptimizationHelper)
    # QuasiMonteCarlo
    # sample n_samples from [0,1]^dimension
    # xs_matrix = QuasiMonteCarlo.sample(ac_policy.n_samples,
    #     dimension(oh),
    #     ac_policy.sampling_algorithm,
    #     domain_eltype(oh))

    # xs_vector = [xs_matrix[:, i] for i in 1:size(xs_matrix, 2)]
    xs_vector = [next!(ac_policy.sobol_generator) for _ in 1:(ac_policy.n_samples)]
    ys = rand(dsm.surrogate, xs_vector)
    # TODO: log in oh
    return [xs_vector[argmax(ys)]]
end
