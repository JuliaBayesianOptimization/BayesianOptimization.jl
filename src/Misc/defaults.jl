function compute_θ_initial(dimension)
    # TODO: compute initial hyperparams from xs, ys, see https://infallible-thompson-49de36.netlify.app/
    #       for now, return each time the same constant inital θ
    l = ones(dimension)
    s_var = 1.0
    n_var = 0.1
    return (;
        lengthscales = bounded(l, 0.004, 4.0),
        signal_var = bounded(s_var, 0.01, 15.0),
        noise_var = bounded(n_var, 0.0001, 0.2))
end

# noise_var is passed into AbstractGPs directly, not via a kernel
function kernel_creator(hyperparameters)
    return hyperparameters.signal_var *
           with_lengthscale(KernelFunctions.Matern52Kernel(), hyperparameters.lengthscales)
end
