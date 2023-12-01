module BayesianOptimization

# -> need to manually add before (they have no versions):
# add https://github.com/samuelbelko/SurrogatesBase.jl.git#finite_posterior
# add https://github.com/JuliaBayesianOptimization/SurrogatesAbstractGPs.jl.git
# add https://github.com/JuliaBayesianOptimization/AbstractBayesianOptimization.jl.git
using AbstractBayesianOptimization
using SurrogatesAbstractGPs
using SurrogatesBase

using Printf
using ParameterHandling
# TODO: remove sobol -> keep quasimontecarlo.jl only
# but does QuasiMonteCarlo support next!(..) interface???
using Sobol
using QuasiMonteCarlo
using KernelFunctions
using Random
using SpecialFunctions

# decision support model
export BasicGPSpecification
export BasicGP
include("DecisionSupportModels/BasicGP.jl")

# helper utilities from AbstractBayesianOptimization
export OptimizationHelper, Min, Max, initialize, optimize!, history, solution

# miscellaneous
include("Misc/defaults.jl")
include("Misc/utilities.jl")
include("Misc/MaximizeAcquisitionNLopt.jl")
using .MaximizeAcquisitionNLopt

# policies & related types
export ExpectedImprovementPolicy,
    MaxMeanPolicy,
    MutualInformationPolicy,
    ProbabilityOfImprovementPolicy,
    ThompsonSamplingPolicy,
    UpperConfidenceBoundPolicy,
    NoBetaScaling,
    BrochuBetaScaling
include("Policies/ExpectedImprovementPolicy.jl")
include("Policies/MaxMeanPolicy.jl")
include("Policies/MutualInformationPolicy.jl")
include("Policies/ProbabilityOfImprovementPolicy.jl")
include("Policies/ThompsonSamplingPolicy.jl")
include("Policies/UpperConfidenceBoundPolicy.jl")

end # module
