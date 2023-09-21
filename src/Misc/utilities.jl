# adopted from https://github.com/jbrea/BayesianOptimization.jl
normal_pdf(μ, σ2) = 1 / √(2π * σ2) * exp(-μ^2 / (2 * σ2))
normal_cdf(μ, σ2) = 1 / 2 * (1 + erf(μ / √(2σ2)))

# TODO: use QuasiMonteCarlo
struct ColumnIterator{T <: AbstractMatrix, Tit}
    data::T
    baseiterator::Tit
end
ColumnIterator(data::AbstractMatrix) = ColumnIterator(data, 1:size(data, 2))
import Base: iterate, length
@inline function dispatch(it::ColumnIterator, next)
    next === nothing && return nothing
    @view(it.data[:, next[1]]), next[2]
end
iterate(it::ColumnIterator) = dispatch(it, iterate(it.baseiterator))
iterate(it::ColumnIterator, s) = dispatch(it, iterate(it.baseiterator, s))
length(it::ColumnIterator) = length(it.baseiterator)

"""
    ScaledLHSIterator(lowerbounds, upperbounds, N)

Returns an iterator over `N` elements of a latin hyper cube sample between
`lowerbounds` and `upperbounds`. See also `ScaledSobolIterator` for an iterator
that has arguably better uniformity.
"""
function ScaledLHSIterator(lowerbounds, upperbounds, N)
    ColumnIterator(latin_hypercube_sampling(lowerbounds, upperbounds, N))
end

# copied from https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/src/utilities/latin_hypercube_sampling.jl
function latin_hypercube_sampling(mins::AbstractVector, maxs::AbstractVector, n::Integer)
    length(mins) == length(maxs) ||
        throw(DimensionMismatch("mins and maxs should have the same length"))
    all(xy -> xy[1] <= xy[2], zip(mins, maxs)) ||
        throw(ArgumentError("mins[i] should not exceed maxs[i]"))
    dims = length(mins)
    result = zeros(dims, n)
    cubedim = Vector(undef, n)
    @inbounds for i in 1:dims
        imin = mins[i]
        dimstep = (maxs[i] - imin) / n
        for j in 1:n
            cubedim[j] = imin + dimstep * (j - 1 + rand())
        end
        result[i, :] .= Random.shuffle!(cubedim)
    end
    result
end
