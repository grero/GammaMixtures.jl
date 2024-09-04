module GammaMixtures
using StatsBase
using Random
using Distributions
using Optim
using SpecialFunctions
using LinearAlgebra
using GaussianMixtures
using Logging

export GammaMixture, NaiveInitializer, MOMInitializer

struct GammaMixture{T<:Real} <: StatsBase.StatisticalModel
    α::Vector{T}
    β::Vector{T}
    λ::Vector{T}
    converged::Bool
end

GammaMixture(α::Vector{T}, β::Vector{T}, λ::Vector{T}) where T <: Real = GammaMixture(α, β, λ, false)

abstract type AbstractInitializer end

struct NaiveInitializer <: AbstractInitializer
end

struct MOMInitializer <: AbstractInitializer
end

function GammaMixture(m::Integer, x::AbstractVector{T}, ::Type{NaiveInitializer}) where T <: Real
    n = length(x)
    α = fill(0.0, m)
    β = fill(0.0, m)
    λ = fill(0.0, m)

    λ .= rand(Dirichlet(m,1.0))
    ii = argmin(λ)
    if λ[ii] < 0.05
        Δ = 0.05 - λ[ii]
        λ[ii] += Δ 
        λ[1:m .!= ii]  .-= Δ/(m-1)
    end
    λ ./= sum(λ)
    # partition
    xs = sort(x)
    boundaries = round.(Int64,cumsum(λ,dims=2)[:].*n)
    boundaries = [1;boundaries]
    for (j,(i1,i2)) in enumerate(zip(boundaries[1:end-1], boundaries[2:end]))
        _xs = xs[i1:i2]
        g = fit(Gamma, _xs)
        α[j],β[j] = (g.α, g.θ)
    end
    GammaMixture(α, β, λ)
end

function GammaMixture(m::Integer, x::AbstractVector{T}, ::Type{MOMInitializer}) where T <: Real
    n = length(x)
    xg = x.^(1/3)
    gm = GMM(m, xg)

    # use the memmber ship posteriors
    z, = gmmposterior(gm, reshape(xg,n,1))
    λ = dropdims(mean(z,dims=1),dims=1)

    zs = sum(z,dims=1)
    xp = ((x'*z)./zs)
    sp = (n*z'*(x .- xp).^2)./((n-1)*zs)
    α = diag((xp.^2)./sp)
    β = diag(sp./xp)
    GammaMixture(α, β, λ)
end



StatsBase.dof(model::GammaMixture) = length(model.α) + length(model.β) + length(model.λ)-1

function StatsBase.loglikelihood(model::GammaMixture, x)
    m = length(model.β)
    β = reshape(model.β, 1, m)
    α = reshape(model.α,1,m)
    λ = reshape(model.λ, 1, m)
    sum(log.(sum(λ.*pdf.(Gamma.(α,β), x),dims=2)))
end

function StatsBase.bic(model::GammaMixture, x)
    n = length(x)
    k = dof(model)
    -2*loglikelihood(model,x) +  k*log(n)
end

function StatsBase.bic(d::Distribution, x)
    n = length(x)
    k = length(params(d))
    -2*loglikelihood(d,x) +  k*log(n)
end

function Distributions.rand(rng::AbstractRNG, model::GammaMixture)
    # find the category
    λs = cumsum(model.λ)
    q = rand(rng)
    ii = searchsortedfirst(λs, q)
    # sample from the corresponding Gamma distribution
    rand(rng, Gamma(model.α[ii], model.β[ii]))
end

function Distributions.rand!(x::AbstractArray{T}, model::GammaMixture{T})  where T <: Real
    rand!(x, Random.default_rng(), model)
end

function Distributions.rand!(x, rng::AbstractRNG, model::GammaMixture)
    for i in eachindex(x) 
        x[i] = rand(rng, model)
    end
end

Distributions.rand(model::GammaMixture, d::Integer, dims::Integer...) = rand(Random.default_rng(), model, d, dims...)

function Distributions.rand(rng::AbstractRNG, model::GammaMixture, d::Integer, dims::Integer...)
    x = fill(0.0, d,dims...)
    rand!(x, rng, model)
    x
end
"""
Fit a mixture model of `m` Gamma distributions using the algoritm in 
https://doi.org/10.1007/s11634-019-00361-y
"""
function fit_gamma_mixture(x::AbstractVector{T}, m::Integer;niter=100,α0::Union{Nothing, Vector{T}}=nothing,
                                                            β0::Union{Nothing, Vector{T}}=nothing,
                                                            λ0::Union{Nothing,Vector{T}}=nothing) where T <: Real
    n = length(x)
    # initial values
    α = fill(1.0, 1,m)
    β = fill(1.0, 1,m)
    λ = fill(0.0, 1,m)
    if λ0 === nothing
        λ[1,:] .= rand(Dirichlet(m,1.0))
        λ .= max.(λ, 0.05)
        ii = argmin(λ)
        if λ[ii] < 0.05
            Δ = 0.05 - λ[ii]
            λ[ii] += Δ 
            λ[1:m .!= ii]  .-= Δ/(m-1)
        end
    else
        λ[1,:] .= λ0
    end
    # renormalize
    λ ./=sum(λ)

    if α0 === nothing || β0 === nothing
        # partition
        xs = sort(x)
        boundaries = round.(Int64,cumsum(λ,dims=2)[:].*n)
        boundaries = [1;boundaries]
        for (j,(i1,i2)) in enumerate(zip(boundaries[1:end-1], boundaries[2:end]))
            _xs = xs[i1:i2]
            g = fit(Gamma, _xs)
            α[j],β[j] = (g.α, g.θ)
        end
    else
        α[1,:] .= α0
        β[1,:] .= β0
    end

    g(x::T,α,β) = max.(0.0, pdf.(Gamma.(α,β),x))
    f(x,α,β,λ) = sum(λ.*g.(x,α,β),dims=2)
    dQ(z,x,α,β) = sum(z.*(log.(x) .- log.(β) .- digamma.(α)),dims=1)
    ll(z,x,α,β) = sum(z.*log.(λ.*g.(x, α, β)))

    # indicator variable
    z = λ.*g.(x,α,β)./f(x,α,β,λ)
    l0 = ll(z,x,α,β)
    Δl = 0.0
    converged = false
    for i in 1:niter
        λ .= mean(z,dims=1)
        for j in eachindex(α)
            q = optimize(αt->norm(dQ(z[:,j],x,exp.(αt), β[j])), [log(α[j])])
            α[1,j] = exp(first(Optim.minimizer(q)))
        end
        β .= sum(x.*z,dims=1)./(α.*sum(z,dims=1))
        z .= λ.*g.(x,α,β)./f(x,α,β,λ)
        l1 = ll(z,x,α,β)
        Δl = l1 -l0
        l0 = l1
        if 0.0 < Δl < 1e-5
            converged = true
            break
        end
    end
    GammaMixture(α[:], β[:], λ[:], converged), Δl
end

function StatsBase.fit(::Type{GammaMixture}, x,k::Integer;initializer::Type{<:AbstractInitializer}=MOMInitializer, kwargs...)
    model0 = GammaMixture(k, x, initializer)
    model, Δl = fit!(model0, x;kwargs...)
end

function StatsBase.fit!(model::GammaMixture, x;kwargs...)
    k = length(model.λ)
    α0, β0, λ0 = (model.α, model.β, model.λ)
    model,Δl = fit_gamma_mixture(x, k;α0=α0, β0=β0, λ0=λ0,kwargs...)
end

end # module GammaMixtures
