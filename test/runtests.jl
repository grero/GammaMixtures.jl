using Test
using GammaMixtures
using StatsBase
using StableRNGs


@testset "Basic" begin
    rng = StableRNG(1234)
    model = GammaMixture([1.0, 10.0],[1.0, 1.0], [0.2, 0.8])
    x = rand(rng, model, 500)

    ll0 = loglikelihood(model, x)
    @test ll0 ≈ -1355.7955874052238

    #test initializers
    t0 = time()
    model10 = GammaMixture(2, x, MOMInitializer)
    Δt = time() - t0
    ll1 = loglikelihood(model10, x)

    t0 = time()
    model1,Δl = fit!(model10, x;niter=10_000)
    Δt = time() - t0
    @test model1.converged

    t0 = time()
    model20 = GammaMixture(2, x, NaiveInitializer)
    Δt = time() - t0

    t0 = time()
    model2,Δl = fit!(model20, x;niter=10_000)
    @test model2.converged
    Δt = time() - t0
end
