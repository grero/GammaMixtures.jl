using Test
using GammaMixtures
using StatsBase
using StableRNGs


@testset "Basic" begin
    rng = StableRNG(1234)
    model = GammaMixture([1.0, 10.0],[1.0, 1.0], [0.2, 0.8])
    x = rand(rng, model, 500)

    #test initializers
    t0 = time()
    model10 = GammaMixture(2, x, MOMInitializer)
    Δt = time() - t0
    @show Δt

    t0 = time()
    model1,Δl = fit!(model10, x;niter=10_000)
    Δt = time() - t0
    @test model1.converged
    @show model1 Δl Δt

    t0 = time()
    model20 = GammaMixture(2, x, NaiveInitializer)
    Δt = time() - t0
    @show Δt

    t0 = time()
    model2,Δl = fit!(model20, x;niter=10_000)
    Δt = time() - t0
    @show model2 Δl Δt
end