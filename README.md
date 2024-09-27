# GammaMixtures
Fit mixtures of gamma distributions using the algorithm discussed in  
https://doi.org/10.1007/s11634-019-00361-y 

## Install

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url="https://github.com/grero/NeuraCodingRegistry.jl.git"))
Pkg.add("GammaMixtures")
```

## Usage

```julia
    using GammaMixtures
    
    # generate a model
    model = GammaMixture([1.0, 10.0],[1.0, 1.0], [0.2, 0.8])
    # sample from the model
    x = rand(rng, model, 500)

    # initalize a mode
    model10 = GammaMixture(2, x, MOMInitializer)
    
    # fit a new model
    model1,Δl = fit!(model10, x;niter=10_000)
    
```
