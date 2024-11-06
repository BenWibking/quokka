[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=alert_status&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=bugs&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=ncloc&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9489/badge)](https://www.bestpractices.dev/projects/9489)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/quokka-astro/quokka/badge)](https://scorecard.dev/viewer/?uri=github.com/quokka-astro/quokka)
[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)
[![yt-project](https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet")](https://yt-project.org)

# QUOKKA
*Quadrilateral, Umbra-producing, Orthogonal, Kangaroo-conserving Kode for Astrophysics!*

**The Quokka methods paper is now available: https://arxiv.org/abs/2110.01792**

**For detailed instructions on installing the code, please refer to the [Quokka Documentation](https://quokka-astro.github.io/quokka/index.html). You can start a [Discussion](https://github.com/BenWibking/quokka/discussions) for technical support, or open an [Issue](https://github.com/BenWibking/quokka/issues) for any bug reports.**

Quokka is a two-moment radiation hydrodynamics code that uses the piecewise-parabolic method, with AMR and subcycling in time. Runs on CPUs (MPI+vectorized) or NVIDIA GPUs (MPI+CUDA) with a single-source codebase. Written in C++17. (100% Fortran-free.)

Here is a [a Kelvin-Helmholz instability simulated with Quokka](https://vimeo.com/714653592) on a 512x512 uniform grid:

![Animated GIF of KH Instability](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/1f468be6-6d7b-4d53-a02c-4dd8f3ad5154.gif?ClientID=vimeo-core-prod&Date=1653705774&Signature=9bea89d5c9657180391a9538a10fd4f8f7099025)

This is [a 3D Rayleigh-Taylor instability](https://vimeo.com/746363534) simulated on a $256^3$ grid:

![Image of 3D RT instability](extern/rt3d_visit.png)

Quokka also features advanced Adaptive Quokka Refinement:tm: technology:

![Image of Quokka with Baby in Pouch](extern/quokka2.png)

## Dependencies
* C++ compiler (with C++17 support)
* CMake 3.16+
* Python 3.8+
* MPI library with GPU-aware support (OpenMPI, MPICH, or Cray MPI)
* HDF5 1.10+ (serial version)
* CUDA 11.7+ (optional, for NVIDIA GPUs)
* ROCm 5.2.0+ (optional, for AMD GPUs)
* ADIOS2 2.9+ with GPU-aware support (optional, for writing terabyte-sized or larger outputs)

## Problems?
If you run into problems, please start a [Discussion](https://github.com/BenWibking/quokka/discussions) for technical support. If you discover a bug, please let us know by opening an [Issue](https://github.com/BenWibking/quokka/issues).
