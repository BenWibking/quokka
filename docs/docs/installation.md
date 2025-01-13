# Installation

To run Quokka, download this repository and its submodules to your local machine:

    git clone --recursive https://github.com/quokka-astro/quokka.git

Quokka uses CMake (and optionally, Ninja) as its build system. If you don't have CMake and Ninja installed, the easiest way to install them is to run:

    python3 -m pip install cmake ninja --user

Now that CMake is installed, create a ``build/`` subdirectory and compile Quokka, as shown below.

    cd quokka
    mkdir build; cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
    ninja -j6

Congratuations! You have now built all of the 1D test problems on CPU. You can run the automated test suite:

    ninja test

You should see output that indicates all tests have passed, like this:

    100% tests passed, 0 tests failed out of 20

    Total Test time (real) = 111.74 sec

To run in 2D or 3D, build with the ``-DAMReX_SPACEDIM`` CMake option, for example:

    cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3 -G Ninja
    ninja -j6

to compile Quokka for 3D problems.

**By default, Quokka compiles itself only for CPUs. If you want to run Quokka on GPUs, see the section "Running on GPUs" below.**

Have fun!

## Building with CMake + `make`

If you are unable to install Ninja, you can instead use CMake with the Makefile generator, which should produce identical results but is slower:

    cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
    make -j6
    make test

## Could NOT find Python error

If CMake prints an error saying that Python could not be found, e.g.:

    -- Could NOT find Python (missing: Python_EXECUTABLE Python_INCLUDE_DIRS Python_LIBRARIES Python_NumPy_INCLUDE_DIRS Interpreter Development NumPy Development.Module Development.Embed)

you should be able to fix this by installing NumPy (and matplotlib) by running

    python3 -m pip install numpy matplotlib --user

This should enable CMake to find the NumPy header files that are needed to successfully compile.

Alternatively, you can work around this problem by disabling Python support. Python and NumPy are only used to plot the results of some test problems, so this does not otherwise affect Quokka's functionality. Add the option

    -DQUOKKA_PYTHON=OFF

to the CMake command-line options (or change the `QUOKKA_PYTHON` option to `OFF` in CMakeLists.txt).

## Running on GPUs

By default, Quokka compiles itself to run only on CPUs. Quokka can run on either NVIDIA or AMD GPUs. Consult the sub-sections below for the build instructions for a given GPU vendor.

### NVIDIA GPUs

If you want to run on NVIDIA GPUs, re-build Quokka as shown below. (*CUDA >= 11.7 is required. Quokka is only supported on Volta V100 GPUs or newer models. Your MPI library* **must** *support CUDA-aware MPI.*)

    cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=CUDA -DAMReX_SPACEDIM=3 -G Ninja
    ninja -j6

**All GPUs on a node must be visible from each MPI rank on the node for efficient GPU-aware MPI communication to take place via CUDA IPC.** When using the SLURM job scheduler, this means that `--gpu-bind` should be set to `none`.

The compiled test problems are in the test problem subdirectories in `build/src/`. Example scripts for running Quokka on compute clusters are in the `scripts/` subdirectory.

Note that 1D problems can run very slowly on GPUs due to a lack of sufficient parallelism. To run the test suite in a reasonable amount of time, you may wish to exclude the matter-energy exchange tests, e.g.:

    ctest -E "MatterEnergyExchange*"

which should end with output similar to the following:

    100% tests passed, 0 tests failed out of 18

    Total Test time (real) = 353.77 sec

### AMD GPUs

Compile with `-DAMReX_GPU_BACKEND=HIP`. Requires ROCm 6.3.0 or newer. Your MPI library **must** support GPU-aware MPI for AMD GPUs. Quokka has been tested on MI100 and MI250X GPUs.

### Intel GPUs *(does not compile)*

Due to limitations in the Intel GPU programming model, Quokka currently cannot be compiled for Intel GPUs. (See <https://github.com/quokka-astro/quokka/issues/619> for the technical details.)

## Building a specific test problem

By default, all available test problems will be compiled. If you only want to build a specific problem, you can list all of the available CMake targets:

    cmake --build . --target help

and then build the problem of interest:

    ninja -j6 test_hydro3d_blast
