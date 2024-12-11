#!/bin/bash

## NOTE: CCE and ROCm versions must match according to this table:
##  https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#compatible-compiler-rocm-toolchain-versions
## These are matching versions: cce 17.0.0 <--> cpe 23.12 <--> rocm 5.7.1

source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a

module load rocm/5.7.1 # matches cce/17 clang version
module load cray-mpich
module load cce/17.0.0

# hdf5
module load cray-hdf5

# python
module load cray-python/3.11.5

# cmake -- missing from cce/17.0.0 module environment!
pip install cmake --user

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a

# compiler environment hints
export CC=$(which cc)
export CXX=$(which CC)
export FC=$(which ftn)

# these flags are REQUIRED
export CFLAGS="-I${ROCM_PATH}/include -Wno-#warnings -Wno-ignored-attributes"
export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed -Wno-#warnings -Wno-ignored-attributes"
export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
