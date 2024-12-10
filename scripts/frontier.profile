module purge

# module versions of CPE, cce, and rocm *MUST* match according to this table:
# https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#compatible-compiler-rocm-toolchain-versions

module load Core/24.07
module load cpe/24.07

module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a

module load rocm/6.1.3
module load cray-mpich
module load cce/18.0.0  # must be loaded after rocm

# hdf5
module load cray-hdf5

# cmake
module load cmake/3.27.9

# optional
module load emacs

# optional
module load cray-python/3.11.5

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# optimize GPU compilation for MI250X
export AMREX_AMD_ARCH=gfx90a

# compiler environment hints
export CC=$(which cc)
export CXX=$(which CC)
export FC=$(which ftn)

# these flags are REQUIRED
export CFLAGS="-I${ROCM_PATH}/include"
export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed"
export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
