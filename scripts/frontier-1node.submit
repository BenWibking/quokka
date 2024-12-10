#!/usr/bin/env zsh

#SBATCH -A ast146
#SBATCH -J amrex_quokka
#SBATCH -o 1node_%x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -N 1

# load cray modules (must be done manually before submitting the job)
# module load cpe/22.08 craype-accel-amd-gfx90a rocm/5.2.0 cray-mpich cce/14.0.2 cray-hdf5

# note (5-16-22, OLCFHELP-6888)
# this environment setting is currently needed on Crusher to work-around a
# known issue with Libfabric
export FI_MR_CACHE_MAX_COUNT=0  # libfabric disable caching

# always run with GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# use correct NIC-to-GPU binding
# (https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#olcfdev-1292-crusher-default-nic-binding-is-not-ideal)
export MPICH_OFI_NIC_POLICY=NUMA

srun build/src/HydroBlast3D/test_hydro3d_blast tests/benchmark_unigrid_512.in

