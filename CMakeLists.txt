cmake_minimum_required(VERSION 3.16)
project(QuokkaCode VERSION 1.0
  DESCRIPTION "Radiation hydrodynamics with structured AMR"
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")
#set(CMAKE_C_COMPILER_LAUNCHER "ccache")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/extern/amrex/Tools/CMake)
include(CTest)

set(AMReX_SPACEDIM 1 CACHE STRING "")
set(AMReX_MPI ON CACHE BOOL "" FORCE)
set(AMReX_HDF5 OFF CACHE BOOL "" FORCE) # do not use
set(AMReX_LINEAR_SOLVERS ON CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS_INCFLO OFF CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS_EM OFF CACHE BOOL "" FORCE)
set(AMReX_CONDUIT OFF CACHE BOOL "") # optional
set(AMReX_ASCENT OFF CACHE BOOL "") # optional
set(AMReX_AMRLEVEL OFF CACHE BOOL "" FORCE)
set(AMReX_PROBINIT OFF CACHE BOOL "" FORCE)
set(AMReX_TINY_PROFILE ON CACHE BOOL "" FORCE)

## a Python installation can be specified with Python_ROOT_DIR
##   see: https://cmake.org/cmake/help/latest/module/FindPython.html#hints
## NumPy and Matplotlib can be installed with `python3 -m pip install numpy matplotlib --user`
option(QUOKKA_PYTHON "Compile with Python support (on/off)" ON)
option(DISABLE_FMAD "Disable fused multiply-add instructions on GPU (on/off)" ON)
option(ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENABLE_HWASAN "Enable HWAddressSanitizer" OFF)
option(ENABLE_TESTS_FPE "Enable floating-point exceptions when running tests" ON)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(QUOKKA_OPENPMD "Enable OpenPMD output (on/off)" OFF)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
  enable_language(CUDA)
  
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.7)
    message(FATAL_ERROR "You must use CUDA version 11.7 or newer to compile Quokka. All previous CUDA versions have compiler bugs that cause Quokka to crash.")
  endif()

  set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "")
  
  if(CMAKE_VERSION VERSION_LESS 3.20)
    include(AMReX_SetupCUDA)
  endif(CMAKE_VERSION VERSION_LESS 3.20)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

if(AMReX_GPU_BACKEND MATCHES "HIP")
  execute_process(
      COMMAND hipcc --version
      RESULT_VARIABLE hipcc_result
      OUTPUT_VARIABLE hipcc_version
      ERROR_VARIABLE hipcc_error
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT hipcc_result EQUAL 0)
      message(FATAL_ERROR "Failed to detect HIP compiler version: ${hipcc_error}")
  else()
      message(STATUS "HIP compiler version: ${hipcc_version}")
  endif()

  if(hipcc_version VERSION_LESS 6.3)
    message(FATAL_ERROR "You must use ROCm version >= 6.3 to compile Quokka. All previous ROCm versions have compiler bugs that cause Quokka to produce incorrect results.")
  endif()
endif(AMReX_GPU_BACKEND MATCHES "HIP")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # abort! Intel Compiler Classic cannot compile Quokka!
  message(FATAL_ERROR "You have configured CMake to use the 'classic' Intel compilers (icc/icpc), which are the old Intel compilers. They cannot compile Quokka correctly! You must use the new LLVM-based Intel compilers (icx/icpx) instead by adding the following CMake command-line options: -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")

# this is necessary to prevent GCC from warning about ARM64 ABI changes in GCC 10.1
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wno-psabi)
endif()

add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/amrex ${QuokkaCode_BINARY_DIR}/amrex)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/fmt ${QuokkaCode_BINARY_DIR}/fmt)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/yaml-cpp ${QuokkaCode_BINARY_DIR}/yaml-cpp)

#add compiler definitions again because
#they do not carry forward from Microphysics
#We use Strang operator-spliting while integrating the network
#NAUX_NET defines the number of auxiliary variables
add_definitions(-DNAUX_NET -DSTRANG)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/extern/Microphysics ${QuokkaCode_BINARY_DIR}/Microphysics)
add_subdirectory(${QuokkaCode_SOURCE_DIR}/src ${QuokkaCode_BINARY_DIR}/src)
