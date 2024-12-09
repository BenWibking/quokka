add_executable(test_radhydro_shock_super test_radhydro_shock.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_shock_super)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroShockSuper COMMAND test_radhydro_shock_super radshock_super_dimensionless.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
