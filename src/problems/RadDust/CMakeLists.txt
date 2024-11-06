add_executable(test_rad_dust test_rad_dust.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_rad_dust)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadDust COMMAND test_rad_dust RadDust.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
