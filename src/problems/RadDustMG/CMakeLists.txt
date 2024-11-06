add_executable(test_rad_dust_MG test_rad_dust_MG.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_rad_dust_MG)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadDustMG COMMAND test_rad_dust_MG RadDust.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
