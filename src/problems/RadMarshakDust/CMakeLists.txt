add_executable(test_radiation_marshak_dust test_radiation_marshak_dust.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_marshak_dust)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadiationMarshakDust COMMAND test_radiation_marshak_dust RadMarshakDust.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
