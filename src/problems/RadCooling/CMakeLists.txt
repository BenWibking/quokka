add_executable(test_radiation_cooling test_radiation_cooling.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_cooling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadiationCooling COMMAND test_radiation_cooling RadCooling.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
