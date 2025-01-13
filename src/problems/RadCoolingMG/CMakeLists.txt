add_executable(test_radiation_cooling_MG test_radiation_cooling_MG.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_cooling_MG)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadiationCoolingMG COMMAND test_radiation_cooling_MG RadCooling.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
