add_executable(test_radiation_marshak_asymptotic test_radiation_marshak_asymptotic.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_marshak_asymptotic)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME MarshakWaveAsymptoticDiffusion COMMAND test_radiation_marshak_asymptotic MarshakAsymptotic.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

add_test(NAME MarshakWaveAsymptoticCorr COMMAND test_radiation_marshak_asymptotic MarshakAsymptoticCorr.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
