add_executable(test_radiation_marshak_Vaytet test_radiation_marshak_Vaytet.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_marshak_Vaytet)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME MarshakWaveVaytet COMMAND test_radiation_marshak_Vaytet MarshakVaytet.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
