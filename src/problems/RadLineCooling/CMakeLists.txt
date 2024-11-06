add_executable(test_rad_line_cooling test_rad_line_cooling.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_rad_line_cooling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadLineCooling COMMAND test_rad_line_cooling RadLineCooling.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
