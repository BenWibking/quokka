add_executable(test_radhydro_bb test_radhydro_bb.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radhydro_bb)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadhydroBB COMMAND test_radhydro_bb RadhydroBB.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
