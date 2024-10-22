add_executable(test_rad_thermal_cooling test_rad_thermal_cooling.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_rad_thermal_cooling)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadThermalCooling COMMAND test_rad_thermal_cooling RadThermalCooling.in WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
