add_executable(test_radiation_marshak_dust_PE test_radiation_marshak_dust_and_PE.cpp ../../util/fextract.cpp ${QuokkaObjSources})

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(test_radiation_marshak_dust_PE)
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(NAME RadiationMarshakDustPE-coupled COMMAND test_radiation_marshak_dust_PE RadMarshakDustPEcoupled.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME RadiationMarshakDustPE-decoupled COMMAND test_radiation_marshak_dust_PE RadMarshakDustPEdecoupled.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
