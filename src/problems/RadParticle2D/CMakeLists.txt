if (AMReX_SPACEDIM EQUAL 2)
	add_executable(test_radparticle_2D test_radparticle_2D.cpp ../../util/fextract.cpp ${QuokkaObjSources})

	if(AMReX_GPU_BACKEND MATCHES "CUDA")
			setup_target_for_cuda_compilation(test_radparticle_2D)
	endif(AMReX_GPU_BACKEND MATCHES "CUDA")

	add_test(NAME RadParticle2D COMMAND test_radparticle_2D RadParticle2D.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
