if(AMREX_SPACEDIM EQUAL 1)
	if(AMReX_GPU_BACKEND MATCHES "CUDA")
			setup_target_for_cuda_compilation(test_radparticle)
	endif(AMReX_GPU_BACKEND MATCHES "CUDA")

	add_test(NAME RadParticle COMMAND test_radparticle RadParticle.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()