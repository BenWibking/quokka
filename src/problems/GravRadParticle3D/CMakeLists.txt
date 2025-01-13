if (AMReX_SPACEDIM EQUAL 3)
	add_executable(test_grav_rad_particle_3D test_grav_rad_particle_3D.cpp ../../util/fextract.cpp ${QuokkaObjSources})

	if(AMReX_GPU_BACKEND MATCHES "CUDA")
			setup_target_for_cuda_compilation(test_grav_rad_particle_3D)
	endif(AMReX_GPU_BACKEND MATCHES "CUDA")

	add_test(NAME GravRadParticle3D COMMAND test_grav_rad_particle_3D GravRadParticle3D.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
