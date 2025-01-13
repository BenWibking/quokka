if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_radhydro3d_shell test_radhydro_shell.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(test_radhydro3d_shell)
    endif()

    add_test(NAME RadhydroShell COMMAND test_radhydro3d_shell radhydro_shell.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
