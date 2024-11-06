if (AMReX_SPACEDIM EQUAL 3)
    # Define a custom target that runs the Python script to produce the input perturbations file


    set(microphysics_network_name "primordial_chem") #this will override network_name to primordial_chem for this directory only
    setup_target_for_microphysics_compilation(${microphysics_network_name} "${CMAKE_CURRENT_BINARY_DIR}/")

    #use the BEFORE keyword so that these files get priority in compilation for targets in this directory
    #this is critical to ensure the correct Microphysics files are linked to primordial chem target
    include_directories(BEFORE ${primordial_chem_dirs} "${CMAKE_CURRENT_BINARY_DIR}/" "includes/extern_parameters.H" "includes/network_properties.H")

    add_executable(popiii popiii.cpp "${QuokkaSourcesNoEOS}" ../../turbulence/TurbDataReader.cpp ../../chemistry/Chemistry.cpp ${primordial_chem_sources})
    target_compile_definitions(popiii PUBLIC CHEMISTRY) #this will add #define CHEMISTRY

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(popiii)
    endif()

    #need h5py to run perturbations.py file below
    #purpose of this code is to check whether the h5py python package is installed
    execute_process(
        COMMAND Python3::Interpreter -c "h5py"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_QUIET
    )

    #re-run cmake if this file is changed
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/PopIII.in)

    # Read the 'PopIII.in' file line by line
    file(READ ${CMAKE_SOURCE_DIR}/tests/PopIII.in pop_in_contents)

    # Split the file contents into lines
    string(REPLACE "\n" ";" pop_in_lines "${pop_in_contents}")

    # Iterate through the lines and look for lines starting with 'amr.n_cell'
    foreach(line IN LISTS pop_in_lines)
        if (line MATCHES "^amr\\.n_cell.*")
            set(match_line "${line}")
            break()
        endif()
    endforeach()

    if (match_line)
        message(STATUS "picked line is ${match_line}")
        # Use a regular expression to find consecutive digit
        string(REGEX MATCHALL "[0-9]+" digits_list "${match_line}")
        # Iterate through the matched digits and extract the first one
        list(GET digits_list 0 first_digits)
        # Check if matched digits were found
        if (first_digits)
           #first_digits give the box size, but you also want box size / 2 as kmax in the script below
           math(EXPR int_first_digits "${first_digits}")
           # Check if the conversion was successful
           if (int_first_digits)
               # Divide the integer by two
               math(EXPR result "${int_first_digits} / 2")
           else()
               message(FATAL_ERROR "Conversion to integer failed, so cannot find kmax!")
           endif()
           message(STATUS "Will create initial perturbations of box size ${first_digits} and kmax ${result}")

        else()
           message(FATAL_ERROR "No box size found to create initial perturbations for!")
        endif()
    else()
        message(FATAL_ERROR "No matching line found in ${CMAKE_SOURCE_DIR}/tests/PopIII.in!")
    endif()


    add_test(NAME ComputePerturbations COMMAND python3 ${CMAKE_SOURCE_DIR}/src/turbulence/perturbation.py --kmin=2 --kmax=${result} --size=${first_digits} --alpha=1.8 --f_solenoidal=0.66667 ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME PopIII COMMAND popiii PopIII.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
    set_tests_properties(ComputePerturbations PROPERTIES FIXTURES_SETUP PopIII_fixture)
    set_tests_properties(PopIII PROPERTIES FIXTURES_REQUIRED PopIII_fixture)

    # AMR test only works on Setonix because Gadi and avatar do not have enough memory per GPU
    # add_test(NAME PopIIIAMR COMMAND popiii popiii_AMR.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
