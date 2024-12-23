add_executable(paper3_linear_diffusion test_linear_diffusion_multigroup.cpp ../../util/fextract.cpp ../../math/interpolate.cpp ${QuokkaObjSources})

add_test(NAME Paper3LinearDiffusion COMMAND paper3_linear_diffusion LinearDiffusion.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
