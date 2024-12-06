#ifndef TEST_RADHYDRO_BB_HPP_ // NOLINT
#define TEST_RADHYDRO_BB_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radhydro_bb.hpp
/// \brief Defines a test problem for blackbody spectrum in a uniform advecting medium.
///

// external headers
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

#include <fstream>

// internal headers

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "radiation/radiation_system.hpp"

// function definitions
auto problem_main() -> int;

#endif // TEST_RADHYDRO_BB_HPP_
