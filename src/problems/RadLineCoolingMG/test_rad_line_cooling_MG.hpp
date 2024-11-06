#ifndef TEST_RAD_LINE_COOLING_MG_HPP_ // NOLINT
#define TEST_RAD_LINE_COOLING_MG_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_rad_line_cooling_MG.hpp
/// \brief Defines a test problem for line cooling and cosmic-ray heating in a uniform medium.
///

// external headers
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif
#include <fmt/format.h>
#include <fstream>

// internal headers

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "radiation/radiation_dust_system.hpp"

// function definitions
auto problem_main() -> int;

#endif // TEST_RAD_LINE_COOLING_MG_HPP_
