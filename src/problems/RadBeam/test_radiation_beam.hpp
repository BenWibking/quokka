#ifndef TEST_RADIATION_BEAM_HPP_ // NOLINT
#define TEST_RADIATION_BEAM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.hpp
/// \brief Defines a test problem for radiation in the static diffusion regime.
///

// external headers

#include <fstream>

// internal headers
#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"

// function definitions
auto testproblem_radiation_beam() -> int;

#endif // TEST_RADIATION_BEAM_HPP_
