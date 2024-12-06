#ifndef TEST_ODE_HPP_ // NOLINT
#define TEST_ODE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_ode.hpp
/// \brief Defines a test problem for ODE integration.
///

// external headers


// internal headers
#include "math/ODEIntegrate.hpp"
#include "util/valarray.hpp"

// types

struct ODETest {
};

constexpr double seconds_in_year = 3.154e7;

// function definitions

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real t, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int;

#endif // TEST_ODE_HPP_
