//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak_vaytet.cpp
/// \brief Defines a test problem for radiation in the asymptotic diffusion regime.
///

#include "AMReX_BLassert.H"

#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "matplotlibcpp.h"
#include "radiation_system.hpp"
#include "test_radiation_marshak_Vaytet.hpp"

// constexpr int the_model = 0; // 0: constant opacity (Vaytet et al. Sec 3.2.1), 1: nu-dependent opacity (Vaytet et al. Sec 3.2.2), 2: nu-and-T-dependent opacity (Vaytet et al. Sec 3.2.3)
// constexpr int n_groups_ = 6;
// constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = {0.3e12, 0.3e14, 0.6e14, 0.9e14, 1.2e14, 1.5e14, 1.5e16};
// constexpr amrex::GpuArray<double, n_groups_> group_opacities_ = {1000., 750., 500., 250., 10., 10.};

// const std::string modelname = "PPL_free_slope_with_PPL_delta_terms";
const std::string modelname = "PPL_fixed_slope_with_PPL_delta_terms";

// constexpr double nu_pivot = 0.6e14;
constexpr int the_model = 10; // 0: constant opacity (Vaytet et al. Sec 3.2.1), 1: nu-dependent opacity (Vaytet et al. Sec 3.2.2), 2: nu-and-T-dependent opacity (Vaytet et al. Sec 3.2.3)
// 10: bin-centered method with opacities propto nu^-2

constexpr double nu_pivot = 4.0e13;
constexpr int n_coll = 4; // number of collections = 6, to align with Vaytet

// constexpr int n_groups_ = 3;
// constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = {1.0e+10, 6.0e+13, 1.2e+14, 1.0e+16};
// constexpr int n_groups_ = 6;
// constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = {1.0e+10, 3.0e+13, 6.0e+13, 9.0e+13, 1.2e+14, 1.5e+14, 1.0e+16};

// constexpr int n_groups_ = 2; // Be careful
constexpr int n_groups_ = 4;
// constexpr int n_groups_ = 8;
// constexpr int n_groups_ = 16;
// constexpr int n_groups_ = 64;
// constexpr int n_groups_ = 128;
// constexpr int n_groups_ = 256;
// constexpr OpacityModel opacity_model_ = OpacityModel::piecewise_constant_opacity;
constexpr OpacityModel opacity_model_ = OpacityModel::PPL_opacity_fixed_slope_spectrum;
// constexpr OpacityModel opacity_model_ = OpacityModel::PPL_opacity_full_spectrum;

// NEW
constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = []() constexpr {
	// bins cover four orders of magnitutde from 6.0e10 to 6.0e14, with N bins logarithmically spaced
	// in the space of x = h nu / (k_B T) where T = 1000 K, this roughly corresponds to x = 3e-3 to 3e1
	// n_groups_ = 2, 4, 8, 16, 128, 256
	if constexpr (n_groups_ == 2) {
		return amrex::GpuArray<double, 3>{6.0e10, 6.0e12, 6.0e14};
	} else if constexpr (n_groups_ == 4) {
		return amrex::GpuArray<double, 5>{6.0e10, 6.0e11, 6.0e12, 6.0e13, 6.0e14};
	} else if constexpr (n_groups_ == 8) {
		return amrex::GpuArray<double, 9>{6.0000000e+10, 1.8973666e+11, 6.0000000e+11, 1.8973666e+12,
       6.0000000e+12, 1.8973666e+13, 6.0000000e+13, 1.8973666e+14,
       6.0000000e+14};
	} else if constexpr (n_groups_ == 16) {
		return amrex::GpuArray<double, 17>{6.00000000e+10, 1.06696765e+11, 1.89736660e+11, 3.37404795e+11,
       6.00000000e+11, 1.06696765e+12, 1.89736660e+12, 3.37404795e+12,
       6.00000000e+12, 1.06696765e+13, 1.89736660e+13, 3.37404795e+13,
       6.00000000e+13, 1.06696765e+14, 1.89736660e+14, 3.37404795e+14,
       6.00000000e+14};
	} else if constexpr (n_groups_ == 64) {
		return amrex::GpuArray<double, 65>{
			 6.00000000e+10, 6.92869191e+10, 8.00112859e+10, 9.23955916e+10,
       1.06696765e+11, 1.23211502e+11, 1.42282422e+11, 1.64305178e+11,
       1.89736660e+11, 2.19104476e+11, 2.53017902e+11, 2.92180515e+11,
       3.37404795e+11, 3.89628979e+11, 4.49936526e+11, 5.19578594e+11,
       6.00000000e+11, 6.92869191e+11, 8.00112859e+11, 9.23955916e+11,
       1.06696765e+12, 1.23211502e+12, 1.42282422e+12, 1.64305178e+12,
       1.89736660e+12, 2.19104476e+12, 2.53017902e+12, 2.92180515e+12,
       3.37404795e+12, 3.89628979e+12, 4.49936526e+12, 5.19578594e+12,
       6.00000000e+12, 6.92869191e+12, 8.00112859e+12, 9.23955916e+12,
       1.06696765e+13, 1.23211502e+13, 1.42282422e+13, 1.64305178e+13,
       1.89736660e+13, 2.19104476e+13, 2.53017902e+13, 2.92180515e+13,
       3.37404795e+13, 3.89628979e+13, 4.49936526e+13, 5.19578594e+13,
       6.00000000e+13, 6.92869191e+13, 8.00112859e+13, 9.23955916e+13,
       1.06696765e+14, 1.23211502e+14, 1.42282422e+14, 1.64305178e+14,
       1.89736660e+14, 2.19104476e+14, 2.53017902e+14, 2.92180515e+14,
       3.37404795e+14, 3.89628979e+14, 4.49936526e+14, 5.19578594e+14,
       6.00000000e+14};
	} else if constexpr (n_groups_ == 128) {
		return amrex::GpuArray<double, 129>{6.00000000e+10, 6.44764697e+10, 6.92869191e+10, 7.44562656e+10,
       8.00112859e+10, 8.59807542e+10, 9.23955916e+10, 9.92890260e+10,
       1.06696765e+11, 1.14657178e+11, 1.23211502e+11, 1.32404044e+11,
       1.42282422e+11, 1.52897805e+11, 1.64305178e+11, 1.76563631e+11,
       1.89736660e+11, 2.03892500e+11, 2.19104476e+11, 2.35451386e+11,
       2.53017902e+11, 2.71895018e+11, 2.92180515e+11, 3.13979469e+11,
       3.37404795e+11, 3.62577834e+11, 3.89628979e+11, 4.18698351e+11,
       4.49936526e+11, 4.83505313e+11, 5.19578594e+11, 5.58343225e+11,
       6.00000000e+11, 6.44764697e+11, 6.92869191e+11, 7.44562656e+11,
       8.00112859e+11, 8.59807542e+11, 9.23955916e+11, 9.92890260e+11,
       1.06696765e+12, 1.14657178e+12, 1.23211502e+12, 1.32404044e+12,
       1.42282422e+12, 1.52897805e+12, 1.64305178e+12, 1.76563631e+12,
       1.89736660e+12, 2.03892500e+12, 2.19104476e+12, 2.35451386e+12,
       2.53017902e+12, 2.71895018e+12, 2.92180515e+12, 3.13979469e+12,
       3.37404795e+12, 3.62577834e+12, 3.89628979e+12, 4.18698351e+12,
       4.49936526e+12, 4.83505313e+12, 5.19578594e+12, 5.58343225e+12,
       6.00000000e+12, 6.44764697e+12, 6.92869191e+12, 7.44562656e+12,
       8.00112859e+12, 8.59807542e+12, 9.23955916e+12, 9.92890260e+12,
       1.06696765e+13, 1.14657178e+13, 1.23211502e+13, 1.32404044e+13,
       1.42282422e+13, 1.52897805e+13, 1.64305178e+13, 1.76563631e+13,
       1.89736660e+13, 2.03892500e+13, 2.19104476e+13, 2.35451386e+13,
       2.53017902e+13, 2.71895018e+13, 2.92180515e+13, 3.13979469e+13,
       3.37404795e+13, 3.62577834e+13, 3.89628979e+13, 4.18698351e+13,
       4.49936526e+13, 4.83505313e+13, 5.19578594e+13, 5.58343225e+13,
       6.00000000e+13, 6.44764697e+13, 6.92869191e+13, 7.44562656e+13,
       8.00112859e+13, 8.59807542e+13, 9.23955916e+13, 9.92890260e+13,
       1.06696765e+14, 1.14657178e+14, 1.23211502e+14, 1.32404044e+14,
       1.42282422e+14, 1.52897805e+14, 1.64305178e+14, 1.76563631e+14,
       1.89736660e+14, 2.03892500e+14, 2.19104476e+14, 2.35451386e+14,
       2.53017902e+14, 2.71895018e+14, 2.92180515e+14, 3.13979469e+14,
       3.37404795e+14, 3.62577834e+14, 3.89628979e+14, 4.18698351e+14,
       4.49936526e+14, 4.83505313e+14, 5.19578594e+14, 5.58343225e+14,
       6.00000000e+14};
	} else if constexpr (n_groups_ == 256) {
		return amrex::GpuArray<double, 257>{
			 6.00000000e+10, 6.21979757e+10, 6.44764697e+10, 6.68384316e+10,
       6.92869191e+10, 7.18251018e+10, 7.44562656e+10, 7.71838167e+10,
       8.00112859e+10, 8.29423336e+10, 8.59807542e+10, 8.91304810e+10,
       9.23955916e+10, 9.57803127e+10, 9.92890260e+10, 1.02926274e+11,
       1.06696765e+11, 1.10605380e+11, 1.14657178e+11, 1.18857407e+11,
       1.23211502e+11, 1.27725100e+11, 1.32404044e+11, 1.37254392e+11,
       1.42282422e+11, 1.47494644e+11, 1.52897805e+11, 1.58498899e+11,
       1.64305178e+11, 1.70324158e+11, 1.76563631e+11, 1.83031673e+11,
       1.89736660e+11, 1.96687269e+11, 2.03892500e+11, 2.11361679e+11,
       2.19104476e+11, 2.27130915e+11, 2.35451386e+11, 2.44076659e+11,
       2.53017902e+11, 2.62286689e+11, 2.71895018e+11, 2.81855329e+11,
       2.92180515e+11, 3.02883943e+11, 3.13979469e+11, 3.25481456e+11,
       3.37404795e+11, 3.49764921e+11, 3.62577834e+11, 3.75860122e+11,
       3.89628979e+11, 4.03902229e+11, 4.18698351e+11, 4.34036498e+11,
       4.49936526e+11, 4.66419018e+11, 4.83505313e+11, 5.01217528e+11,
       5.19578594e+11, 5.38612279e+11, 5.58343225e+11, 5.78796972e+11,
       6.00000000e+11, 6.21979757e+11, 6.44764697e+11, 6.68384316e+11,
       6.92869191e+11, 7.18251018e+11, 7.44562656e+11, 7.71838167e+11,
       8.00112859e+11, 8.29423336e+11, 8.59807542e+11, 8.91304810e+11,
       9.23955916e+11, 9.57803127e+11, 9.92890260e+11, 1.02926274e+12,
       1.06696765e+12, 1.10605380e+12, 1.14657178e+12, 1.18857407e+12,
       1.23211502e+12, 1.27725100e+12, 1.32404044e+12, 1.37254392e+12,
       1.42282422e+12, 1.47494644e+12, 1.52897805e+12, 1.58498899e+12,
       1.64305178e+12, 1.70324158e+12, 1.76563631e+12, 1.83031673e+12,
       1.89736660e+12, 1.96687269e+12, 2.03892500e+12, 2.11361679e+12,
       2.19104476e+12, 2.27130915e+12, 2.35451386e+12, 2.44076659e+12,
       2.53017902e+12, 2.62286689e+12, 2.71895018e+12, 2.81855329e+12,
       2.92180515e+12, 3.02883943e+12, 3.13979469e+12, 3.25481456e+12,
       3.37404795e+12, 3.49764921e+12, 3.62577834e+12, 3.75860122e+12,
       3.89628979e+12, 4.03902229e+12, 4.18698351e+12, 4.34036498e+12,
       4.49936526e+12, 4.66419018e+12, 4.83505313e+12, 5.01217528e+12,
       5.19578594e+12, 5.38612279e+12, 5.58343225e+12, 5.78796972e+12,
       6.00000000e+12, 6.21979757e+12, 6.44764697e+12, 6.68384316e+12,
       6.92869191e+12, 7.18251018e+12, 7.44562656e+12, 7.71838167e+12,
       8.00112859e+12, 8.29423336e+12, 8.59807542e+12, 8.91304810e+12,
       9.23955916e+12, 9.57803127e+12, 9.92890260e+12, 1.02926274e+13,
       1.06696765e+13, 1.10605380e+13, 1.14657178e+13, 1.18857407e+13,
       1.23211502e+13, 1.27725100e+13, 1.32404044e+13, 1.37254392e+13,
       1.42282422e+13, 1.47494644e+13, 1.52897805e+13, 1.58498899e+13,
       1.64305178e+13, 1.70324158e+13, 1.76563631e+13, 1.83031673e+13,
       1.89736660e+13, 1.96687269e+13, 2.03892500e+13, 2.11361679e+13,
       2.19104476e+13, 2.27130915e+13, 2.35451386e+13, 2.44076659e+13,
       2.53017902e+13, 2.62286689e+13, 2.71895018e+13, 2.81855329e+13,
       2.92180515e+13, 3.02883943e+13, 3.13979469e+13, 3.25481456e+13,
       3.37404795e+13, 3.49764921e+13, 3.62577834e+13, 3.75860122e+13,
       3.89628979e+13, 4.03902229e+13, 4.18698351e+13, 4.34036498e+13,
       4.49936526e+13, 4.66419018e+13, 4.83505313e+13, 5.01217528e+13,
       5.19578594e+13, 5.38612279e+13, 5.58343225e+13, 5.78796972e+13,
       6.00000000e+13, 6.21979757e+13, 6.44764697e+13, 6.68384316e+13,
       6.92869191e+13, 7.18251018e+13, 7.44562656e+13, 7.71838167e+13,
       8.00112859e+13, 8.29423336e+13, 8.59807542e+13, 8.91304810e+13,
       9.23955916e+13, 9.57803127e+13, 9.92890260e+13, 1.02926274e+14,
       1.06696765e+14, 1.10605380e+14, 1.14657178e+14, 1.18857407e+14,
       1.23211502e+14, 1.27725100e+14, 1.32404044e+14, 1.37254392e+14,
       1.42282422e+14, 1.47494644e+14, 1.52897805e+14, 1.58498899e+14,
       1.64305178e+14, 1.70324158e+14, 1.76563631e+14, 1.83031673e+14,
       1.89736660e+14, 1.96687269e+14, 2.03892500e+14, 2.11361679e+14,
       2.19104476e+14, 2.27130915e+14, 2.35451386e+14, 2.44076659e+14,
       2.53017902e+14, 2.62286689e+14, 2.71895018e+14, 2.81855329e+14,
       2.92180515e+14, 3.02883943e+14, 3.13979469e+14, 3.25481456e+14,
       3.37404795e+14, 3.49764921e+14, 3.62577834e+14, 3.75860122e+14,
       3.89628979e+14, 4.03902229e+14, 4.18698351e+14, 4.34036498e+14,
       4.49936526e+14, 4.66419018e+14, 4.83505313e+14, 5.01217528e+14,
       5.19578594e+14, 5.38612279e+14, 5.58343225e+14, 5.78796972e+14,
       6.00000000e+14};
	} 
}();


// constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = []() constexpr {
// 	if constexpr (n_groups_ == 2) {
// 		return amrex::GpuArray<double, 3>{1.0e+11, 8.0e+13, 1.0e+16};
// 	} else if constexpr (n_groups_ == 4) {
// 		return amrex::GpuArray<double, 5>{1.0e+11, 4.0e+13, 8.0e+13, 1.2e+14, 1.0e+16};
// 	} else if constexpr (n_groups_ == 8) {
// 		return amrex::GpuArray<double, 9>{1.0e+11, 2.0e+13, 4.0e+13, 6.0e+13, 8.0e+13, 1.0e+14, 1.2e+14, 1.4e+14, 1.0e+16};
// 	} else if constexpr (n_groups_ == 16) {
// 		return amrex::GpuArray<double, 17>{1.0e+11, 1.0e+13, 2.0e+13, 3.0e+13, 4.0e+13, 5.0e+13, 6.0e+13,
//        7.0e+13, 8.0e+13, 9.0e+13, 1.0e+14, 1.1e+14, 1.2e+14, 1.3e+14,
//        1.4e+14, 1.5e+14, 1.0e+16};
// 	} else if constexpr (n_groups_ == 128) {
// 		return amrex::GpuArray<double, 129>{
// 			 1.0000e+11, 1.2500e+12, 2.5000e+12, 3.7500e+12, 5.0000e+12,
//        6.2500e+12, 7.5000e+12, 8.7500e+12, 1.0000e+13, 1.1250e+13,
//        1.2500e+13, 1.3750e+13, 1.5000e+13, 1.6250e+13, 1.7500e+13,
//        1.8750e+13, 2.0000e+13, 2.1250e+13, 2.2500e+13, 2.3750e+13,
//        2.5000e+13, 2.6250e+13, 2.7500e+13, 2.8750e+13, 3.0000e+13,
//        3.1250e+13, 3.2500e+13, 3.3750e+13, 3.5000e+13, 3.6250e+13,
//        3.7500e+13, 3.8750e+13, 4.0000e+13, 4.1250e+13, 4.2500e+13,
//        4.3750e+13, 4.5000e+13, 4.6250e+13, 4.7500e+13, 4.8750e+13,
//        5.0000e+13, 5.1250e+13, 5.2500e+13, 5.3750e+13, 5.5000e+13,
//        5.6250e+13, 5.7500e+13, 5.8750e+13, 6.0000e+13, 6.1250e+13,
//        6.2500e+13, 6.3750e+13, 6.5000e+13, 6.6250e+13, 6.7500e+13,
//        6.8750e+13, 7.0000e+13, 7.1250e+13, 7.2500e+13, 7.3750e+13,
//        7.5000e+13, 7.6250e+13, 7.7500e+13, 7.8750e+13, 8.0000e+13,
//        8.1250e+13, 8.2500e+13, 8.3750e+13, 8.5000e+13, 8.6250e+13,
//        8.7500e+13, 8.8750e+13, 9.0000e+13, 9.1250e+13, 9.2500e+13,
//        9.3750e+13, 9.5000e+13, 9.6250e+13, 9.7500e+13, 9.8750e+13,
//        1.0000e+14, 1.0125e+14, 1.0250e+14, 1.0375e+14, 1.0500e+14,
//        1.0625e+14, 1.0750e+14, 1.0875e+14, 1.1000e+14, 1.1125e+14,
//        1.1250e+14, 1.1375e+14, 1.1500e+14, 1.1625e+14, 1.1750e+14,
//        1.1875e+14, 1.2000e+14, 1.2125e+14, 1.2250e+14, 1.2375e+14,
//        1.2500e+14, 1.2625e+14, 1.2750e+14, 1.2875e+14, 1.3000e+14,
//        1.3125e+14, 1.3250e+14, 1.3375e+14, 1.3500e+14, 1.3625e+14,
//        1.3750e+14, 1.3875e+14, 1.4000e+14, 1.4125e+14, 1.4250e+14,
//        1.4375e+14, 1.4500e+14, 1.4625e+14, 1.4750e+14, 1.4875e+14,
//        1.5000e+14, 1.5125e+14, 1.5250e+14, 1.5375e+14, 1.5500e+14,
//        1.5625e+14, 1.5750e+14, 1.5875e+14, 1.0000e+16};
// 	} else if constexpr (n_groups_ == 256) {
// 		return amrex::GpuArray<double, 257>{1.00000e+11, 6.25000e+11, 1.25000e+12, 1.87500e+12, 2.50000e+12,
//        3.12500e+12, 3.75000e+12, 4.37500e+12, 5.00000e+12, 5.62500e+12,
//        6.25000e+12, 6.87500e+12, 7.50000e+12, 8.12500e+12, 8.75000e+12,
//        9.37500e+12, 1.00000e+13, 1.06250e+13, 1.12500e+13, 1.18750e+13,
//        1.25000e+13, 1.31250e+13, 1.37500e+13, 1.43750e+13, 1.50000e+13,
//        1.56250e+13, 1.62500e+13, 1.68750e+13, 1.75000e+13, 1.81250e+13,
//        1.87500e+13, 1.93750e+13, 2.00000e+13, 2.06250e+13, 2.12500e+13,
//        2.18750e+13, 2.25000e+13, 2.31250e+13, 2.37500e+13, 2.43750e+13,
//        2.50000e+13, 2.56250e+13, 2.62500e+13, 2.68750e+13, 2.75000e+13,
//        2.81250e+13, 2.87500e+13, 2.93750e+13, 3.00000e+13, 3.06250e+13,
//        3.12500e+13, 3.18750e+13, 3.25000e+13, 3.31250e+13, 3.37500e+13,
//        3.43750e+13, 3.50000e+13, 3.56250e+13, 3.62500e+13, 3.68750e+13,
//        3.75000e+13, 3.81250e+13, 3.87500e+13, 3.93750e+13, 4.00000e+13,
//        4.06250e+13, 4.12500e+13, 4.18750e+13, 4.25000e+13, 4.31250e+13,
//        4.37500e+13, 4.43750e+13, 4.50000e+13, 4.56250e+13, 4.62500e+13,
//        4.68750e+13, 4.75000e+13, 4.81250e+13, 4.87500e+13, 4.93750e+13,
//        5.00000e+13, 5.06250e+13, 5.12500e+13, 5.18750e+13, 5.25000e+13,
//        5.31250e+13, 5.37500e+13, 5.43750e+13, 5.50000e+13, 5.56250e+13,
//        5.62500e+13, 5.68750e+13, 5.75000e+13, 5.81250e+13, 5.87500e+13,
//        5.93750e+13, 6.00000e+13, 6.06250e+13, 6.12500e+13, 6.18750e+13,
//        6.25000e+13, 6.31250e+13, 6.37500e+13, 6.43750e+13, 6.50000e+13,
//        6.56250e+13, 6.62500e+13, 6.68750e+13, 6.75000e+13, 6.81250e+13,
//        6.87500e+13, 6.93750e+13, 7.00000e+13, 7.06250e+13, 7.12500e+13,
//        7.18750e+13, 7.25000e+13, 7.31250e+13, 7.37500e+13, 7.43750e+13,
//        7.50000e+13, 7.56250e+13, 7.62500e+13, 7.68750e+13, 7.75000e+13,
//        7.81250e+13, 7.87500e+13, 7.93750e+13, 8.00000e+13, 8.06250e+13,
//        8.12500e+13, 8.18750e+13, 8.25000e+13, 8.31250e+13, 8.37500e+13,
//        8.43750e+13, 8.50000e+13, 8.56250e+13, 8.62500e+13, 8.68750e+13,
//        8.75000e+13, 8.81250e+13, 8.87500e+13, 8.93750e+13, 9.00000e+13,
//        9.06250e+13, 9.12500e+13, 9.18750e+13, 9.25000e+13, 9.31250e+13,
//        9.37500e+13, 9.43750e+13, 9.50000e+13, 9.56250e+13, 9.62500e+13,
//        9.68750e+13, 9.75000e+13, 9.81250e+13, 9.87500e+13, 9.93750e+13,
//        1.00000e+14, 1.00625e+14, 1.01250e+14, 1.01875e+14, 1.02500e+14,
//        1.03125e+14, 1.03750e+14, 1.04375e+14, 1.05000e+14, 1.05625e+14,
//        1.06250e+14, 1.06875e+14, 1.07500e+14, 1.08125e+14, 1.08750e+14,
//        1.09375e+14, 1.10000e+14, 1.10625e+14, 1.11250e+14, 1.11875e+14,
//        1.12500e+14, 1.13125e+14, 1.13750e+14, 1.14375e+14, 1.15000e+14,
//        1.15625e+14, 1.16250e+14, 1.16875e+14, 1.17500e+14, 1.18125e+14,
//        1.18750e+14, 1.19375e+14, 1.20000e+14, 1.20625e+14, 1.21250e+14,
//        1.21875e+14, 1.22500e+14, 1.23125e+14, 1.23750e+14, 1.24375e+14,
//        1.25000e+14, 1.25625e+14, 1.26250e+14, 1.26875e+14, 1.27500e+14,
//        1.28125e+14, 1.28750e+14, 1.29375e+14, 1.30000e+14, 1.30625e+14,
//        1.31250e+14, 1.31875e+14, 1.32500e+14, 1.33125e+14, 1.33750e+14,
//        1.34375e+14, 1.35000e+14, 1.35625e+14, 1.36250e+14, 1.36875e+14,
//        1.37500e+14, 1.38125e+14, 1.38750e+14, 1.39375e+14, 1.40000e+14,
//        1.40625e+14, 1.41250e+14, 1.41875e+14, 1.42500e+14, 1.43125e+14,
//        1.43750e+14, 1.44375e+14, 1.45000e+14, 1.45625e+14, 1.46250e+14,
//        1.46875e+14, 1.47500e+14, 1.48125e+14, 1.48750e+14, 1.49375e+14,
//        1.50000e+14, 1.50625e+14, 1.51250e+14, 1.51875e+14, 1.52500e+14,
//        1.53125e+14, 1.53750e+14, 1.54375e+14, 1.55000e+14, 1.55625e+14,
//        1.56250e+14, 1.56875e+14, 1.57500e+14, 1.58125e+14, 1.58750e+14,
//        1.59375e+14, 1.00000e+16};
// 	}
// }();

// constexpr int n_groups_ = 6 * 8;
// constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = {1e+10, 3.7500e+12, 7.5000e+12, 1.1250e+13, 1.5000e+13,
//        1.8750e+13, 2.2500e+13, 2.6250e+13, 3.0000e+13, 3.3750e+13,
//        3.7500e+13, 4.1250e+13, 4.5000e+13, 4.8750e+13, 5.2500e+13,
//        5.6250e+13, 6.0000e+13, 6.3750e+13, 6.7500e+13, 7.1250e+13,
//        7.5000e+13, 7.8750e+13, 8.2500e+13, 8.6250e+13, 9.0000e+13,
//        9.3750e+13, 9.7500e+13, 1.0125e+14, 1.0500e+14, 1.0875e+14,
//        1.1250e+14, 1.1625e+14, 1.2000e+14, 1.2375e+14, 1.2750e+14,
//        1.3125e+14, 1.3500e+14, 1.3875e+14, 1.4250e+14, 1.4625e+14,
//        1.5000e+14, 1.5375e+14, 1.5750e+14, 1.6125e+14, 1.6500e+14,
//        1.6875e+14, 1.7250e+14, 1.7625e+14, 1e+16};

// constexpr int n_groups_ = 6 * 16;
// constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = {1.00000e+10, 1.87500e+12, 3.75000e+12, 5.62500e+12, 7.50000e+12,
//        9.37500e+12, 1.12500e+13, 1.31250e+13, 1.50000e+13, 1.68750e+13,
//        1.87500e+13, 2.06250e+13, 2.25000e+13, 2.43750e+13, 2.62500e+13,
//        2.81250e+13, 3.00000e+13, 3.18750e+13, 3.37500e+13, 3.56250e+13,
//        3.75000e+13, 3.93750e+13, 4.12500e+13, 4.31250e+13, 4.50000e+13,
//        4.68750e+13, 4.87500e+13, 5.06250e+13, 5.25000e+13, 5.43750e+13,
//        5.62500e+13, 5.81250e+13, 6.00000e+13, 6.18750e+13, 6.37500e+13,
//        6.56250e+13, 6.75000e+13, 6.93750e+13, 7.12500e+13, 7.31250e+13,
//        7.50000e+13, 7.68750e+13, 7.87500e+13, 8.06250e+13, 8.25000e+13,
//        8.43750e+13, 8.62500e+13, 8.81250e+13, 9.00000e+13, 9.18750e+13,
//        9.37500e+13, 9.56250e+13, 9.75000e+13, 9.93750e+13, 1.01250e+14,
//        1.03125e+14, 1.05000e+14, 1.06875e+14, 1.08750e+14, 1.10625e+14,
//        1.12500e+14, 1.14375e+14, 1.16250e+14, 1.18125e+14, 1.20000e+14,
//        1.21875e+14, 1.23750e+14, 1.25625e+14, 1.27500e+14, 1.29375e+14,
//        1.31250e+14, 1.33125e+14, 1.35000e+14, 1.36875e+14, 1.38750e+14,
//        1.40625e+14, 1.42500e+14, 1.44375e+14, 1.46250e+14, 1.48125e+14,
//        1.50000e+14, 1.51875e+14, 1.53750e+14, 1.55625e+14, 1.57500e+14,
//        1.59375e+14, 1.61250e+14, 1.63125e+14, 1.65000e+14, 1.66875e+14,
//        1.68750e+14, 1.70625e+14, 1.72500e+14, 1.74375e+14, 1.76250e+14,
//        1.78125e+14, 1.00000e+16};

constexpr amrex::GpuArray<double, n_groups_> group_opacities_{};

struct SuOlsonProblemCgs {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int max_step_ = 1e6;
constexpr double kappa = 2000.0; // cm^2 g^-1 (opacity)
constexpr double rho0 = 1.0e-3; // g cm^-3
constexpr double T_initial = 300.0; // K
constexpr double T_L = 1000.0; // K
constexpr double T_R = 300.0; // K
constexpr double rho_C_V = 1.0e-3; // erg cm^-3 K^-1
constexpr double c_v = rho_C_V / rho0;
constexpr double mu = 1.0 / (5. / 3. - 1.) * C::k_B / c_v;

constexpr double a_rad = radiation_constant_cgs_;
constexpr double Erad_floor_ = a_rad * T_initial * T_initial * T_initial * T_initial * 1e-20;

template <> struct quokka::EOS_Traits<SuOlsonProblemCgs> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<SuOlsonProblemCgs> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_; // number of radiation groups
};

template <> struct RadSystem_Traits<SuOlsonProblemCgs> {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = c_light_cgs_;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::hplanck; // set boundary unit to Hz
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = group_edges_;
	static constexpr OpacityModel opacity_model = opacity_model_;
};

// template <>
// template <typename ArrayType>
// AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblemCgs>::ComputeRadQuantityExponents(ArrayType const &/*quant*/, amrex::GpuArray<double, nGroups_ + 1> const &/*boundaries*/)
//     -> amrex::GpuArray<double, nGroups_>
// {
// 	amrex::GpuArray<double, nGroups_> exponents{};
// 	// for (int i = 0; i < nGroups_; ++i) {
// 	// 	exponents[i] = -1.0;
// 	// }
// 	exponents[0] = 2.0;
// 	exponents[nGroups_ - 1] = -4.0;
// 	for (int i = 1; i < nGroups_ - 1; ++i) {
// 		exponents[i] = -1.0;
// 	}
// 	return exponents;
// }

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/, const double Tgas)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		if constexpr (the_model < 10) {
			exponents_and_values[0][i] = 0.0;
		} else {
			exponents_and_values[0][i] = -2.0;
		}
	}
		if constexpr (the_model == 0) {
			for (int i = 0; i < nGroups_; ++i) {
				exponents_and_values[1][i] = kappa;
			}
		} else if constexpr (the_model == 1) {
			for (int i = 0; i < nGroups_; ++i) {
				exponents_and_values[1][i] = group_opacities_[i];
			}
		} else if constexpr (the_model == 2) {
			for (int i = 0; i < nGroups_; ++i) {
				exponents_and_values[1][i] = group_opacities_[i] * std::pow(Tgas / T_initial, 3./2.);
			}
		} else if constexpr (the_model == 10) {
			if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
				for (int i = 0; i < nGroups_; ++i) {
					auto const bin_center = std::sqrt(group_edges_[i] * group_edges_[i + 1]);
					exponents_and_values[1][i] = kappa * std::pow(bin_center / nu_pivot, -2.);
				}
			} else {
				for (int i = 0; i < nGroups_ + 1; ++i) {
					exponents_and_values[1][i] = kappa * std::pow(group_edges_[i] / nu_pivot, -2.);
				}
			}
		}	
	return exponents_and_values;
}

// template <>
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(const double /*rho*/, const double Tgas)
//     -> quokka::valarray<double, nGroups_>
// {
// 	quokka::valarray<double, nGroups_> kappaPVec{};
// 	for (int i = 0; i < nGroups_; ++i) {
// 		if constexpr (the_model == 0) {
// 			kappaPVec[i] = kappa;
// 		} else if constexpr (the_model == 10) {
// 			auto const bin_center = std::sqrt(group_edges_[i] * group_edges_[i + 1]);
// 			kappaPVec[i] = kappa * std::pow(bin_center / nu_pivot, -2.);
// 		} else if constexpr (the_model == 1) {
// 			kappaPVec[i] = group_opacities_[i];
// 		} else {
// 			kappaPVec[i] = group_opacities_[i] * std::pow(Tgas / T_initial, 3./2.);
// 		}
// 	}
// 	return kappaPVec;
// }

// template <>
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputeFluxMeanOpacity(const double rho, const double Tgas)
//     -> quokka::valarray<double, nGroups_>
// {
// 	return ComputePlanckOpacity(rho, Tgas);
// }

// template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<SuOlsonProblemCgs>::ComputeEddingtonFactor(double /*f*/) -> double
// {
// 	return (1. / 3.); // Eddington approximation
// }

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<SuOlsonProblemCgs>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
							      int /*numcomp*/, amrex::GeometryData const & geom, const amrex::Real /*time*/,
							      const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int j = 0;
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();

	auto const radBoundaries_g = RadSystem<SuOlsonProblemCgs>::radBoundaries_;

	if (i < lo[0] || i >= hi[0]) {
		double T_H = NAN;
		if (i < lo[0]) {
			T_H = T_L;
		} else {
			T_H = T_R;
		}

		auto Erad_g = RadSystem<SuOlsonProblemCgs>::ComputeThermalRadiation(T_H, radBoundaries_g);
		const double Egas = quokka::EOS<SuOlsonProblemCgs>::ComputeEintFromTgas(rho0, T_initial);

		for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}

		// gas boundary conditions are the same on both sides
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasDensity_index) = rho0;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3GasMomentum_index) = 0.;
	}
}

template <> void RadhydroSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto radBoundaries_g = RadSystem<SuOlsonProblemCgs>::radBoundaries_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double Egas = quokka::EOS<SuOlsonProblemCgs>::ComputeEintFromTgas(rho0, T_initial);
		// const double Erad = a_rad * std::pow(T_initial, 4);
		auto Erad_g = RadSystem<SuOlsonProblemCgs>::ComputeThermalRadiation(T_initial, radBoundaries_g);

		// state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index) = Erad_g;
		// state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index) = 0;
		// state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index) = 0;
		// state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index) = 0;
		for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem tests whether the numerical scheme is asymptotic preserving.
	// This requires both a spatial discretization *and* a temporal discretization
	// that have the asymptotic-preserving property. Operator splitting the
	// transport and source terms can give a splitting error that is arbitrarily
	// large in the asymptotic limit! A fully implicit method or a semi-implicit
	// predictor-corrector method [2] similar to SDC is REQUIRED for a correct solution.
	//
	// For discussion of the asymptotic-preserving property, see [1] and [2]. For
	// a discussion of the exact, self-similar solution to this problem, see [3].
	// Note that when used with an SDC time integrator, PLM (w/ asymptotic correction
	// in Riemann solver) does a good job, but not quite as good as linear DG on this
	// problem. There are some 'stair-stepping' artifacts that appear with PLM at low
	// resolution that do not appear when using DG. This is likely the "wide stencil"
	// issue discussed in [4].
	//
	// 1. R.G. McClarren, R.B. Lowrie, The effects of slope limiting on asymptotic-preserving
	//     numerical methods for hyperbolic conservation laws, Journal of
	//     Computational Physics 227 (2008) 9711–9726.
	// 2. R.G. McClarren, T.M. Evans, R.B. Lowrie, J.D. Densmore, Semi-implicit time integration
	//     for PN thermal radiative transfer, Journal of Computational Physics 227
	//     (2008) 7561-7586.
	// 3. Y. Zel'dovich, Y. Raizer, Physics of Shock Waves and High-Temperature Hydrodynamic
	//     Phenomena (1964), Ch. X.: Thermal Waves.
	// 4. Lowrie, R. B. and Morel, J. E., Issues with high-resolution Godunov methods for
	//     radiation hydrodynamics, Journal of Quantitative Spectroscopy and
	//     Radiative Transfer, 69, 475–489, 2001.

	// Problem parameters
	const int max_timesteps = max_step_;
	const double CFL_number = 0.8;
	// const double initial_dt = 5.0e-12; // s
	const double max_dt = 1.0;	   // s
	const double max_time = 1.36e-7;

	constexpr int nvars = RadSystem<SuOlsonProblemCgs>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0,
				amrex::BCType::ext_dir);     // custom (Marshak) x1
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	RadhydroSimulation<SuOlsonProblemCgs> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	// sim.initDt_ = initial_dt;
	sim.maxDt_ = max_dt;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// compare against diffusion solution
	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> Trad(nx);
		std::vector<double> Tgas(nx);
		std::vector<double> vgas(nx);
		// define a vector of n_groups_ vectors
		std::vector<std::vector<double>> Trad_g(n_groups_);

		int const n_item = n_groups_ / n_coll;
		std::vector<std::vector<double>> Trad_coll(n_coll);

		for (int i = 0; i < nx; ++i) {
			double Erad_t = 0.;
			// const double Erad_t = values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index)[i];
			for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
				Erad_t += values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
			}
			const double Egas_t = values.at(RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index)[i];
			const double rho = values.at(RadSystem<SuOlsonProblemCgs>::gasDensity_index)[i];
			amrex::Real const x = position[i];
			xs.at(i) = x;
			Tgas.at(i) = quokka::EOS<SuOlsonProblemCgs>::ComputeTgasFromEint(rho, Egas_t);
			Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);

			// vgas
			const auto x1GasMomentum = values.at(RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index)[i];
			vgas.at(i) = x1GasMomentum / rho;

			int counter = 0;
			int group_counter = 0;
			double Erad_sum = 0.;
			for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
				auto Erad_g = values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
				Trad_g[g].push_back(std::pow(Erad_g / a_rad, 1. / 4.));

				if (counter == 0) {
					Erad_sum = 0.0;
				}
				Erad_sum += Erad_g;
				if (counter == n_item - 1) {
					Trad_coll[group_counter].push_back(std::pow(Erad_sum / a_rad, 1. / 4.));
					group_counter++;
					counter = 0;
				} else {
					counter++;
				}
			}
		}

		// // read in exact solution

		// std::vector<double> xs_exact;
		// std::vector<double> Tmat_exact;

		// std::string filename = "../extern/marshak_similarity.csv";
		// std::ifstream fstream(filename, std::ios::in);
		// AMREX_ALWAYS_ASSERT(fstream.is_open());

		// std::string header;
		// std::getline(fstream, header);

		// for (std::string line; std::getline(fstream, line);) {
		// 	std::istringstream iss(line);
		// 	std::vector<double> values;

		// 	for (double value = NAN; iss >> value;) {
		// 		values.push_back(value);
		// 	}
		// 	auto x_val = values.at(0);
		// 	auto Tmat_val = values.at(1);

		// 	xs_exact.push_back(x_val);
		// 	Tmat_exact.push_back(Tmat_val);
		// }

		// // compute error norm

		// // interpolate numerical solution onto exact tabulated solution
		// std::vector<double> Tmat_interp(xs_exact.size());
		// interpolate_arrays(xs_exact.data(), Tmat_interp.data(), static_cast<int>(xs_exact.size()), xs.data(), Tgas.data(), static_cast<int>(xs.size()));

		// double err_norm = 0.;
		// double sol_norm = 0.;
		// for (size_t i = 0; i < xs_exact.size(); ++i) {
		// 	err_norm += std::abs(Tmat_interp[i] - Tmat_exact[i]);
		// 	sol_norm += std::abs(Tmat_exact[i]);
		// }

		// const double error_tol = 0.09;
		// const double rel_error = err_norm / sol_norm;
		// amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

		// save data to file
		std::ofstream fstream;
		fstream.open("marshak_wave_Vaytet.csv");
		fstream << "# x, Tgas, Trad";
		for (int i = 0; i < n_groups_; ++i) {
			fstream << ", " << "Trad_" << i;
		}
		for (int i = 0; i < nx; ++i) {
			fstream << std::endl;
			fstream << std::scientific << std::setprecision(14) << xs[i] << ", " << Tgas[i] << ", " << Trad[i];
			for (int j = 0; j < n_groups_; ++j) {
				fstream << ", " << Trad_g[j][i];
			}
		}
		fstream.close();
		
		// save Trad_coll to file
		std::ofstream fstream_coll;
		fstream_coll.open("marshak_wave_Vaytet_coll.csv");
		fstream_coll << "# x, Tgas, Trad";
		for (int i = 0; i < n_coll; ++i) {
			fstream_coll << ", " << "Trad_" << i;
		}
		for (int i = 0; i < nx; ++i) {
			fstream_coll << std::endl;
			fstream_coll << std::scientific << std::setprecision(14) << xs[i] << ", " << Tgas[i] << ", " << Trad[i];
			for (int j = 0; j < n_coll; ++j) {
				fstream_coll << ", " << Trad_coll[j][i];
			}
		}
		fstream_coll.close();

		// // check if velocity is strictly zero
		// const double error_v_tol = 1.0e-10;
		// double error_v = 0.0;
		// const double cs = std::sqrt(5. / 3. * C::k_B / mu * T_initial); // sound speed
		// for (size_t i = 0; i < xs.size(); ++i) {
		// 	error_v += std::abs(vgas[i]) / cs;
		// }
		// amrex::Print() << "Sum of abs(v) / cs = " << error_v << std::endl;
		// if ((error_v > error_v_tol) || std::isnan(error_v)) {
		// 	status = 1;
		// }

#ifdef HAVE_PYTHON
		// plot results
		matplotlibcpp::clf();
		std::map<std::string, std::string> args;
		args["label"] = "gas";
		args["linestyle"] = "-";
		args["color"] = "k";
		matplotlibcpp::plot(xs, Tgas, args);
		args["label"] = "radiation";
		args["linestyle"] = "--";
		args["color"] = "k";
		// args["marker"] = "x";
		matplotlibcpp::plot(xs, Trad, args);

		for (int g = 0; g < n_coll; ++g) {
			std::map<std::string, std::string> Trad_coll_args;
			Trad_coll_args["label"] = fmt::format("group {}", g);
			Trad_coll_args["linestyle"] = "-";
			Trad_coll_args["color"] = "C" + std::to_string(g);
			matplotlibcpp::plot(xs, Trad_coll[g], Trad_coll_args);
		}

		// Tgas_exact_args["label"] = "gas temperature (exact)";
		// Tgas_exact_args["color"] = "C0";
		// // Tgas_exact_args["marker"] = "x";
		// matplotlibcpp::plot(xs_exact, Tmat_exact, Tgas_exact_args);

		matplotlibcpp::xlim(0.0, 12.0); // cm
		matplotlibcpp::ylim(0.0, 1000.0);	// K
		matplotlibcpp::xlabel("length x (cm)");
		matplotlibcpp::ylabel("temperature (K)");
		matplotlibcpp::legend();
		// matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
		matplotlibcpp::title(modelname);
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./marshak_wave_Vaytet.pdf");
#endif // HAVE_PYTHON
	}

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	// if ((rel_error > error_tol) || std::isnan(rel_error)) {
	// 	status = 1;
	// }
	return status;
}