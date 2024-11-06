#ifndef HYPERBOLIC_SYSTEM_HPP_ // NOLINT
#define HYPERBOLIC_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file hyperbolic_system.hpp
/// \brief Defines classes and functions for use with hyperbolic systems of
/// conservation laws.
///
/// This file provides classes, data structures and functions for hyperbolic
/// systems of conservation laws.
///

// c++ headers
#include <cmath>

// library headers
#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_SPACE.H"

// internal headers
#include "math/math_impl.hpp"
#include "util/ArrayView.hpp"

// #define MULTIDIM_EXTREMA_CHECK

namespace quokka
{
enum redoFlag { none = 0, redo = 1 };
} // namespace quokka

// Define enum for slope limiter type
enum SlopeLimiter { minmod = 0, MC };

using array_t = amrex::Array4<amrex::Real> const;
using arrayconst_t = amrex::Array4<const amrex::Real> const;

/// Class for a hyperbolic system of conservation laws
template <typename problem_t> class HyperbolicSystem
{
      public:
	template <SlopeLimiter limiter> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto SlopeFunc(amrex::Real x, amrex::Real y) -> amrex::Real
	{
		static_assert(limiter == SlopeLimiter::minmod || limiter == SlopeLimiter::MC, "Invalid slope limiter specified.");
		if constexpr (limiter == SlopeLimiter::minmod) {
			return minmod(x, y);
		}
		if constexpr (limiter == SlopeLimiter::MC) {
			return MC(x, y);
		}
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) * std::min(0.5 * std::abs(a + b), std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod(double a, double b) -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) * std::min(std::abs(a), std::abs(b));
	}

	[[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE static auto GetMinmaxSurroundingCell(arrayconst_t &q, int i, int j, int k, int n)
	    -> std::pair<double, double>;

	template <FluxDir DIR>
	static void ReconstructStatesConstant(amrex::MultiFab const &q, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int nghost, int nvars);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesConstant(arrayconst_t &q, array_t &leftState, array_t &rightState,
										       amrex::Box const &indexRange, int nvars);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
	ReconstructStatesConstant(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState,
				  quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in);

	template <FluxDir DIR, SlopeLimiter limiter>
	static void ReconstructStatesPLM(amrex::MultiFab const &q, amrex::MultiFab &leftState, amrex::MultiFab &rightState, int nghost, int nvars);

	template <FluxDir DIR, SlopeLimiter limiter>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPLM(arrayconst_t &q, array_t &leftState, array_t &rightState,
										  amrex::Box const &indexRange, int nvars);

	template <FluxDir DIR, SlopeLimiter limiter>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
	ReconstructStatesPLM(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState,
			     quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in);

	template <FluxDir DIR>
	static void ReconstructStatesPPM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf, int nghost, int nvars,
					 int iReadFrom = 0, int iWriteFrom = 0);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
										  amrex::Box const &cellRange, amrex::Box const &interfaceRange, int nvars,
										  int iReadFrom = 0, int iWriteFrom = 0);

	template <FluxDir DIR>
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void ReconstructStatesPPM(quokka::Array4View<amrex::Real const, DIR> const &q,
										  quokka::Array4View<amrex::Real, DIR> const &leftState,
										  quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in,
										  int j_in, int k_in, int iReadFrom = 0, int iWriteFrom = 0);

	template <typename F>
#if defined(__x86_64__)
	__attribute__((__target__("no-fma")))
#endif
	static void
	AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, double dt_in,
		     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars, F &&isStateValid,
		     amrex::Array4<int> const &redoFlag);

	template <typename F>
#if defined(__x86_64__)
	__attribute__((__target__("no-fma")))
#endif
	static void
	PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray, double dt_in,
		    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars, F &&isStateValid,
		    amrex::Array4<int> const &redoFlag);
};

template <typename problem_t>
template <FluxDir DIR>
void HyperbolicSystem<problem_t>::ReconstructStatesConstant(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf,
							    const int nghost, const int nvars)
{
	auto const &q_in = q_mf.const_arrays();
	auto leftState_in = leftState_mf.arrays();
	auto rightState_in = rightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> leftState(leftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> rightState(rightState_in[bx]);

		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(q, leftState, rightState, n, i_in, j_in, k_in);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesConstant(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
										  amrex::Box const &indexRange, const int nvars)
{
	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(q, leftState, rightState, n, i_in, j_in, k_in);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesConstant(quokka::Array4View<amrex::Real const, DIR> const &q,
										  quokka::Array4View<amrex::Real, DIR> const &leftState,
										  quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in,
										  int j_in, int k_in)
{
	// permute array indices according to dir
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	// Use piecewise-constant reconstruction (This converges at first order in spatial resolution.)
	// By convention, the interfaces are defined on the left edge of each zone, i.e. xleft_(i)
	// is the "left"-side of the interface at the left edge of zone i, and xright_(i) is the
	// "right"-side of the interface at the *left* edge of zone i. [Indexing note: There are (nx
	// + 1) interfaces for nx zones.]
	leftState(i, j, k, n) = q(i - 1, j, k, n);
	rightState(i, j, k, n) = q(i, j, k, n);
}

template <typename problem_t>
template <FluxDir DIR, SlopeLimiter limiter>
void HyperbolicSystem<problem_t>::ReconstructStatesPLM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf,
						       const int nghost, const int nvars)
{
	auto const &q_in = q_mf.const_arrays();
	auto leftState_in = leftState_mf.arrays();
	auto rightState_in = rightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> leftState(leftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> rightState(rightState_in[bx]);

		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, limiter>(q, leftState, rightState, n, i_in, j_in, k_in);
	});
}

template <typename problem_t>
template <FluxDir DIR, SlopeLimiter limiter>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPLM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
									     amrex::Box const &indexRange, const int nvars)
{
	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, limiter>(q, leftState, rightState, n, i_in, j_in, k_in);
	});
}

template <typename problem_t>
template <FluxDir DIR, SlopeLimiter limiter>
AMREX_GPU_HOST_DEVICE void
HyperbolicSystem<problem_t>::ReconstructStatesPLM(quokka::Array4View<amrex::Real const, DIR> const &q, quokka::Array4View<amrex::Real, DIR> const &leftState,
						  quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in, int k_in)
{
	// permute array indices according to dir
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	// Unlike PPM, PLM with MC or minmod limiters is TVD.
	// (There are no spurious oscillations, *except* in the slow-moving shock problem,
	// which can produce unphysical oscillations even when using upwind Godunov fluxes.)
	// However, most tests fail when using PLM reconstruction because
	// the accuracy tolerances are very strict, and the L1 error is significantly
	// worse compared to PPM for a fixed number of mesh elements.

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// Use piecewise-linear reconstruction
	// (This converges at second order in spatial resolution.)
	const auto lslope = HyperbolicSystem<problem_t>::template SlopeFunc<limiter>(q(i, j, k, n) - q(i - 1, j, k, n), q(i - 1, j, k, n) - q(i - 2, j, k, n));
	const auto rslope = HyperbolicSystem<problem_t>::template SlopeFunc<limiter>(q(i + 1, j, k, n) - q(i, j, k, n), q(i, j, k, n) - q(i - 1, j, k, n));
	leftState(i, j, k, n) = q(i - 1, j, k, n) + 0.25 * lslope; // NOLINT
	rightState(i, j, k, n) = q(i, j, k, n) - 0.25 * rslope;	   // NOLINT
}

template <typename problem_t>
AMREX_GPU_DEVICE auto HyperbolicSystem<problem_t>::GetMinmaxSurroundingCell(arrayconst_t &q, int i, int j, int k, int n) -> std::pair<double, double>
{
#if (AMREX_SPACEDIM == 1)
	// 1D: compute bounds from self + all 2 surrounding cells
	const std::pair<double, double> bounds = std::minmax({q(i, j, k, n), q(i - 1, j, k, n), q(i + 1, j, k, n)});

#elif (AMREX_SPACEDIM == 2)
	// 2D: compute bounds from self + all 8 surrounding cells
	const std::pair<double, double> bounds = std::minmax({q(i, j, k, n), q(i - 1, j, k, n), q(i + 1, j, k, n), q(i, j - 1, k, n), q(i, j + 1, k, n),
							      q(i - 1, j - 1, k, n), q(i + 1, j - 1, k, n), q(i - 1, j + 1, k, n), q(i + 1, j + 1, k, n)});

#else  // AMREX_SPACEDIM == 3
       // 3D: compute bounds from self + all 26 surrounding cells
	const std::pair<double, double> bounds = std::minmax({q(i, j, k, n),
							      q(i - 1, j, k, n),
							      q(i + 1, j, k, n),
							      q(i, j - 1, k, n),
							      q(i, j + 1, k, n),
							      q(i, j, k - 1, n),
							      q(i, j, k + 1, n),
							      q(i - 1, j - 1, k, n),
							      q(i + 1, j - 1, k, n),
							      q(i - 1, j + 1, k, n),
							      q(i + 1, j + 1, k, n),
							      q(i, j - 1, k - 1, n),
							      q(i, j + 1, k - 1, n),
							      q(i, j - 1, k + 1, n),
							      q(i, j + 1, k + 1, n),
							      q(i - 1, j, k - 1, n),
							      q(i + 1, j, k - 1, n),
							      q(i - 1, j, k + 1, n),
							      q(i + 1, j, k + 1, n),
							      q(i - 1, j - 1, k - 1, n),
							      q(i + 1, j - 1, k - 1, n),
							      q(i - 1, j - 1, k + 1, n),
							      q(i + 1, j - 1, k + 1, n),
							      q(i - 1, j + 1, k - 1, n),
							      q(i + 1, j + 1, k - 1, n),
							      q(i - 1, j + 1, k + 1, n),
							      q(i + 1, j + 1, k + 1, n)});
#endif // AMREX_SPACEDIM

	return bounds;
}

template <typename problem_t>
template <FluxDir DIR>
void HyperbolicSystem<problem_t>::ReconstructStatesPPM(amrex::MultiFab const &q_mf, amrex::MultiFab &leftState_mf, amrex::MultiFab &rightState_mf,
						       const int nghost, const int nvars, const int iReadFrom, const int iWriteFrom)
{
	const BL_PROFILE("HyperbolicSystem::ReconstructStatesPPM(MultiFabs)");

	auto const &q_in = q_mf.const_arrays();
	auto leftState_in = leftState_mf.arrays();
	auto rightState_in = rightState_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	// cell-centered kernel
	amrex::ParallelFor(q_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> q(q_in[bx]);
		quokka::Array4View<amrex::Real, DIR> leftState(leftState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> rightState(rightState_in[bx]);

		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(q, leftState, rightState, n, i_in, j_in, k_in, iReadFrom, iWriteFrom);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM(arrayconst_t &q_in, array_t &leftState_in, array_t &rightState_in,
									     amrex::Box const &cellRange, amrex::Box const & /*interfaceRange*/,
									     const int nvars, const int iReadFrom, const int iWriteFrom)
{
	const BL_PROFILE("HyperbolicSystem::ReconstructStatesPPM(Arrays)");

	// construct ArrayViews for permuted indices
	quokka::Array4View<amrex::Real const, DIR> q(q_in);
	quokka::Array4View<amrex::Real, DIR> leftState(leftState_in);
	quokka::Array4View<amrex::Real, DIR> rightState(rightState_in);

	// cell-centered kernel
	amrex::ParallelFor(cellRange, nvars, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in, int n) noexcept {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(q, leftState, rightState, n, i_in, j_in, k_in, iReadFrom, iWriteFrom);
	});
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_HOST_DEVICE void HyperbolicSystem<problem_t>::ReconstructStatesPPM(quokka::Array4View<amrex::Real const, DIR> const &q,
									     quokka::Array4View<amrex::Real, DIR> const &leftState,
									     quokka::Array4View<amrex::Real, DIR> const &rightState, int n, int i_in, int j_in,
									     int k_in, int iReadFrom, int iWriteFrom)
{
	// permute array indices according to dir
	auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at the left
	// edge of zone i, and xright_(i) is the "right"-side of the interface
	// at the *left* edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// (2.) Constrain interfaces to lie between surrounding cell-averaged
	// values (equivalent to step 2b in Athena++ [ppm_simple.cpp]).
	// [See Eq. B8 of Mignone+ 2005.]

#ifdef MULTIDIM_EXTREMA_CHECK
	// N.B.: Checking all 27 nearest neighbors is *very* expensive on GPU
	// (presumably due to lots of cache misses), so it is hard-coded disabled.
	// Fortunately, almost all problems run stably without enabling this.
	auto bounds = GetMinmaxSurroundingCell(q_in, i_in, j_in, k_in, iReadFrom + n);
#else
	// compute bounds from neighboring cell-averaged values along axis
	const std::pair<double, double> bounds = std::minmax({q(i, j, k, iReadFrom + n), q(i - 1, j, k, iReadFrom + n), q(i + 1, j, k, iReadFrom + n)});
#endif

	// get interfaces
	// PPM reconstruction following Colella & Woodward (1984), with
	// some modifications following Mignone (2014), as implemented in
	// Athena++.

	// (1.) Estimate the interface a_{i - 1/2}. Equivalent to step 1
	// in Athena++ [ppm_simple.cpp].

	// C&W Eq. (1.9) [parabola midpoint for the case of
	// equally-spaced zones]: a_{j+1/2} = (7/12)(a_j + a_{j+1}) -
	// (1/12)(a_{j+2} + a_{j-1}). Terms are grouped to preserve exact
	// symmetry in floating-point arithmetic, following Athena++.
	const double coef_1 = (7. / 12.);
	const double coef_2 = (-1. / 12.);
	const double a_minus = (coef_1 * q(i, j, k, iReadFrom + n) + coef_2 * q(i + 1, j, k, iReadFrom + n)) +
			       (coef_1 * q(i - 1, j, k, iReadFrom + n) + coef_2 * q(i - 2, j, k, iReadFrom + n));
	const double a_plus = (coef_1 * q(i + 1, j, k, iReadFrom + n) + coef_2 * q(i + 2, j, k, iReadFrom + n)) +
			      (coef_1 * q(i, j, k, iReadFrom + n) + coef_2 * q(i - 1, j, k, iReadFrom + n));

	// left side of zone i
	double new_a_minus = clamp(a_minus, bounds.first, bounds.second);

	// right side of zone i
	double new_a_plus = clamp(a_plus, bounds.first, bounds.second);

	// (3.) Monotonicity correction, using Eq. (1.10) in PPM paper. Equivalent
	// to step 4b in Athena++ [ppm_simple.cpp].

	const double a = q(i, j, k, iReadFrom + n); // a_i in C&W
	const double dq_minus = (a - new_a_minus);
	const double dq_plus = (new_a_plus - a);

	const double qa = dq_plus * dq_minus; // interface extrema

	if (qa <= 0.0) { // local extremum

		// Causes subtle, but very weird, oscillations in the Shu-Osher test
		// problem. However, it is necessary to get a reasonable solution
		// for the sawtooth advection problem.
		const double dq0 = MC(q(i + 1, j, k, iReadFrom + n) - q(i, j, k, iReadFrom + n), q(i, j, k, iReadFrom + n) - q(i - 1, j, k, iReadFrom + n));

		// use linear reconstruction, following Balsara (2017) [Living Rev
		// Comput Astrophys (2017) 3:2]
		new_a_minus = a - 0.5 * dq0;
		new_a_plus = a + 0.5 * dq0;

		// original C&W method for this case
		// new_a_minus = a;
		// new_a_plus = a;

	} else { // no local extrema

		// parabola overshoots near a_plus -> reset a_minus
		if (std::abs(dq_minus) >= 2.0 * std::abs(dq_plus)) {
			new_a_minus = a - 2.0 * dq_plus;
		}

		// parabola overshoots near a_minus -> reset a_plus
		if (std::abs(dq_plus) >= 2.0 * std::abs(dq_minus)) {
			new_a_plus = a + 2.0 * dq_minus;
		}
	}

	rightState(i, j, k, iWriteFrom + n) = new_a_minus;
	leftState(i + 1, j, k, iWriteFrom + n) = new_a_plus;
}

template <typename problem_t>
template <typename F>
void HyperbolicSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray,
					      const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
					      const int nvars, F &&isStateValid, amrex::Array4<int> const &redoFlag)
{
	BL_PROFILE("HyperbolicSystem::PredictStep()");

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	auto const dx = dx_in[0];
	auto const x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
	auto const dy = dx_in[1];
	auto const x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	auto const dz = dx_in[2];
	auto const x3Flux = fluxArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			consVarNew(i, j, k, n) = consVarOld(i, j, k, n) + (AMREX_D_TERM((dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
											+(dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
											+(dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))));
		}

		// check if state is valid -- flag for re-do if not
		if (!isStateValid(consVarNew, i, j, k)) {
			redoFlag(i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag(i, j, k) = quokka::redoFlag::none;
		}
	});
}

template <typename problem_t>
template <typename F>
void HyperbolicSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, std::array<arrayconst_t, AMREX_SPACEDIM> fluxArray,
					       const double dt_in, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange,
					       const int nvars, F &&isStateValid, amrex::Array4<int> const &redoFlag)
{
	BL_PROFILE("HyperbolicSystem::AddFluxesRK2()");

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	auto const dx = dx_in[0];
	auto const x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
	auto const dy = dx_in[1];
	auto const x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	auto const dz = dx_in[2];
	auto const x3Flux = fluxArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			// RK-SSP2 integrator
			const double U_0 = U0(i, j, k, n);
			const double U_1 = U1(i, j, k, n);

			const double FxU_1 = (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n));
#if (AMREX_SPACEDIM >= 2)
			const double FyU_1 = (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n));
#endif
#if (AMREX_SPACEDIM == 3)
			const double FzU_1 = (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n));
#endif

			// save results in U_new
			U_new(i, j, k, n) = (0.5 * U_0 + 0.5 * U_1) + (AMREX_D_TERM(0.5 * FxU_1, +0.5 * FyU_1, +0.5 * FzU_1));
		}

		// check if state is valid -- flag for re-do if not
		if (!isStateValid(U_new, i, j, k)) {
			redoFlag(i, j, k) = quokka::redoFlag::redo;
		} else {
			redoFlag(i, j, k) = quokka::redoFlag::none;
		}
	});
}

#endif // HYPERBOLIC_SYSTEM_HPP_
