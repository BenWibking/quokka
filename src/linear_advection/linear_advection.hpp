#ifndef LINEAR_ADVECTION_HPP_ // NOLINT
#define LINEAR_ADVECTION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file linear_advection.hpp
/// \brief Defines a class for solving a scalar linear advection equation.
///

// c++ headers
#include <cmath>

// library headers

// internal headers
#include "hyperbolic_system.hpp"

/// Class for a linear, scalar advection equation
///
template <typename problem_t> class LinearAdvectionSystem : public HyperbolicSystem<problem_t>
{
      public:
	enum varIndex { density_index = 0 };

	// static member functions

	static void ConservedToPrimitive(amrex::MultiFab const &cons_mf, amrex::MultiFab &primVar_mf, int nghost, int nvars);

	static void ComputeMaxSignalSpeed(amrex::Array4<amrex::Real const> const & /*cons*/, amrex::Array4<amrex::Real> const &maxSignal, double advectionVx,
					  double advectionVy, double advectionVz, amrex::Box const &indexRange);

	AMREX_GPU_DEVICE
	static auto isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k) -> bool;

	static void PredictStep(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf,
				std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
				int nvars);

	static void AddFluxesRK2(amrex::MultiFab &U_new_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf,
				 std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, double dt, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in,
				 int nvars);

	template <FluxDir DIR>
	static void ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab const &x1LeftState_mf, amrex::MultiFab const &x1RightState_mf, double advectionVx,
				  int nvars);
};

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<amrex::Real const> const & /*cons*/, amrex::Array4<amrex::Real> const &maxSignal,
							     const double advectionVx, const double advectionVy, const double advectionVz,
							     amrex::Box const &indexRange)
{
	const auto vx = advectionVx;
	const auto vy = advectionVy;
	const auto vz = advectionVz;
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double signal_max = std::sqrt(vx * vx + vy * vy + vz * vz);
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::ConservedToPrimitive(amrex::MultiFab const &cons_mf, amrex::MultiFab &primVar_mf, const int nghost, const int nvars)
{
	auto const &cons = cons_mf.const_arrays();
	auto primVar = primVar_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(nghost, nghost, nghost)};

	amrex::ParallelFor(primVar_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) { primVar[bx](i, j, k, n) = cons[bx](i, j, k, n); });
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto LinearAdvectionSystem<problem_t>::isStateValid(amrex::Array4<const amrex::Real> const &cons, int i, int j, int k)
    -> bool
{
	// check if cons(i, j, k) is a valid state
	const auto rho = cons(i, j, k, density_index);
	bool isDensityPositive = (rho > 0.);
	return isDensityPositive;
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::PredictStep(amrex::MultiFab const &consVarOld_mf, amrex::MultiFab &consVarNew_mf,
						   std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, const double dt,
						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, const int nvars)
{
	BL_PROFILE("LinearAdvectionSystem::PredictStep()");

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dx = dx_in[0];
	auto const x1Flux = fluxArray[0].const_arrays();
#if (AMREX_SPACEDIM >= 2)
	auto const dy = dx_in[1];
	auto const x2Flux = fluxArray[1].const_arrays();
#endif
#if (AMREX_SPACEDIM == 3)
	auto const dz = dx_in[2];
	auto const x3Flux = fluxArray[2].const_arrays();
#endif
	auto const &consVarOld = consVarOld_mf.const_arrays();
	auto consVarNew = consVarNew_mf.arrays();

	amrex::ParallelFor(consVarNew_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			consVarNew[bx](i, j, k, n) =
			    consVarOld[bx](i, j, k, n) + (AMREX_D_TERM((dt / dx) * (x1Flux[bx](i, j, k, n) - x1Flux[bx](i + 1, j, k, n)),
								       +(dt / dy) * (x2Flux[bx](i, j, k, n) - x2Flux[bx](i, j + 1, k, n)),
								       +(dt / dz) * (x3Flux[bx](i, j, k, n) - x3Flux[bx](i, j, k + 1, n))));
		}
	});
}

template <typename problem_t>
void LinearAdvectionSystem<problem_t>::AddFluxesRK2(amrex::MultiFab &U_new_mf, amrex::MultiFab const &U0_mf, amrex::MultiFab const &U1_mf,
						    std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fluxArray, const double dt,
						    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, const int nvars)
{
	BL_PROFILE("LinearAdvectionSystem::AddFluxesRK2()");

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dx = dx_in[0];
	auto const x1Flux = fluxArray[0].const_arrays();
#if (AMREX_SPACEDIM >= 2)
	auto const dy = dx_in[1];
	auto const x2Flux = fluxArray[1].const_arrays();
#endif
#if (AMREX_SPACEDIM == 3)
	auto const dz = dx_in[2];
	auto const x3Flux = fluxArray[2].const_arrays();
#endif
	auto const &U0 = U0_mf.const_arrays();
	auto const &U1 = U1_mf.const_arrays();
	auto U_new = U_new_mf.arrays();

	amrex::ParallelFor(U_new_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		for (int n = 0; n < nvars; ++n) {
			// RK-SSP2 integrator
			const double U_0 = U0[bx](i, j, k, n);
			const double U_1 = U1[bx](i, j, k, n);

			const double FxU_1 = (dt / dx) * (x1Flux[bx](i, j, k, n) - x1Flux[bx](i + 1, j, k, n));
#if (AMREX_SPACEDIM >= 2)
			const double FyU_1 = (dt / dy) * (x2Flux[bx](i, j, k, n) - x2Flux[bx](i, j + 1, k, n));
#endif
#if (AMREX_SPACEDIM == 3)
			const double FzU_1 = (dt / dz) * (x3Flux[bx](i, j, k, n) - x3Flux[bx](i, j, k + 1, n));
#endif

			// save results in U_new
			U_new[bx](i, j, k, n) = (0.5 * U_0 + 0.5 * U_1) + (AMREX_D_TERM(0.5 * FxU_1, +0.5 * FyU_1, +0.5 * FzU_1));
		}
	});
}

template <typename problem_t>
template <FluxDir DIR>
void LinearAdvectionSystem<problem_t>::ComputeFluxes(amrex::MultiFab &x1Flux_mf, amrex::MultiFab const &x1LeftState_mf, amrex::MultiFab const &x1RightState_mf,
						     const double vx, const int nvars)
{
	// By convention, the interfaces are defined on the left edge of each zone, i.e.
	// xinterface_(i) is the solution to the Riemann problem at the left edge of zone i.
	// [Indexing note: There are (nx + 1) interfaces for nx zones.]

	auto const &x1LeftState_in = x1LeftState_mf.const_arrays();
	auto const &x1RightState_in = x1RightState_mf.const_arrays();
	auto x1Flux_in = x1Flux_mf.arrays();
	amrex::IntVect ng{AMREX_D_DECL(0, 0, 0)};

	amrex::ParallelFor(x1Flux_mf, ng, nvars, [=] AMREX_GPU_DEVICE(int bx, int i_in, int j_in, int k_in, int n) noexcept {
		// construct ArrayViews for permuted indices
		quokka::Array4View<amrex::Real const, DIR> x1LeftState(x1LeftState_in[bx]);
		quokka::Array4View<amrex::Real const, DIR> x1RightState(x1RightState_in[bx]);
		quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in[bx]);

		// permute array indices according to dir
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// For advection, simply choose upwind side of the interface.
		if (vx < 0.0) { // upwind switch
			// upwind direction is the right-side of the interface
			x1Flux(i, j, k, n) = vx * x1RightState(i, j, k, n);

		} else {
			// upwind direction is the left-side of the interface
			x1Flux(i, j, k, n) = vx * x1LeftState(i, j, k, n);
		}
	});
}

#endif // LINEAR_ADVECTION_HPP_
