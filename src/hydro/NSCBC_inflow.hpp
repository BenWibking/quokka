#ifndef NSCBC_INFLOW_HPP_ // NOLINT
#define NSCBC_INFLOW_HPP_
//==============================================================================
// Quokka -- two-moment radiation hydrodynamics on GPUs for astrophysics
// Copyright 2023 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file NSCBC_inflow.hpp
/// \brief Implements the Navier-Stokes Characteristic Boundary Condition for
/// subsonic, continuous inflow. This should NOT be used for a shock boundary
/// condition.

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "util/valarray.hpp"

namespace NSCBC
{
namespace detail
{
template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_inflow_x1_lower(quokka::valarray<Real, HydroSystem<problem_t>::nvar_> const &Q,
							       quokka::valarray<Real, HydroSystem<problem_t>::nvar_> const &dQ_dx_data, const Real T_t,
							       const Real u_t, const Real v_t, const Real w_t,
							       amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t, const Real L_x)
    -> quokka::valarray<Real, HydroSystem<problem_t>::nvar_>
{
	// return dQ/dx corresponding to subsonic inflow on the x1 lower boundary
	// (This is only necessary for continuous inflows, i.e., where a shock or contact discontinuity is not desired.)
	// NOTE: This boundary condition is only defined for an ideal gas (with constant k_B/mu).

	// TODO(bwibking): add transverse terms

	const Real rho = Q[0];
	const Real u = Q[1];
	const Real v = Q[2];
	const Real w = Q[3];
	const Real P = Q[4];
	const Real Eint_aux = Q[5];

	amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> s{};
	for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
		s[i] = Q[6 + i];
	}

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	amrex::GpuArray<Real, nmscalars_> massScalars;
	for (int n = 0; n < nmscalars_; ++n) {
		// indexing here follows primVars
		massScalars[n] = Q[HydroSystem<problem_t>::primScalar0_index + n];
	}

	const Real T = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, quokka::EOS<problem_t>::ComputeEintFromPres(rho, P, massScalars), massScalars);
	const Real Eint_aux_t = quokka::EOS<problem_t>::ComputeEintFromTgas(rho, T_t, massScalars);

	const Real du_dx = dQ_dx_data[1];
	const Real dP_dx = dQ_dx_data[4];

	const Real c = quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P, massScalars);
	const amrex::Real M = amrex::Clamp(std::sqrt(u * u + v * v + w * w) / c, 0., 1.);

	const Real eta_2 = 2.;
	const Real eta_3 = 2.;
	const Real eta_4 = 2.;
	const Real eta_5 = 2.;
	const Real eta_6 = 2.;

	const Real R = quokka::EOS<problem_t>::boltzmann_constant_ / quokka::EOS_Traits<problem_t>::mean_molecular_weight;

	// see SymPy notebook for derivation
	quokka::valarray<Real, HydroSystem<problem_t>::nvar_> dQ_dx{};
	if (u != 0.) {
		dQ_dx[0] = 0.5 *
			   (L_x * u * (c + u) * (-c * du_dx * rho + dP_dx) - 2 * R * c * eta_2 * rho * (c + u) * (T - T_t) -
			    std::pow(c, 2) * eta_5 * rho * u * (std::pow(M, 2) - 1) * (u - u_t)) /
			   (L_x * std::pow(c, 2) * u * (c + u));
		dQ_dx[1] = 0.5 * (L_x * (c + u) * (c * du_dx * rho - dP_dx) - std::pow(c, 2) * eta_5 * rho * (std::pow(M, 2) - 1) * (u - u_t)) /
			   (L_x * c * rho * (c + u));
		dQ_dx[2] = c * eta_3 * (v - v_t) / (L_x * u);
		dQ_dx[3] = c * eta_4 * (w - w_t) / (L_x * u);
		dQ_dx[4] =
		    0.5 * (L_x * (c + u) * (-c * du_dx * rho + dP_dx) - std::pow(c, 2) * eta_5 * rho * (std::pow(M, 2) - 1) * (u - u_t)) / (L_x * (c + u));
		dQ_dx[5] = c * eta_6 * (Eint_aux - Eint_aux_t) / (L_x * u);
		for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
			dQ_dx[6 + i] = c * eta_6 * (s[i] - s_t[i]) / (L_x * u);
		}
	} else { // u == 0
		dQ_dx[0] = 0.5 * (L_x * c * (-c * du_dx * rho + dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * std::pow(c, 3));
		dQ_dx[1] = 0.5 * (L_x * c * (c * du_dx * rho - dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * (c * c) * rho);
		dQ_dx[2] = 0;
		dQ_dx[3] = 0;
		dQ_dx[4] = 0.5 * (L_x * c * (-c * du_dx * rho + dP_dx) + (c * c) * eta_5 * rho * u_t * ((M * M) - 1)) / (L_x * c);
		dQ_dx[5] = 0;
		for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
			dQ_dx[6 + i] = 0;
		}
	}

	return dQ_dx;
}
} // namespace detail

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setInflowX1Lower(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, amrex::GeometryData const &geom,
							  const amrex::Real T_t, const amrex::Real u_t, const amrex::Real v_t, const amrex::Real w_t,
							  amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t)
{
	// x1 upper boundary -- subsonic outflow
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const int ilo = domain_lo[0];
	const Real dx = geom.CellSize(0);
	const Real Lx = geom.prob_domain.length(0);
	constexpr int N = HydroSystem<problem_t>::nvar_;

	/// x1 lower boundary -- subsonic inflow

	// compute one-sided dQ/dx from the data
	quokka::valarray<amrex::Real, N> const Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, ilo, j, k);
	quokka::valarray<amrex::Real, N> const Q_ip1 = HydroSystem<problem_t>::ComputePrimVars(consVar, ilo + 1, j, k);
	quokka::valarray<amrex::Real, N> const Q_ip2 = HydroSystem<problem_t>::ComputePrimVars(consVar, ilo + 2, j, k);
	quokka::valarray<amrex::Real, N> const dQ_dx_data = (-3. * Q_i + 4. * Q_ip1 - Q_ip2) / (2. * dx);

	// compute dQ/dx with modified characteristics
	quokka::valarray<amrex::Real, N> const dQ_dx = detail::dQ_dx_inflow_x1_lower<problem_t>(Q_i, dQ_dx_data, T_t, u_t, v_t, w_t, s_t, Lx);

	// compute centered ghost values
	quokka::valarray<amrex::Real, N> const Q_im1 = Q_ip1 - 2.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> const Q_im2 = -2.0 * Q_ip1 - 3.0 * Q_i + 6.0 * Q_im1 + 6.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> const Q_im3 = 3.0 * Q_ip1 + 10.0 * Q_i - 18.0 * Q_im1 + 6.0 * Q_im2 - 12.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> const Q_im4 = -2.0 * Q_ip1 - 13.0 * Q_i + 24.0 * Q_im1 - 12.0 * Q_im2 + 4.0 * Q_im3 + 12.0 * dx * dQ_dx;

	// set cell values
	quokka::valarray<amrex::Real, N> consCell{};
	if (i == ilo - 1) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_im1);
	} else if (i == ilo - 2) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_im2);
	} else if (i == ilo - 3) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_im3);
	} else if (i == ilo - 4) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_im4);
	}

	consVar(i, j, k, HydroSystem<problem_t>::density_index) = consCell[0];
	consVar(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = consCell[1];
	consVar(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = consCell[2];
	consVar(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = consCell[3];
	consVar(i, j, k, HydroSystem<problem_t>::energy_index) = consCell[4];
	consVar(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = consCell[5];
	for (int n = 0; n < HydroSystem<problem_t>::nscalars_; ++n) {
		consVar(i, j, k, HydroSystem<problem_t>::scalar0_index + n) = consCell[6 + n];
	}
}

template <typename problem_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setInflowX1LowerLowOrder(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
								  amrex::GeometryData const &geom, const amrex::Real T_t, const amrex::Real u_t,
								  const amrex::Real v_t, const amrex::Real w_t,
								  amrex::GpuArray<Real, HydroSystem<problem_t>::nscalars_> const &s_t)
{
	// x1 upper boundary -- subsonic outflow
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const int ilo = domain_lo[0];
	constexpr int N = HydroSystem<problem_t>::nvar_;

	/// x1 lower boundary -- subsonic inflow

	// compute primitive vars from valid region
	quokka::valarray<amrex::Real, N> const Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, ilo, j, k);

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	amrex::GpuArray<Real, nmscalars_> massScalars;
	for (int n = 0; n < nmscalars_; ++n) {
		// indexing here follows primVars
		massScalars[n] = Q_i[HydroSystem<problem_t>::primScalar0_index + n];
	}

	// compute centered ghost values
	const Real rho = Q_i[0];
	const Real Eint = quokka::EOS<problem_t>::ComputeEintFromTgas(rho, T_t, massScalars);
	quokka::valarray<amrex::Real, N> Q_im1{};
	Q_im1[0] = rho; // extrapolate density
	Q_im1[1] = u_t; // prescribe velocity
	Q_im1[2] = v_t;
	Q_im1[3] = w_t;
	Q_im1[4] = quokka::EOS<problem_t>::ComputePressure(rho, Eint, massScalars); // prescribe temperature
	Q_im1[5] = Eint;							    // prescribe temperature
	for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
		Q_im1[6 + i] = s_t[i]; // prescribe passive scalars
	}

	// set cell values
	quokka::valarray<amrex::Real, N> consCell{};
	consCell = HydroSystem<problem_t>::ComputeConsVars(Q_im1);

	consVar(i, j, k, HydroSystem<problem_t>::density_index) = consCell[0];
	consVar(i, j, k, HydroSystem<problem_t>::x1Momentum_index) = consCell[1];
	consVar(i, j, k, HydroSystem<problem_t>::x2Momentum_index) = consCell[2];
	consVar(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = consCell[3];
	consVar(i, j, k, HydroSystem<problem_t>::energy_index) = consCell[4];
	consVar(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = consCell[5];
	for (int n = 0; n < HydroSystem<problem_t>::nscalars_; ++n) {
		consVar(i, j, k, HydroSystem<problem_t>::scalar0_index + n) = consCell[6 + n];
	}
}
} // namespace NSCBC

#endif // NSCBC_INFLOW_HPP_