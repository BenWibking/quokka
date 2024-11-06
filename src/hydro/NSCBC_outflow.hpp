#ifndef NSCBC_OUTFLOW_HPP_ // NOLINT
#define NSCBC_OUTFLOW_HPP_
//==============================================================================
// Quokka -- two-moment radiation hydrodynamics on GPUs for astrophysics
// Copyright 2023 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file NSCBC_outflow.hpp
/// \brief Implements the Navier-Stokes Characteristic Boundary Condition for
/// subsonic outflows. (This also works trivially for _super_sonic outflows.)

#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "util/valarray.hpp"

namespace NSCBC
{
enum class BoundarySide { Lower, Upper };
namespace detail
{
template <typename problem_t, BoundarySide SIDE>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto dQ_dx_outflow(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q,
						       quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dx_data,
						       quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dy_data,
						       quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &dQ_dz_data, const amrex::Real P_t,
						       const amrex::Real L_x) -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>
{
	// return dQ/dx corresponding to subsonic outflow at the (SIDE) boundary

	const amrex::Real rho = Q[0];
	const amrex::Real u = Q[1];
	const amrex::Real v = Q[2];
	const amrex::Real w = Q[3];
	const amrex::Real P = Q[4];

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	amrex::GpuArray<Real, nmscalars_> massScalars;
	for (int n = 0; n < nmscalars_; ++n) {
		// indexing here follows primVars
		massScalars[n] = Q[HydroSystem<problem_t>::primScalar0_index + n];
	}

	// normal derivatives
	const amrex::Real drho_dx = dQ_dx_data[0];
	const amrex::Real du_dx = dQ_dx_data[1];
	const amrex::Real dv_dx = dQ_dx_data[2];
	const amrex::Real dw_dx = dQ_dx_data[3];
	const amrex::Real dP_dx = dQ_dx_data[4];
	const amrex::Real dEint_aux_dx = dQ_dx_data[5];

	// x2 transverse derivatives
	const amrex::Real du_dy = dQ_dy_data[1];
	const amrex::Real dv_dy = dQ_dy_data[2];
	const amrex::Real dP_dy = dQ_dy_data[4];

	// x3 transverse derivatives
	const amrex::Real du_dz = dQ_dz_data[1];
	const amrex::Real dw_dz = dQ_dz_data[3];
	const amrex::Real dP_dz = dQ_dz_data[4];

	const amrex::Real c = quokka::EOS<problem_t>::ComputeSoundSpeed(rho, P, massScalars);
	const amrex::Real M = std::clamp(std::sqrt(u * u + v * v + w * w) / c, 0., 1.);
	const amrex::Real beta = M;
	const amrex::Real K = 0.25 * c * (1 - M * M) / L_x; // must be non-zero for well-posedness

	// see SymPy notebook for derivation of dQ_dx
	quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> dQ_dx{};

	const amrex::Real x0 = std::pow(c, 2);
	const amrex::Real x2 = c * rho;
	const amrex::Real x3 = du_dx * x2;
	const amrex::Real x4 = rho * x0;

	if (SIDE == BoundarySide::Upper) {
		const amrex::Real x1 = c - u;
		const amrex::Real x5 = K * (P - P_t) + (beta - 1) * (dP_dy * v + dP_dz * w - du_dy * v * x2 - du_dz * w * x2 + dv_dy * x4 + dw_dz * x4);
		const amrex::Real x6 = (1.0 / 2.0) / x1;
		const amrex::Real x7 = dP_dx + x3;

		dQ_dx[0] = x6 * (x1 * (-dP_dx + 2 * drho_dx * x0 + x3) - x5) / x0;
		dQ_dx[1] = x6 * (x1 * x7 + x5) / (c * rho);
		dQ_dx[4] = x6 * (x1 * x7 - x5);

	} else if (SIDE == BoundarySide::Lower) {
		const amrex::Real x1 = c + u;
		const amrex::Real x5 = K * (P - P_t) + (beta - 1) * (dP_dy * v + dP_dz * w + du_dy * v * x2 + du_dz * w * x2 + dv_dy * x4 + dw_dz * x4);
		const amrex::Real x6 = (1.0 / 2.0) / x1;
		const amrex::Real x7 = -dP_dx + x3;

		dQ_dx[0] = x6 * (x1 * (-dP_dx + 2 * drho_dx * x0 - x3) + x5) / x0;
		dQ_dx[1] = x6 * (x1 * x7 + x5) / (c * rho);
		dQ_dx[4] = x6 * (-x1 * x7 + x5);
	}

	dQ_dx[2] = dv_dx;
	dQ_dx[3] = dw_dx;
	dQ_dx[5] = dEint_aux_dx;

	for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
		dQ_dx[6 + i] = dQ_dx_data[6 + i];
	}

	return dQ_dx;
}

template <typename problem_t, BoundarySide SIDE>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_xdir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
								 amrex::GeometryData const &geom)
    -> std::tuple<quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>>
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	constexpr int N = HydroSystem<problem_t>::nvar_;
	const auto &boundary_idx = (SIDE == BoundarySide::Lower) ? box.loVect3d() : box.hiVect3d();
	const int ibr = boundary_idx[0];

	quokka::valarray<amrex::Real, N> dQ_dy_data{};
	quokka::valarray<amrex::Real, N> dQ_dz_data{};

	// dQ/dy
	if constexpr (AMREX_SPACEDIM >= 2) {
		if (consVar.contains(ibr, j + 1, k) && consVar.contains(ibr, j - 1, k)) {
			quokka::valarray<amrex::Real, N> const Qp = HydroSystem<problem_t>::ComputePrimVars(consVar, ibr, j + 1, k);
			quokka::valarray<amrex::Real, N> const Qm = HydroSystem<problem_t>::ComputePrimVars(consVar, ibr, j - 1, k);
			dQ_dy_data = (Qp - Qm) / (2.0 * geom.CellSize(1));
		}
	}
	// dQ/dz
	if constexpr (AMREX_SPACEDIM == 3) {
		if (consVar.contains(ibr, j, k + 1) && consVar.contains(ibr, j, k - 1)) {
			quokka::valarray<amrex::Real, N> const Qp = HydroSystem<problem_t>::ComputePrimVars(consVar, ibr, j, k + 1);
			quokka::valarray<amrex::Real, N> const Qm = HydroSystem<problem_t>::ComputePrimVars(consVar, ibr, j, k - 1);
			dQ_dy_data = (Qp - Qm) / (2.0 * geom.CellSize(2));
		}
	}

	return std::make_tuple(dQ_dy_data, dQ_dz_data);
}

template <typename problem_t, BoundarySide SIDE>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_ydir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
								 amrex::GeometryData const &geom)
    -> std::tuple<quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>>
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	constexpr int N = HydroSystem<problem_t>::nvar_;
	const auto &boundary_idx = (SIDE == BoundarySide::Lower) ? box.loVect3d() : box.hiVect3d();
	const int jbr = boundary_idx[1];

	quokka::valarray<amrex::Real, N> dQ_dz_data{};
	quokka::valarray<amrex::Real, N> dQ_dx_data{};

	// dQ/dz
	if constexpr (AMREX_SPACEDIM == 3) {
		if (consVar.contains(i, jbr, k + 1) && consVar.contains(i, jbr, k - 1)) {
			quokka::valarray<amrex::Real, N> const Qp = HydroSystem<problem_t>::ComputePrimVars(consVar, i, jbr, k + 1);
			quokka::valarray<amrex::Real, N> const Qm = HydroSystem<problem_t>::ComputePrimVars(consVar, i, jbr, k - 1);
			dQ_dz_data = (Qp - Qm) / (2.0 * geom.CellSize(2));
		}
	}
	// dQ/dx
	{
		if (consVar.contains(i + 1, jbr, k) && consVar.contains(i - 1, jbr, k)) {
			quokka::valarray<amrex::Real, N> const Qp = HydroSystem<problem_t>::ComputePrimVars(consVar, i + 1, jbr, k);
			quokka::valarray<amrex::Real, N> const Qm = HydroSystem<problem_t>::ComputePrimVars(consVar, i - 1, jbr, k);
			dQ_dx_data = (Qp - Qm) / (2.0 * geom.CellSize(0));
		}
	}
	return std::make_tuple(dQ_dz_data, dQ_dx_data);
}

template <typename problem_t, BoundarySide SIDE>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto transverse_zdir_dQ_data(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
								 amrex::GeometryData const &geom)
    -> std::tuple<quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>, quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>>
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	constexpr int N = HydroSystem<problem_t>::nvar_;
	const auto &boundary_idx = (SIDE == BoundarySide::Lower) ? box.loVect3d() : box.hiVect3d();
	const int kbr = boundary_idx[2];

	quokka::valarray<amrex::Real, N> dQ_dx_data{};
	quokka::valarray<amrex::Real, N> dQ_dy_data{};

	// dQ/dx
	{
		if (consVar.contains(i + 1, j, kbr) && consVar.contains(i - 1, j, kbr)) {
			quokka::valarray<amrex::Real, N> const Qp = HydroSystem<problem_t>::ComputePrimVars(consVar, i + 1, j, kbr);
			quokka::valarray<amrex::Real, N> const Qm = HydroSystem<problem_t>::ComputePrimVars(consVar, i - 1, j, kbr);
			dQ_dx_data = (Qp - Qm) / (2.0 * geom.CellSize(0));
		}
	}
	// dQ/dy
	{
		if (consVar.contains(i, j + 1, kbr) && consVar.contains(i, j - 1, kbr)) {
			quokka::valarray<amrex::Real, N> const Qp = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j + 1, kbr);
			quokka::valarray<amrex::Real, N> const Qm = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j - 1, kbr);
			dQ_dy_data = (Qp - Qm) / (2.0 * geom.CellSize(1));
		}
	}
	return std::make_tuple(dQ_dx_data, dQ_dy_data);
}

template <typename problem_t, FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto permute_vel(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q)
    -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>
{
	// with normal direction DIR, permutes the velocity components so that
	//  u, v, w are the normal and transverse components, respectively.
	const amrex::Real u = Q[1];
	const amrex::Real v = Q[2];
	const amrex::Real w = Q[3];

	quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> newPrim{};
	if constexpr (DIR == FluxDir::X1) {
		newPrim = {Q[0], u, v, w, Q[4], Q[5]};
	} else if constexpr (DIR == FluxDir::X2) {
		newPrim = {Q[0], v, w, u, Q[4], Q[5]};
	} else if constexpr (DIR == FluxDir::X3) {
		newPrim = {Q[0], w, u, v, Q[4], Q[5]};
	}

	// copy passive scalars
	for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
		newPrim[6 + i] = Q[6 + i];
	}
	return newPrim;
}

template <typename problem_t, FluxDir DIR>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto unpermute_vel(quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> const &Q)
    -> quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_>
{
	// with normal direction DIR, un-permutes the velocity components so that
	//  u, v, w are the normal and transverse components *prior to calling permute_vel*.
	const amrex::Real v1 = Q[1];
	const amrex::Real v2 = Q[2];
	const amrex::Real v3 = Q[3];

	quokka::valarray<amrex::Real, HydroSystem<problem_t>::nvar_> newPrim{};
	if constexpr (DIR == FluxDir::X1) {
		newPrim = {Q[0], v1, v2, v3, Q[4], Q[5]};
	} else if constexpr (DIR == FluxDir::X2) {
		newPrim = {Q[0], v3, v1, v2, Q[4], Q[5]};
	} else if constexpr (DIR == FluxDir::X3) {
		newPrim = {Q[0], v2, v3, v1, Q[4], Q[5]};
	}

	// copy passive scalars
	for (int i = 0; i < HydroSystem<problem_t>::nscalars_; ++i) {
		newPrim[6 + i] = Q[6 + i];
	}
	return newPrim;
}
} // namespace detail

template <typename problem_t, FluxDir DIR, BoundarySide SIDE>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setOutflowBoundary(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
							    amrex::GeometryData const &geom, const amrex::Real P_outflow)
{
	// subsonic outflow on the DIR SIDE boundary

	auto [i, j, k] = iv.dim3();
	std::array<int, 3> idx{i, j, k};
	amrex::Box const &box = geom.Domain();
	constexpr int N = HydroSystem<problem_t>::nvar_;

	const auto &boundary_idx = (SIDE == BoundarySide::Lower) ? box.loVect3d() : box.hiVect3d();
	const int ibr = boundary_idx[static_cast<int>(DIR)];
	const int im1 = (SIDE == BoundarySide::Lower) ? ibr + 1 : ibr - 1;
	const int im2 = (SIDE == BoundarySide::Lower) ? ibr + 2 : ibr - 2;
	const Real dx = geom.CellSize(static_cast<int>(DIR));

	// compute one-sided dQ/dx_n from the data
	quokka::valarray<amrex::Real, N> Q_i{};
	quokka::valarray<amrex::Real, N> Q_im1{};
	quokka::valarray<amrex::Real, N> Q_im2{};

	if constexpr (DIR == FluxDir::X1) {
		Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, ibr, j, k);
		Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, im1, j, k);
		Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, im2, j, k);
	} else if constexpr (DIR == FluxDir::X2) {
		Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, i, ibr, k);
		Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, im1, k);
		Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, im2, k);
	} else if constexpr (DIR == FluxDir::X3) {
		Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, ibr);
		Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im1);
		Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im2);
	}

	quokka::valarray<amrex::Real, N> dQ_dx_data = (Q_im2 - 4.0 * Q_im1 + 3.0 * Q_i) / (2.0 * dx);
	dQ_dx_data *= (SIDE == BoundarySide::Lower) ? -1.0 : 1.0;

	// compute dQ/dx with modified characteristics
	quokka::valarray<amrex::Real, N> dQ_dt1_data{};
	quokka::valarray<amrex::Real, N> dQ_dt2_data{};
	if constexpr (DIR == FluxDir::X1) {
		auto [dQ_dt1, dQ_dt2] = detail::transverse_xdir_dQ_data<problem_t, SIDE>(iv, consVar, geom);
		dQ_dt1_data = dQ_dt1;
		dQ_dt2_data = dQ_dt2;
	} else if constexpr (DIR == FluxDir::X2) {
		auto [dQ_dt1, dQ_dt2] = detail::transverse_ydir_dQ_data<problem_t, SIDE>(iv, consVar, geom);
		dQ_dt1_data = dQ_dt1;
		dQ_dt2_data = dQ_dt2;
	} else if constexpr (DIR == FluxDir::X3) {
		auto [dQ_dt1, dQ_dt2] = detail::transverse_zdir_dQ_data<problem_t, SIDE>(iv, consVar, geom);
		dQ_dt1_data = dQ_dt1;
		dQ_dt2_data = dQ_dt2;
	}

	const Real Lbox = geom.prob_domain.length(static_cast<int>(DIR));
	quokka::valarray<amrex::Real, N> dQ_dx = detail::unpermute_vel<problem_t, DIR>(detail::dQ_dx_outflow<problem_t, SIDE>(
	    detail::permute_vel<problem_t, DIR>(Q_i), detail::permute_vel<problem_t, DIR>(dQ_dx_data), detail::permute_vel<problem_t, DIR>(dQ_dt1_data),
	    detail::permute_vel<problem_t, DIR>(dQ_dt2_data), P_outflow, Lbox));

	// compute centered ghost values
	dQ_dx *= (SIDE == BoundarySide::Lower) ? -1.0 : 1.0;
	quokka::valarray<amrex::Real, N> Q_ip1 = Q_im1 + 2.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> Q_ip2 = -2.0 * Q_im1 - 3.0 * Q_i + 6.0 * Q_ip1 - 6.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> Q_ip3 = 3.0 * Q_im1 + 10.0 * Q_i - 18.0 * Q_ip1 + 6.0 * Q_ip2 + 12.0 * dx * dQ_dx;
	quokka::valarray<amrex::Real, N> Q_ip4 = -2.0 * Q_im1 - 13.0 * Q_i + 24.0 * Q_ip1 - 12.0 * Q_ip2 + 4.0 * Q_ip3 - 12.0 * dx * dQ_dx;

	// set cell values
	const int ip1 = (SIDE == BoundarySide::Lower) ? ibr - 1 : ibr + 1;
	const int ip2 = (SIDE == BoundarySide::Lower) ? ibr - 2 : ibr + 2;
	const int ip3 = (SIDE == BoundarySide::Lower) ? ibr - 3 : ibr + 3;
	const int ip4 = (SIDE == BoundarySide::Lower) ? ibr - 4 : ibr + 4;

	quokka::valarray<amrex::Real, N> consCell{};
	if (idx[static_cast<int>(DIR)] == ip1) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip1);
	} else if (idx[static_cast<int>(DIR)] == ip2) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip2);
	} else if (idx[static_cast<int>(DIR)] == ip3) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip3);
	} else if (idx[static_cast<int>(DIR)] == ip4) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip4);
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

template <typename problem_t, FluxDir DIR, BoundarySide SIDE>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void setOutflowBoundaryLowOrder(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar,
								    amrex::GeometryData const &geom, const amrex::Real P_outflow)
{
	// subsonic outflow on the DIR SIDE boundary

	auto [i, j, k] = iv.dim3();
	std::array<int, 3> idx{i, j, k};
	amrex::Box const &box = geom.Domain();
	constexpr int N = HydroSystem<problem_t>::nvar_;

	const auto &boundary_idx = (SIDE == BoundarySide::Lower) ? box.loVect3d() : box.hiVect3d();
	const int ibr = boundary_idx[static_cast<int>(DIR)];
	const int im1 = (SIDE == BoundarySide::Lower) ? ibr + 1 : ibr - 1;
	const int im2 = (SIDE == BoundarySide::Lower) ? ibr + 2 : ibr - 2;
	const int im3 = (SIDE == BoundarySide::Lower) ? ibr + 3 : ibr - 3;

	// compute primitive vars
	quokka::valarray<amrex::Real, N> Q_i{};
	if constexpr (DIR == FluxDir::X1) {
		Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, ibr, j, k);
	} else if constexpr (DIR == FluxDir::X2) {
		Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, i, ibr, k);
	} else if constexpr (DIR == FluxDir::X3) {
		Q_i = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, ibr);
	}

	quokka::valarray<amrex::Real, N> Q_ip1{};
	quokka::valarray<amrex::Real, N> Q_ip2{};
	quokka::valarray<amrex::Real, N> Q_ip3{};
	quokka::valarray<amrex::Real, N> Q_ip4{};

	// if gas is inflowing, change to reflecting B.C.
	quokka::valarray<amrex::Real, N> Qr_im0 = detail::permute_vel<problem_t, DIR>(Q_i);
	amrex::Real const v_normal = Qr_im0[1]; // normal velocity component

	if (((SIDE == BoundarySide::Lower) && (v_normal > 0.)) || ((SIDE == BoundarySide::Upper) && (v_normal < 0.))) {
		// compute primitive vars
		quokka::valarray<amrex::Real, N> Q_im1{};
		quokka::valarray<amrex::Real, N> Q_im2{};
		quokka::valarray<amrex::Real, N> Q_im3{};

		if constexpr (DIR == FluxDir::X1) {
			Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, im1, j, k);
			Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, im2, j, k);
			Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, im3, j, k);
		} else if constexpr (DIR == FluxDir::X2) {
			Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, im1, k);
			Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, im2, k);
			Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, im3, k);
		} else if constexpr (DIR == FluxDir::X3) {
			Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im1);
			Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im2);
			Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im3);
		}

		// reflect velocities
		quokka::valarray<amrex::Real, N> Qr_im1 = detail::permute_vel<problem_t, DIR>(Q_im1);
		quokka::valarray<amrex::Real, N> Qr_im2 = detail::permute_vel<problem_t, DIR>(Q_im2);
		quokka::valarray<amrex::Real, N> Qr_im3 = detail::permute_vel<problem_t, DIR>(Q_im3);

		Qr_im0[1] *= -1.0;
		Qr_im1[1] *= -1.0;
		Qr_im2[1] *= -1.0;
		Qr_im3[1] *= -1.0;

		Q_ip1 = detail::unpermute_vel<problem_t, DIR>(Qr_im0);
		Q_ip2 = detail::unpermute_vel<problem_t, DIR>(Qr_im1);
		Q_ip3 = detail::unpermute_vel<problem_t, DIR>(Qr_im2);
		Q_ip4 = detail::unpermute_vel<problem_t, DIR>(Qr_im3);
	} else {
		// specify pressure at boundary
		Q_i[4] = P_outflow;
		Q_ip1 = Q_i;
		Q_ip2 = Q_i;
		Q_ip3 = Q_i;
		Q_ip4 = Q_i;
	}

	// set cell values
	const int ip1 = (SIDE == BoundarySide::Lower) ? ibr - 1 : ibr + 1;
	const int ip2 = (SIDE == BoundarySide::Lower) ? ibr - 2 : ibr + 2;
	const int ip3 = (SIDE == BoundarySide::Lower) ? ibr - 3 : ibr + 3;
	const int ip4 = (SIDE == BoundarySide::Lower) ? ibr - 4 : ibr + 4;

	quokka::valarray<amrex::Real, N> consCell{};
	if (idx[static_cast<int>(DIR)] == ip1) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip1);
	} else if (idx[static_cast<int>(DIR)] == ip2) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip2);
	} else if (idx[static_cast<int>(DIR)] == ip3) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip3);
	} else if (idx[static_cast<int>(DIR)] == ip4) {
		consCell = HydroSystem<problem_t>::ComputeConsVars(Q_ip4);
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
} // namespace NSCBC

#endif // NSCBC_OUTFLOW_HPP_
