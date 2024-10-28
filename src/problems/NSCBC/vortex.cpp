//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file vortex.cpp
/// \brief Implements a subsonic vortex flow problem with Navier-Stokes
///        Characteristic Boundary Conditions (NSCBC).
///
#include <random>
#include <tuple>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"

#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/HydroState.hpp"
#include "hydro/NSCBC_inflow.hpp"
#include "hydro/NSCBC_outflow.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayUtil.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"
#include "vortex.hpp"

using amrex::Real;

struct Vortex {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct quokka::EOS_Traits<Vortex> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = 28.96 * C::m_u; // air
};

template <> struct Physics_Traits<Vortex> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// global variables needed for Dirichlet boundary condition and initial conditions
namespace
{
const bool outflow_boundary_along_x_axis = true;
amrex::Real G_vortex = NAN;						      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real T_ref = NAN;				      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real P_ref = NAN;				      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real u0 = NAN;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real v0 = NAN;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real w0 = NAN;					      // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::GpuArray<Real, HydroSystem<Vortex>::nscalars_> s0{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
}; // namespace

template <> void QuokkaSimulation<Vortex>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;

	constexpr Real gamma = quokka::EOS_Traits<Vortex>::gamma;
	constexpr Real R = C::k_B / quokka::EOS_Traits<Vortex>::mean_molecular_weight;
	const Real c = std::sqrt(gamma * R * T_ref);

	const Real G = ::G_vortex;
	const Real R_v = 0.1 * (prob_hi[0] - prob_lo[0]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];

		Real const P = P_ref + -0.5 * std::pow(G, 2) * P_ref * gamma * std::exp((-std::pow(x, 2) - std::pow(y, 2)) / std::pow(R_v, 2)) /
					   (std::pow(R_v, 2) * std::pow(c, 2));

		Real const rho = P / (R * T_ref);
		Real const u = u0 + -G * y * std::exp(-1.0 / 2.0 * (std::pow(x, 2) + std::pow(y, 2)) / std::pow(R_v, 2)) / std::pow(R_v, 2);
		Real const v = v0 + G * x * std::exp(-1.0 / 2.0 * (std::pow(x, 2) + std::pow(y, 2)) / std::pow(R_v, 2)) / std::pow(R_v, 2);
		Real const w = w0;

		Real const xmom = rho * u;
		Real const ymom = rho * v;
		Real const zmom = rho * w;
		Real const Eint = quokka::EOS<Vortex>::ComputeEintFromPres(rho, P);
		Real const Egas = RadSystem<Vortex>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);
		Real const scalar = ::s0[0];

		state_cc(i, j, k, HydroSystem<Vortex>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<Vortex>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<Vortex>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<Vortex>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<Vortex>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<Vortex>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<Vortex>::scalar0_index) = scalar;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<Vortex>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
											    int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
											    const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
											    int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();

	if constexpr (outflow_boundary_along_x_axis) {
		const int ilo = domain_lo[0];
		const int ihi = domain_hi[0];
		if (i < ilo) {
			NSCBC::setOutflowBoundary<Vortex, FluxDir::X1, NSCBC::BoundarySide::Lower>(iv, consVar, geom, ::P_ref);
		} else if (i > ihi) {
			NSCBC::setOutflowBoundary<Vortex, FluxDir::X1, NSCBC::BoundarySide::Upper>(iv, consVar, geom, ::P_ref);
		}
	} else { // outflow boundary is along the y-axis
		const int jlo = domain_lo[1];
		const int jhi = domain_hi[1];
		if (j < jlo) {
			NSCBC::setOutflowBoundary<Vortex, FluxDir::X2, NSCBC::BoundarySide::Lower>(iv, consVar, geom, ::P_ref);
		} else if (j > jhi) {
			NSCBC::setOutflowBoundary<Vortex, FluxDir::X2, NSCBC::BoundarySide::Upper>(iv, consVar, geom, ::P_ref);
		}
	}
}

auto problem_main() -> int
{
	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<Vortex>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		if constexpr (outflow_boundary_along_x_axis) {
			BCs_cc[n].setLo(0, amrex::BCType::ext_dir); // x- NSCBC outflow
			BCs_cc[n].setHi(0, amrex::BCType::ext_dir);
			BCs_cc[n].setLo(1, amrex::BCType::int_dir); // y- periodic
			BCs_cc[n].setHi(1, amrex::BCType::int_dir);
		} else {
			BCs_cc[n].setLo(0, amrex::BCType::int_dir); // x- periodic
			BCs_cc[n].setHi(0, amrex::BCType::int_dir);
			BCs_cc[n].setLo(1, amrex::BCType::ext_dir); // y- NSCBC outflow
			BCs_cc[n].setHi(1, amrex::BCType::ext_dir);
		}
		if (AMREX_SPACEDIM == 3) {
			BCs_cc[n].setLo(2, amrex::BCType::int_dir); // z- periodic
			BCs_cc[n].setHi(2, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<Vortex> sim(BCs_cc);

	amrex::ParmParse const pp("vortex");
	// initial condition parameters
	pp.query("strength", ::G_vortex); // vortex strength
	pp.query("Tgas0", ::T_ref);	  // initial temperature [K]
	pp.query("P0", ::P_ref);	  // initial pressure [erg cm^-3]
	// boundary condition parameters
	pp.query("u_inflow", ::u0);    // inflow velocity along x-axis [cm/s]
	pp.query("v_inflow", ::v0);    // transverse inflow velocity (v_y) [cm/s]
	pp.query("w_inflow", ::w0);    // transverse inflow velocity (v_z) [cm/s]
	pp.query("s_inflow", ::s0[0]); // inflow passive scalar [dimensionless]

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	const int status = 0;
	return status;
}
