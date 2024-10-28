//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file cloud.cpp
/// \brief Implements a shock-cloud problem with radiative cooling.
///

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MFParallelFor.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "AMReX_SPACE.H"

#include "QuokkaSimulation.hpp"
#include "cooling/TabulatedCooling.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/NSCBC_inflow.hpp"
#include "hydro/NSCBC_outflow.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"

using amrex::Real;

struct ShockCloud {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double seconds_in_year = 3.1536e7; // s == 1 yr
constexpr double parsec_in_cm = 3.086e18;    // cm == 1 pc
constexpr double solarmass_in_g = 1.99e33;   // g == 1 Msun
constexpr double keV_in_ergs = 1.60218e-9;   // ergs == 1 keV
constexpr double m_H = C::m_p + C::m_e;	     // mass of hydrogen atom

template <> struct Physics_Traits<ShockCloud> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 3;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct quokka::EOS_Traits<ShockCloud> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

/// global variables
// Problem properties (set inside problem_main())
bool sharp_cloud_edge = false; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
bool do_frame_shift = true;    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Cloud parameters (set inside problem_main())
AMREX_GPU_MANAGED Real rho0 = NAN;	     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real rho1 = NAN;	     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P0 = NAN;	     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real R_cloud = NAN;	     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real cloud_relpos_x = 0.5; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// cloud-tracking variables needed for Dirichlet boundary condition
AMREX_GPU_MANAGED Real shock_crossing_time = 0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real rho_wind = 0;		// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real v_wind = 0;		// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real P_wind = 0;		// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED Real delta_vx = 0;		// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

template <> void QuokkaSimulation<ShockCloud>::setInitialConditionsOnGrid(quokka::grid const &grid)
{
	amrex::GpuArray<Real, AMREX_SPACEDIM> const dx = grid.dx_;
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo = grid.prob_lo_;
	amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi = grid.prob_hi_;

	Real const x0 = prob_lo[0] + cloud_relpos_x * (prob_hi[0] - prob_lo[0]);
	Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	const bool sharp_cloud_edge = ::sharp_cloud_edge;

	const amrex::Box &indexRange = grid.indexRange_;
	auto const &state = grid.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const x = prob_lo[0] + (i + static_cast<Real>(0.5)) * dx[0];
		Real const y = prob_lo[1] + (j + static_cast<Real>(0.5)) * dx[1];
		Real const z = prob_lo[2] + (k + static_cast<Real>(0.5)) * dx[2];
		Real const R = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

		Real rho = NAN;
		Real C = NAN;

		if (sharp_cloud_edge) {
			if (R < R_cloud) {
				rho = rho1; // cloud density
				C = 1.0;    // concentration is unity inside the cloud
			} else {
				rho = rho0; // background density
				C = 0.0;    // concentration is zero outside the cloud
			}
		} else {
			const Real h_smooth = R_cloud / 20.;
			const Real ramp = 0.5 * (1. - std::tanh((R - R_cloud) / h_smooth));
			rho = ramp * (rho1 - rho0) + rho0;
			C = ramp * 1.0; // concentration is unity inside the cloud
		}

		AMREX_ALWAYS_ASSERT(rho > 0.);
		AMREX_ALWAYS_ASSERT(C >= 0.);
		AMREX_ALWAYS_ASSERT(C <= 1.);

		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = P0 / (quokka::EOS_Traits<ShockCloud>::gamma - 1.);
		Real const Egas = RadSystem<ShockCloud>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		state(i, j, k, RadSystem<ShockCloud>::gasDensity_index) = rho;
		state(i, j, k, RadSystem<ShockCloud>::x1GasMomentum_index) = xmom;
		state(i, j, k, RadSystem<ShockCloud>::x2GasMomentum_index) = ymom;
		state(i, j, k, RadSystem<ShockCloud>::x3GasMomentum_index) = zmom;
		state(i, j, k, RadSystem<ShockCloud>::gasEnergy_index) = Egas;
		state(i, j, k, RadSystem<ShockCloud>::gasInternalEnergy_index) = Eint;
		state(i, j, k, RadSystem<ShockCloud>::scalar0_index) = C;
		state(i, j, k, RadSystem<ShockCloud>::scalar0_index + 1) = C * rho;	    // cloud partial density
		state(i, j, k, RadSystem<ShockCloud>::scalar0_index + 2) = (1.0 - C) * rho; // non-cloud partial density
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<ShockCloud>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
												int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
												const Real time, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
												int /*orig_comp*/)
{
	auto [i, j, k] = iv.toArray();

	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int ilo = domain_lo[0];
	const int ihi = domain_hi[0];

	const Real delta_vx = ::delta_vx;
	const Real rho_wind = ::rho_wind;
	const Real v_wind = ::v_wind;
	const Real P_wind = ::P_wind;

	if (i < ilo) {
		// x1 lower boundary
		Real const rho = rho_wind;
		Real const vx = v_wind - delta_vx;
		Real const Eint = quokka::EOS<ShockCloud>::ComputeEintFromPres(rho, P_wind);
		Real const T_wind = quokka::EOS<ShockCloud>::ComputeTgasFromEint(rho, Eint);
		amrex::GpuArray<amrex::Real, HydroSystem<ShockCloud>::nscalars_> scalars{0, 0, rho};

		if (time < 0.1 * ::shock_crossing_time) {
			// Shock boundary condition [all primitive variables specified]
			Real const xmom = rho_wind * vx;
			Real const ymom = 0;
			Real const zmom = 0;
			Real const Egas = RadSystem<ShockCloud>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);
			consVar(i, j, k, RadSystem<ShockCloud>::gasDensity_index) = rho;
			consVar(i, j, k, RadSystem<ShockCloud>::x1GasMomentum_index) = xmom;
			consVar(i, j, k, RadSystem<ShockCloud>::x2GasMomentum_index) = ymom;
			consVar(i, j, k, RadSystem<ShockCloud>::x3GasMomentum_index) = zmom;
			consVar(i, j, k, RadSystem<ShockCloud>::gasEnergy_index) = Egas;
			consVar(i, j, k, RadSystem<ShockCloud>::gasInternalEnergy_index) = Eint;
			consVar(i, j, k, RadSystem<ShockCloud>::scalar0_index) = scalars[0];
			consVar(i, j, k, RadSystem<ShockCloud>::scalar0_index + 1) = scalars[1]; // cloud partial density
			consVar(i, j, k, RadSystem<ShockCloud>::scalar0_index + 2) = scalars[2]; // non-cloud partial density
		} else {
			// Subsonic inflow boundary condition [all *except one* primitive variable specified]
			NSCBC::setInflowX1LowerLowOrder<ShockCloud>(iv, consVar, geom, T_wind, vx, 0, 0, scalars);
		}
	} else if (i > ihi) {
		// x1 upper boundary -- NSCBC subsonic outflow
		// (For this boundary condition, we must specify the pressure at the boundary.)

		if (time < 1.1 * ::shock_crossing_time) {
			// before the shock hits the boundary, set the boundary pressure to the background pressure P0.
			NSCBC::setOutflowBoundaryLowOrder<ShockCloud, FluxDir::X1, NSCBC::BoundarySide::Upper>(iv, consVar, geom, ::P0);
		} else {
			// shock has passed, so we set the boundary pressure to the wind pressure P_wind.
			NSCBC::setOutflowBoundaryLowOrder<ShockCloud, FluxDir::X1, NSCBC::BoundarySide::Upper>(iv, consVar, geom, P_wind);
		}
	}
}

template <> void QuokkaSimulation<ShockCloud>::computeAfterTimestep()
{
	const amrex::Real dt_coarse = dt_[0];
	const amrex::Real time = tNew_[0];

	// perform Galilean transformation (velocity shift to center-of-mass frame)
	if (::do_frame_shift && (time >= ::shock_crossing_time)) {

		// N.B. must weight by passive scalar of cloud, since the background has
		// non-negligible momentum!
		int const nc = 1; // number of components in temporary MF
		int const ng = 0; // number of ghost cells in temporary MF
		amrex::MultiFab temp_mf(boxArray(0), DistributionMap(0), nc, ng);

		// compute x-momentum
		amrex::MultiFab::Copy(temp_mf, state_new_cc_[0], HydroSystem<ShockCloud>::x1Momentum_index, 0, nc, ng);
		amrex::MultiFab::Multiply(temp_mf, state_new_cc_[0], HydroSystem<ShockCloud>::scalar0_index, 0, nc, ng);
		const Real xmom = temp_mf.sum(0);

		// compute cloud mass within simulation box
		amrex::MultiFab::Copy(temp_mf, state_new_cc_[0], HydroSystem<ShockCloud>::density_index, 0, nc, ng);
		amrex::MultiFab::Multiply(temp_mf, state_new_cc_[0], HydroSystem<ShockCloud>::scalar0_index, 0, nc, ng);
		const Real cloud_mass = temp_mf.sum(0);

		// compute center-of-mass velocity of the cloud
		const Real vx_cm = xmom / cloud_mass;

		// save cumulative position, velocity offsets in simulationMetadata_
		const Real delta_x_prev = std::get<Real>(simulationMetadata_["delta_x"]);
		const Real delta_vx_prev = std::get<Real>(simulationMetadata_["delta_vx"]);
		const Real delta_x = delta_x_prev + dt_coarse * delta_vx_prev;
		const Real delta_vx = delta_vx_prev + vx_cm;
		simulationMetadata_["delta_x"] = delta_x;
		simulationMetadata_["delta_vx"] = delta_vx;
		::delta_vx = delta_vx;

		const Real v_wind = ::v_wind;
		amrex::Print() << "[Cloud Tracking] Delta x = " << (delta_x / parsec_in_cm) << " pc," << " Delta vx = " << (delta_vx / 1.0e5) << " km/s,"
			       << " Inflow velocity = " << ((v_wind - delta_vx) / 1.0e5) << " km/s\n";

		// If we are moving faster than the wind, we should abort the simulation.
		// (otherwise, the boundary conditions become inconsistent.)
		AMREX_ALWAYS_ASSERT(delta_vx < v_wind);

		// subtract center-of-mass y-velocity on each level
		// N.B. must update both y-momentum *and* energy!
		for (int lev = 0; lev <= finest_level; ++lev) {
			auto const &mf = state_new_cc_[lev];
			auto const &state = state_new_cc_[lev].arrays();
			amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) noexcept {
				Real const rho = state[box](i, j, k, HydroSystem<ShockCloud>::density_index);
				Real const xmom = state[box](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
				Real const ymom = state[box](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
				Real const zmom = state[box](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
				Real const E = state[box](i, j, k, HydroSystem<ShockCloud>::energy_index);
				Real const KE = 0.5 * (xmom * xmom + ymom * ymom + zmom * zmom) / rho;
				Real const Eint = E - KE;
				Real const new_xmom = xmom - rho * vx_cm;
				Real const new_KE = 0.5 * (new_xmom * new_xmom + ymom * ymom + zmom * zmom) / rho;

				state[box](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index) = new_xmom;
				state[box](i, j, k, HydroSystem<ShockCloud>::energy_index) = Eint + new_KE;
			});
		}
		amrex::Gpu::streamSynchronizeAll();
	}
}

template <> void QuokkaSimulation<ShockCloud>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_in) const
{
	// compute derived variables and save in 'mf'

	if (dname == "temperature") {
		const int ncomp = ncomp_in;
		auto tables = cloudyTables_.const_tables();
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
			Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
			Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
			Real const Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
			Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			Real const Tgas = ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
			output[bx](i, j, k, ncomp) = Tgas;
		});

	} else if (dname == "c_s") {
		const int ncomp = ncomp_in;
		auto tables = cloudyTables_.const_tables();
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
			Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
			Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
			Real const Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
			Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			Real const Tgas = quokka::TabulatedCooling::ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
			Real const mu = quokka::TabulatedCooling::ComputeMMW(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
			Real const cs = std::sqrt(HydroSystem<ShockCloud>::gamma_ * C::k_B * Tgas / (mu * m_H));
			output[bx](i, j, k, ncomp) = cs / 1.0e5; // km/s
		});

	} else if (dname == "nH") {
		const int ncomp = ncomp_in;
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			Real const nH = (quokka::TabulatedCooling::cloudy_H_mass_fraction * rho) / m_H;
			output[bx](i, j, k, ncomp) = nH;
		});

	} else if (dname == "pressure") {
		const int ncomp = ncomp_in;
		auto tables = cloudyTables_.const_tables();
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
			Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
			Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
			Real const Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
			Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			Real const Tgas = ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
			Real const mu = ComputeMMW(rho, Egas, HydroSystem<ShockCloud>::gamma_, tables);
			Real const ndens = rho / (mu * m_H);
			output[bx](i, j, k, ncomp) = ndens * Tgas; // [K cm^-3]
		});

	} else if (dname == "entropy") {
		const int ncomp = ncomp_in;
		auto tables = cloudyTables_.const_tables();
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
			Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
			Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
			Real const Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
			Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			Real const Tgas = ComputeTgasFromEgas(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
			Real const mu = ComputeMMW(rho, Egas, HydroSystem<ShockCloud>::gamma_, tables);
			Real const ndens = rho / (mu * m_H);
			Real const K_cgs = C::k_B * Tgas * std::pow(ndens, -2. / 3.); // ergs cm^2
			Real const K_keV_cm2 = K_cgs / keV_in_ergs;		      // convert to units of keV cm^2
			output[bx](i, j, k, ncomp) = K_keV_cm2;
		});

	} else if (dname == "mass") {
		const int ncomp = ncomp_in;
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();
		auto const dx = geom[lev].CellSizeArray();
		const Real dvol = dx[0] * dx[1] * dx[2];

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			output[bx](i, j, k, ncomp) = rho * dvol;
		});

	} else if (dname == "cloud_fraction") {
		const int ncomp = ncomp_in;
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// cloud partial density
			Real const rho_cloud = state[bx](i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
			// non-cloud partial density
			Real const rho_bg = state[bx](i, j, k, HydroSystem<ShockCloud>::scalar0_index + 2);

			// NOTE: rho_cloud + rho_bg only equals hydro rho up to truncation error!
			output[bx](i, j, k, ncomp) = rho_cloud / (rho_cloud + rho_bg);
		});

	} else if (dname == "cooling_length") {
		const int ncomp = ncomp_in;
		auto tables = cloudyTables_.const_tables();
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// compute cooling length in parsec
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
			Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
			Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
			Real const Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
			Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			Real const l_cool = ComputeCoolingLength(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);
			output[bx](i, j, k, ncomp) = l_cool / parsec_in_cm;
		});

	} else if (dname == "lab_velocity_x") {
		const int ncomp = ncomp_in;
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();
		const Real delta_vx = ::delta_vx;

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// compute observer velocity in km/s
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
			Real const vx = x1Mom / rho;
			Real const vx_lab = vx + delta_vx;
			output[bx](i, j, k, ncomp) = vx_lab / 1.0e5; // km/s
		});

	} else if (dname == "velocity_mag") {
		const int ncomp = ncomp_in;
		auto const &output = mf.arrays();
		auto const &state = state_new_cc_[lev].const_arrays();

		amrex::ParallelFor(mf, mf.nGrowVect(), [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// compute simulation-frame |v| in km/s
			Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
			Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
			Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
			Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
			Real const v1 = x1Mom / rho;
			Real const v2 = x2Mom / rho;
			Real const v3 = x3Mom / rho;
			output[bx](i, j, k, ncomp) = std::sqrt(v1 * v1 + v2 * v2 + v3 * v3) / 1.0e5; // km/s
		});
	}
	amrex::Gpu::streamSynchronizeAll();
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto ComputeCellTemp(int i, int j, int k, amrex::Array4<const Real> const &state, amrex::Real gamma,
							 quokka::TabulatedCooling::cloudyGpuConstTables const &tables)
{
	// return cell temperature
	Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
	Real const x1Mom = state(i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
	Real const x2Mom = state(i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
	Real const x3Mom = state(i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
	Real const Egas = state(i, j, k, HydroSystem<ShockCloud>::energy_index);
	Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
	return quokka::TabulatedCooling::ComputeTgasFromEgas(rho, Eint, gamma, tables);
}

template <> auto QuokkaSimulation<ShockCloud>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	// compute scalar statistics
	std::map<std::string, amrex::Real> stats;

	// save time
	const Real t_cc = std::get<Real>(simulationMetadata_["t_cc"]);
	const Real time = tNew_[0];
	stats["t_over_tcc"] = time / t_cc;

	// save cloud position, velocity
	const Real dx_cgs = std::get<Real>(simulationMetadata_["delta_x"]);
	const Real dvx_cgs = std::get<Real>(simulationMetadata_["delta_vx"]);
	const Real v_wind = ::v_wind;

	stats["delta_x"] = dx_cgs / parsec_in_cm;	 // pc
	stats["delta_vx"] = dvx_cgs / 1.0e5;		 // km/s
	stats["inflow_vx"] = (v_wind - dvx_cgs) / 1.0e5; // km/s

	// save total simulation mass
	const Real sim_mass = amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(state_new_cc_), HydroSystem<ShockCloud>::density_index, geom, ref_ratio);
	const Real sim_partialcloud_mass =
	    amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(state_new_cc_), HydroSystem<ShockCloud>::scalar0_index + 1, geom, ref_ratio);
	const Real sim_partialwind_mass =
	    amrex::volumeWeightedSum(amrex::GetVecOfConstPtrs(state_new_cc_), HydroSystem<ShockCloud>::scalar0_index + 2, geom, ref_ratio);

	stats["sim_mass"] = sim_mass / solarmass_in_g;
	stats["sim_partialcloud_mass"] = sim_partialcloud_mass / solarmass_in_g;
	stats["sim_partialwind_mass"] = sim_partialwind_mass / solarmass_in_g;

	// compute cloud mass according to temperature threshold
	auto tables = cloudyTables_.const_tables();

	const Real M_cl_1e4 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = (T < 1.0e4) ? rho : 0.0;
		return result;
	});
	const Real M_cl_8000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = (T < 8000.) ? rho : 0.0;
		return result;
	});
	const Real M_cl_9000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = (T < 9000.) ? rho : 0.0;
		return result;
	});
	const Real M_cl_11000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = (T < 1.1e4) ? rho : 0.0;
		return result;
	});
	const Real M_cl_12000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = (T < 1.2e4) ? rho : 0.0;
		return result;
	});

	stats["cloud_mass_1e4"] = M_cl_1e4 / solarmass_in_g;
	stats["cloud_mass_8000"] = M_cl_8000 / solarmass_in_g;
	stats["cloud_mass_9000"] = M_cl_9000 / solarmass_in_g;
	stats["cloud_mass_11000"] = M_cl_11000 / solarmass_in_g;
	stats["cloud_mass_12000"] = M_cl_12000 / solarmass_in_g;

	const Real origM_cl_1e4 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho_cloud = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
		Real const result = (T < 1.0e4) ? rho_cloud : 0.0;
		return result;
	});
	const Real origM_cl_8000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho_cloud = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
		Real const result = (T < 8000.) ? rho_cloud : 0.0;
		return result;
	});
	const Real origM_cl_9000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho_cloud = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
		Real const result = (T < 9000.) ? rho_cloud : 0.0;
		return result;
	});
	const Real origM_cl_11000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho_cloud = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
		Real const result = (T < 1.1e4) ? rho_cloud : 0.0;
		return result;
	});
	const Real origM_cl_12000 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const T = ComputeCellTemp(i, j, k, state, HydroSystem<ShockCloud>::gamma_, tables);
		Real const rho_cloud = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
		Real const result = (T < 1.2e4) ? rho_cloud : 0.0;
		return result;
	});

	stats["cloud_mass_1e4_original"] = origM_cl_1e4 / solarmass_in_g;
	stats["cloud_mass_8000_original"] = origM_cl_8000 / solarmass_in_g;
	stats["cloud_mass_9000_original"] = origM_cl_9000 / solarmass_in_g;
	stats["cloud_mass_11000_original"] = origM_cl_11000 / solarmass_in_g;
	stats["cloud_mass_12000_original"] = origM_cl_12000 / solarmass_in_g;

	// compute cloud mass according to passive scalar threshold
	const Real M_cl_scalar_01 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const C = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index);
		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = (C > 0.1) ? rho : 0.0;
		return result;
	});
	const Real M_cl_scalar_01_09 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const C = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index);
		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = ((C > 0.1) && (C < 0.9)) ? rho : 0.0;
		return result;
	});

	stats["cloud_mass_scalar_01"] = M_cl_scalar_01 / solarmass_in_g;
	stats["cloud_mass_scalar_01_09"] = M_cl_scalar_01_09 / solarmass_in_g;

	// compute cloud mass according to cloud fraction threshold
	const Real M_cl_fraction_01 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const rho_cloud = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
		Real const rho_bg = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 2);
		Real const C_frac = rho_cloud / (rho_cloud + rho_bg);

		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = (C_frac > 0.1) ? rho : 0.0;
		return result;
	});
	const Real M_cl_fraction_01_09 = computeVolumeIntegral([=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		Real const rho_cloud = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
		Real const rho_bg = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 2);
		Real const C_frac = rho_cloud / (rho_cloud + rho_bg);

		Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const result = ((C_frac > 0.1) && (C_frac < 0.9)) ? rho : 0.0;
		return result;
	});

	stats["cloud_mass_fraction_01"] = M_cl_fraction_01 / solarmass_in_g;
	stats["cloud_mass_fraction_01_09"] = M_cl_fraction_01_09 / solarmass_in_g;

	return stats;
}

template <> auto QuokkaSimulation<ShockCloud>::ComputeProjections(const int dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;

	// compute (total) density projection
	proj["nH"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
		    return (quokka::TabulatedCooling::cloudy_H_mass_fraction * rho) / m_H;
	    },
	    dir);

	// compute cloud partial density projection
	proj["nH_cloud"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    // partial cloud density
		    Real const rho_cloud = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 1);
		    return (quokka::TabulatedCooling::cloudy_H_mass_fraction * rho_cloud) / m_H;
	    },
	    dir);

	// compute non-cloud partial density projection
	proj["nH_wind"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    // partial wind density
		    Real const rho_wind = state(i, j, k, HydroSystem<ShockCloud>::scalar0_index + 2);
		    return (quokka::TabulatedCooling::cloudy_H_mass_fraction * rho_wind) / m_H;
	    },
	    dir);

	return proj;
}

template <> void QuokkaSimulation<ShockCloud>::ErrorEst(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	const int Ncells_per_lcool = 10;

	amrex::GpuArray<Real, AMREX_SPACEDIM> dx = geom[lev].CellSizeArray();
	const Real min_dx = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	const Real resolved_length = static_cast<Real>(Ncells_per_lcool) * min_dx;

	auto tables = cloudyTables_.const_tables();
	const auto state = state_new_cc_[lev].const_arrays();
	const auto tag = tags.arrays();

	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		Real const rho = state[bx](i, j, k, HydroSystem<ShockCloud>::density_index);
		Real const x1Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x1Momentum_index);
		Real const x2Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x2Momentum_index);
		Real const x3Mom = state[bx](i, j, k, HydroSystem<ShockCloud>::x3Momentum_index);
		Real const Egas = state[bx](i, j, k, HydroSystem<ShockCloud>::energy_index);
		Real const Eint = RadSystem<ShockCloud>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
		Real const l_cool = ComputeCoolingLength(rho, Eint, HydroSystem<ShockCloud>::gamma_, tables);

		if (l_cool < resolved_length) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	// Problem initialization
	constexpr int nvars = QuokkaSimulation<ShockCloud>::nvarTotal_cc_;
	amrex::Vector<amrex::BCRec> boundaryConditions(nvars);
	for (int n = 0; n < nvars; ++n) {
		boundaryConditions[n].setLo(0, amrex::BCType::ext_dir); // Dirichlet
		boundaryConditions[n].setHi(0, amrex::BCType::ext_dir); // NSCBC outflow

		boundaryConditions[n].setLo(1, amrex::BCType::int_dir); // periodic
		boundaryConditions[n].setHi(1, amrex::BCType::int_dir);

		boundaryConditions[n].setLo(2, amrex::BCType::int_dir);
		boundaryConditions[n].setHi(2, amrex::BCType::int_dir);
	}
	QuokkaSimulation<ShockCloud> sim(boundaryConditions);

	// Read problem parameters
	amrex::ParmParse const pp;
	Real nH_bg = NAN;
	Real nH_cloud = NAN;
	Real P_over_k = NAN;
	Real M0 = NAN;
	Real max_t_cc = NAN;

	// use a sharp cloud edge?
	int sharp_cloud_edge = 0;
	pp.query("sharp_cloud_edge", sharp_cloud_edge);
	::sharp_cloud_edge = sharp_cloud_edge == 1;

	// do frame shifting to follow cloud center-of-mass?
	int do_frame_shift = 1;
	pp.query("do_frame_shift", do_frame_shift);
	::do_frame_shift = do_frame_shift == 1;

	// background gas H number density
	pp.query("nH_bg", nH_bg); // cm^-3

	// cloud H number density
	pp.query("nH_cloud", nH_cloud); // cm^-3

	// background gas pressure
	pp.query("P_over_k", P_over_k); // K cm^-3

	// cloud radius
	pp.query("R_cloud_pc", ::R_cloud); // pc
	::R_cloud *= parsec_in_cm;	   // convert to cm

	// cloud position (relative to box length)
	pp.query("cloud_relpos_x", ::cloud_relpos_x); // dimensionless

	// (pre-shock) Mach number
	pp.query("Mach_shock", M0); // dimensionless

	// simulation end time (in number of cloud-crushing times)
	pp.query("max_t_cc", max_t_cc); // dimensionless

	// compute background pressure
	// (pressure equilibrium should hold *before* the shock enters the box)
	::P0 = P_over_k * C::k_B; // erg cm^-3
	amrex::Print() << fmt::format("Pressure = {} K cm^-3\n", P_over_k);

	// compute mass density of background, cloud
	::rho0 = nH_bg * m_H / quokka::TabulatedCooling::cloudy_H_mass_fraction;    // g cm^-3
	::rho1 = nH_cloud * m_H / quokka::TabulatedCooling::cloudy_H_mass_fraction; // g cm^-3

	AMREX_ALWAYS_ASSERT(!std::isnan(::rho0));
	AMREX_ALWAYS_ASSERT(!std::isnan(::rho1));
	AMREX_ALWAYS_ASSERT(!std::isnan(::P0));

	// check temperature of cloud, background
	constexpr Real gamma = HydroSystem<ShockCloud>::gamma_;
	auto tables = sim.cloudyTables_.const_tables();
	const Real Eint_bg = ::P0 / (gamma - 1.);
	const Real Eint_cl = ::P0 / (gamma - 1.);
	const Real T_bg = ComputeTgasFromEgas(rho0, Eint_bg, gamma, tables);
	const Real T_cl = ComputeTgasFromEgas(rho1, Eint_cl, gamma, tables);
	amrex::Print() << fmt::format("T_bg = {} K\n", T_bg);
	amrex::Print() << fmt::format("T_cl = {} K\n", T_cl);

	// compute shock jump conditions from rho0, P0, and M0
	const Real x0 = M0 * M0;
	const Real x1 = gamma + 1;
	const Real x2 = gamma * x0;
	const Real x3 = 1.0 / x1;
	const Real x4 = std::sqrt(gamma * ::P0 / ::rho0);
	const Real rho_post = ::rho0 * x0 * x1 / (-x0 + x2 + 2);
	const Real v_wind = 2 * x3 * x4 * (x0 - 1) / M0;
	const Real P_post = ::P0 * x3 * (-gamma + 2 * x2 + 1);
	const Real v_shock = M0 * x4;

	const Real Eint_post = P_post / (gamma - 1.);
	const Real T_post = ComputeTgasFromEgas(rho_post, Eint_post, gamma, tables);
	amrex::Print() << fmt::format("T_wind = {} K\n", T_post);

	::v_wind = v_wind; // set global variables
	::rho_wind = rho_post;
	::P_wind = P_post;
	amrex::Print() << fmt::format("v_wind = {} km/s\n", v_wind / 1.0e5);
	amrex::Print() << fmt::format("P_wind = {} K cm^-3\n", P_post / C::k_B);
	amrex::Print() << fmt::format("v_shock = {} km/s\n", v_shock / 1.0e5);

	// compute shock-crossing time
	::shock_crossing_time = sim.geom[0].ProbLength(0) / v_shock;
	amrex::Print() << fmt::format("shock crossing time = {} Myr\n", ::shock_crossing_time / (1.0e6 * seconds_in_year));

	// compute cloud-crushing time
	const Real chi = rho1 / rho0;
	const Real t_cc = std::sqrt(chi) * R_cloud / v_shock;
	amrex::Print() << fmt::format("t_cc = {} Myr\n", t_cc / (1.0e6 * seconds_in_year)) << "\n";

	// compute maximum simulation time
	const double max_time = max_t_cc * t_cc;

	// set simulation parameters
	sim.stopTime_ = max_time;
	sim.pressureFloor_ = 1.0e-3 * ::P0; // set pressure floor

	// set metadata
	sim.simulationMetadata_["delta_x"] = 0._rt;
	sim.simulationMetadata_["delta_vx"] = 0._rt;
	sim.simulationMetadata_["rho_wind"] = rho_wind;
	sim.simulationMetadata_["v_wind"] = v_wind;
	sim.simulationMetadata_["P_wind"] = P_wind;
	sim.simulationMetadata_["M0"] = M0;
	sim.simulationMetadata_["t_cc"] = t_cc;

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Cleanup and exit
	int const status = 0;
	return status;
}
