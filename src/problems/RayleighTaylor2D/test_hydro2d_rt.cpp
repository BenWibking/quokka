//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include <cmath>

#include "AMReX_BLassert.H"
#include "AMReX_ParmParse.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"

struct RTProblem {
};

template <> struct quokka::EOS_Traits<RTProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<RTProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<RTProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

amrex::Real constexpr g_x = 0;
amrex::Real constexpr g_y = -0.1;
amrex::Real constexpr g_z = 0;

template <> void QuokkaSimulation<RTProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const A = 0.01;

	// loop over the grid and set the initial condition
	amrex::ParallelForRNG(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const &rng) noexcept {
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];

		double rho = NAN;
		double scalar = NAN;
		if (y > 0.0) {
			rho = 2.0;
			scalar = 1.0;
		} else {
			rho = 1.0;
			scalar = 0.0;
		}

		double amp = A * amrex::Random(rng);

		double vx = 0;
		double vy = amp * (1.0 + std::cos(8.0 * M_PI * y / 3.0)) / 2.0;
		double vz = 0;
		double P0 = 2.5;
		double P = P0 + rho * g_y * y;

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(vy));
		AMREX_ASSERT(!std::isnan(vz));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto v_sq = vx * vx + vy * vy + vz * vz;
		const auto gamma = quokka::EOS_Traits<RTProblem>::gamma;

		state_cc(i, j, k, HydroSystem<RTProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<RTProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<RTProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<RTProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<RTProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * v_sq;
		state_cc(i, j, k, HydroSystem<RTProblem>::internalEnergy_index) = P / (gamma - 1.);
		state_cc(i, j, k, HydroSystem<RTProblem>::scalar0_index) = scalar;
	});
}

template <>
void QuokkaSimulation<RTProblem>::addStrangSplitSources(amrex::MultiFab &state_mf, const int /*lev*/, const amrex::Real /*time*/, const amrex::Real dt)
{
	// add gravitational source terms
	const auto state = state_mf.arrays();

	amrex::ParallelFor(state_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// save initial KE
		amrex::Real const rho = state[bx](i, j, k, HydroSystem<RTProblem>::density_index);
		amrex::Real px = state[bx](i, j, k, HydroSystem<RTProblem>::x1Momentum_index);
		amrex::Real py = state[bx](i, j, k, HydroSystem<RTProblem>::x2Momentum_index);
		amrex::Real pz = state[bx](i, j, k, HydroSystem<RTProblem>::x3Momentum_index);

		amrex::Real const KE_init = (px * px + py * py + pz * pz) / (2.0 * rho);

		// add body forces
		px += dt * rho * g_x;
		py += dt * rho * g_y;
		pz += dt * rho * g_z;

		// compute new KE
		amrex::Real const KE_final = (px * px + py * py + pz * pz) / (2.0 * rho);
		amrex::Real const dKE = KE_final - KE_init;

		// update variables
		state[bx](i, j, k, HydroSystem<RTProblem>::x1Momentum_index) = px;
		state[bx](i, j, k, HydroSystem<RTProblem>::x2Momentum_index) = py;
		state[bx](i, j, k, HydroSystem<RTProblem>::x3Momentum_index) = pz;
		state[bx](i, j, k, HydroSystem<RTProblem>::energy_index) += dKE;
	});
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<RTProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.2; // gradient refinement threshold
	const amrex::Real rho_min = 0.1;       // minimum density for refinement

	const auto state = state_new_cc_[lev].const_arrays();
	const auto tag = tags.arrays();

	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const int n = HydroSystem<RTProblem>::density_index;
		amrex::Real const rho = state[bx](i, j, k, n);
		amrex::Real const rho_xplus = state[bx](i + 1, j, k, n);
		amrex::Real const rho_xminus = state[bx](i - 1, j, k, n);
		amrex::Real const rho_yplus = state[bx](i, j + 1, k, n);
		amrex::Real const rho_yminus = state[bx](i, j - 1, k, n);

		amrex::Real const del_x = 0.5 * (rho_xplus - rho_xminus);
		amrex::Real const del_y = 0.5 * (rho_yplus - rho_yminus);

		amrex::Real const gradient_indicator = std::sqrt(del_x * del_x + del_y * del_y) / rho;

		if ((gradient_indicator > eta_threshold) && (rho > rho_min)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}

auto problem_main() -> int
{
	// Set boundary conditions
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<RTProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<RTProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<RTProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<RTProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		// periodic in x-direction
		BCs_cc[n].setLo(0, amrex::BCType::int_dir);
		BCs_cc[n].setHi(0, amrex::BCType::int_dir);

		// reflecting in y- and z- directions
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<RTProblem> sim(BCs_cc);

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	return 0;
}
