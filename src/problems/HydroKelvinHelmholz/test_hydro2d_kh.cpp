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
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "test_hydro2d_kh.hpp"

struct KelvinHelmholzProblem {
};

template <> struct quokka::EOS_Traits<KelvinHelmholzProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<KelvinHelmholzProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<KelvinHelmholzProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> void QuokkaSimulation<KelvinHelmholzProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const A = 0.01;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];

		double const L = 0.01;	  // shearing layer thickness
		double const sigma = 0.2; // perturbation thickness
		double const yy = std::abs(y - y0) - 0.25;

		double rho = 1.5 - 0.5 * std::tanh(yy / L);
		double vx = 0.5 * std::tanh(yy / L);
		double vy = A * std::cos(4.0 * M_PI * (x - x0)) * std::exp(-(yy * yy) / (sigma * sigma));
		double vz = 0.;
		double P = 2.5;

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(vy));
		AMREX_ASSERT(!std::isnan(vz));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto v_sq = vx * vx + vy * vy + vz * vz;
		const auto gamma = quokka::EOS_Traits<KelvinHelmholzProblem>::gamma;

		state_cc(i, j, k, HydroSystem<KelvinHelmholzProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<KelvinHelmholzProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<KelvinHelmholzProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<KelvinHelmholzProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<KelvinHelmholzProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * v_sq;
	});
}

template <> void QuokkaSimulation<KelvinHelmholzProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.2; // gradient refinement threshold
	const amrex::Real rho_min = 0.1;       // minimum density for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const int n = HydroSystem<KelvinHelmholzProblem>::density_index;
			amrex::Real const rho = state(i, j, k, n);
			amrex::Real const rho_xplus = state(i + 1, j, k, n);
			amrex::Real const rho_xminus = state(i - 1, j, k, n);
			amrex::Real const rho_yplus = state(i, j + 1, k, n);
			amrex::Real const rho_yminus = state(i, j - 1, k, n);

			amrex::Real const del_x = std::max(std::abs(rho_xplus - rho), std::abs(rho - rho_xminus));
			amrex::Real const del_y = std::max(std::abs(rho_yplus - rho), std::abs(rho - rho_yminus));

			amrex::Real const gradient_indicator = std::max(del_x, del_y) / std::max(rho, rho_min);

			if (gradient_indicator > eta_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const int ncomp_cc = Physics_Indices<KelvinHelmholzProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir); // periodic
		}
	}

	// Problem initialization
	QuokkaSimulation<KelvinHelmholzProblem> sim(BCs_cc);

	sim.stopTime_ = 1.5;
	sim.cflNumber_ = 0.4;
	sim.maxTimesteps_ = 40000;
	sim.plotfileInterval_ = 100;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	return 0;
}
