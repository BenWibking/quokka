//==============================================================================
// Copyright 2022 Neco Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_fc_quantities.cpp
/// \brief Defines a test problem to make sure face-centred quantities are created correctly.
///

#include <cassert>
#include <ostream>
#include <stdexcept>
#include <valarray>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "physics_info.hpp"
#include "test_fc_quantities.hpp"
#include "util/fextract.hpp"

struct FCQuantities {
};

template <> struct quokka::EOS_Traits<FCQuantities> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<FCQuantities> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double rho0 = 1.0;					     // background density
constexpr double P0 = 1.0 / quokka::EOS_Traits<FCQuantities>::gamma; // background pressure
constexpr double v0 = 0.;					     // background velocity
constexpr double amp = 1.0e-6;					     // perturbation amplitude

AMREX_GPU_DEVICE void computeWaveSolution(int i, int j, int k, amrex::Array4<amrex::Real> const &state, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	const amrex::Real x_L = prob_lo[0] + (i + amrex::Real(0.0)) * dx[0];
	const amrex::Real x_R = prob_lo[0] + (i + amrex::Real(1.0)) * dx[0];
	const amrex::Real A = amp;

	const quokka::valarray<double, 3> R = {1.0, -1.0, 1.5}; // right eigenvector of sound wave
	const quokka::valarray<double, 3> U_0 = {rho0, rho0 * v0, P0 / (quokka::EOS_Traits<FCQuantities>::gamma - 1.0) + 0.5 * rho0 * std::pow(v0, 2)};
	const quokka::valarray<double, 3> dU = (A * R / (2.0 * M_PI * dx[0])) * (std::cos(2.0 * M_PI * x_L) - std::cos(2.0 * M_PI * x_R));

	double rho = U_0[0] + dU[0];
	double xmom = U_0[1] + dU[1];
	double Etot = U_0[2] + dU[2];
	double Eint = Etot - 0.5 * (xmom * xmom) / rho;

	state(i, j, k, HydroSystem<FCQuantities>::density_index) = rho;
	state(i, j, k, HydroSystem<FCQuantities>::x1Momentum_index) = xmom;
	state(i, j, k, HydroSystem<FCQuantities>::x2Momentum_index) = 0;
	state(i, j, k, HydroSystem<FCQuantities>::x3Momentum_index) = 0;
	state(i, j, k, HydroSystem<FCQuantities>::energy_index) = Etot;
	state(i, j, k, HydroSystem<FCQuantities>::internalEnergy_index) = Eint;
}

template <> void QuokkaSimulation<FCQuantities>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	const int ncomp_cc = Physics_Indices<FCQuantities>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int n = 0; n < ncomp_cc; ++n) {
			state(i, j, k, n) = 0; // fill unused quantities with zeros
		}
		computeWaveSolution(i, j, k, state, dx, prob_lo);
	});
}

template <> void QuokkaSimulation<FCQuantities>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::Array4<double> &state = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	if (dir == quokka::direction::x) {
		amrex::ParallelFor(
		    indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = 1.0 + (i % 2); });
	} else if (dir == quokka::direction::y) {
		amrex::ParallelFor(
		    indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = 2.0 + (j % 2); });
	} else if (dir == quokka::direction::z) {
		amrex::ParallelFor(
		    indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept { state(i, j, k, MHDSystem<FCQuantities>::bfield_index) = 3.0 + (k % 2); });
	}
}

void checkMFs(amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &state1,
	      amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &state2)
{
	double err = 0.0;
	for (int level = 0; level < state1.size(); ++level) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			// initialise MF
			const amrex::BoxArray &ba = state1[level][idim].boxArray();
			const amrex::DistributionMapping &dm = state1[level][idim].DistributionMap();
			int ncomp = state1[level][idim].nComp();
			int ngrow = state1[level][idim].nGrow();
			amrex::MultiFab mf_diff(ba, dm, ncomp, ngrow);
			// compute difference between two MFs (at level)
			amrex::MultiFab::Copy(mf_diff, state1[level][idim], 0, 0, ncomp, ngrow);
			amrex::MultiFab::Subtract(mf_diff, state2[level][idim], 0, 0, ncomp, ngrow);
			// compute error (summed over each component)
			for (int icomp = 0; icomp < Physics_Indices<FCQuantities>::nvarPerDim_fc; ++icomp) {
				err += mf_diff.norm1(icomp);
			}
		}
	}
	amrex::Print() << "Accumulated error in MFs read from chk-file: " << err << "\n";
	amrex::Print() << "\n";
	AMREX_ALWAYS_ASSERT(std::abs(err) == 0.0);
}

auto problem_main() -> int
{
	// Problem initialization
	const int ncomp_cc = Physics_Indices<FCQuantities>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	const int nvars_fc = Physics_Indices<FCQuantities>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int n = 0; n < nvars_fc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_fc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_fc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<FCQuantities> sim_write(BCs_cc, BCs_fc);
	sim_write.setInitialConditions();
	amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &state_new_fc_write = sim_write.getNewMF_fc();
	amrex::Print() << "\n";

	QuokkaSimulation<FCQuantities> sim_restart(BCs_cc, BCs_fc);
	sim_restart.setChkFile("chk00000");
	sim_restart.setInitialConditions();
	amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> const &state_new_fc_restart = sim_restart.getNewMF_fc();
	amrex::Print() << "\n";

	amrex::Print() << "Checking new FC MFs...\n";
	checkMFs(state_new_fc_write, state_new_fc_restart);

	return 0;
}
