//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"

struct RichtmeyerMeshkovProblem {
};

template <> struct quokka::EOS_Traits<RichtmeyerMeshkovProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<RichtmeyerMeshkovProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<RichtmeyerMeshkovProblem> {
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

template <> void QuokkaSimulation<RichtmeyerMeshkovProblem>::computeAfterTimestep()
{
	const int ncomp_cc = Physics_Indices<RichtmeyerMeshkovProblem>::nvarTotal_cc;

	// copy all FAB data to a single FAB on rank zero
	amrex::Box const domainBox = geom[0].Domain();
	amrex::BoxArray const localBoxes(domainBox);
	amrex::Vector<int> const ranks({0}); // workaround nvcc bug
	amrex::DistributionMapping const localDistribution(ranks);
	amrex::MultiFab state_mf(localBoxes, localDistribution, ncomp_cc, 0);
	state_mf.ParallelCopy(state_new_cc_[0]);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &state = state_mf.array(0);
		auto const prob_lo = geom[0].ProbLoArray();
		auto const dx = geom[0].CellSizeArray();

		amrex::Long asymmetry = 0;
		auto nx = domainBox.length(0);
		auto ny = domainBox.length(1);
		auto nz = domainBox.length(2);
		auto ncomp = ncomp_cc;
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				for (int k = 0; k < nz; ++k) {
					amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
					amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
					for (int n = 0; n < ncomp; ++n) {
						const amrex::Real comp_upper = state(i, j, k, n);

						// reflect across x/y diagonal
						int n_lower = n;
						if (n == HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index) {
							n_lower = HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index;
						} else if (n == HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index) {
							n_lower = HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index;
						}

						amrex::Real comp_lower = state(j, i, k, n_lower);

						const amrex::Real average = std::fabs(comp_upper + comp_lower);
						const amrex::Real residual = std::abs(comp_upper - comp_lower) / average;

						if (comp_upper != comp_lower) {
							amrex::Print() << i << ", " << j << ", " << k << ", " << n << ", " << comp_upper << ", " << comp_lower
								       << " " << residual << "\n";
							amrex::Print() << "x = " << x << "\n";
							amrex::Print() << "y = " << y << "\n";
							asymmetry++;
							AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, "x/y not symmetric!");
						}
					}
				}
			}
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(asymmetry == 0, "x/y not symmetric!");
	}
}

template <> void QuokkaSimulation<RichtmeyerMeshkovProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];

		double vx = 0.;
		double vy = 0.;
		double vz = 0.;
		double rho = NAN;
		double P = NAN;

		if ((x + y) > 0.15) {
			P = 1.0;
			rho = 1.0;
		} else {
			P = 0.14;
			rho = 0.125;
		}

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(vy));
		AMREX_ASSERT(!std::isnan(vz));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto v_sq = vx * vx + vy * vy + vz * vz;
		const auto gamma = quokka::EOS_Traits<RichtmeyerMeshkovProblem>::gamma;

		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<RichtmeyerMeshkovProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * v_sq;
	});
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<RichtmeyerMeshkovProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<RichtmeyerMeshkovProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<RichtmeyerMeshkovProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<RichtmeyerMeshkovProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
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
	QuokkaSimulation<RichtmeyerMeshkovProblem> sim(BCs_cc);

	sim.stopTime_ = 2.5;
	sim.cflNumber_ = 0.4;
	sim.maxTimesteps_ = 50000;
	sim.plotfileInterval_ = 100;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
