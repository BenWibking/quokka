//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include <csignal>
#include <limits>

#include "AMReX_Algorithm.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_BoxArray.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"

#include "linear_advection/AdvectionSimulation.hpp"
#include "test_advection2d.hpp"

using amrex::Real;

struct SquareProblem {
};

template <> struct Physics_Traits<SquareProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto exactSolutionAtIndex(int i, int j, amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo,
							      amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_hi,
							      amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx) -> Real
{
	Real const x = prob_lo[0] + (i + Real(0.5)) * dx[0];
	Real const y = prob_lo[1] + (j + Real(0.5)) * dx[1];
	Real const x0 = prob_lo[0] + Real(0.5) * (prob_hi[0] - prob_lo[0]);
	Real const y0 = prob_lo[1] + Real(0.5) * (prob_hi[1] - prob_lo[1]);

	Real rho = 0.;
	if ((std::abs(x - x0) < 0.1) && (std::abs(y - y0) < 0.1)) {
		rho = 1.;
	}
	return rho;
}

template <> void AdvectionSimulation<SquareProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, Physics_Indices<SquareProblem>::nvarTotal_cc,
			   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { state_cc(i, j, k, n) = exactSolutionAtIndex(i, j, prob_lo, prob_hi, dx); });
}

template <>
void AdvectionSimulation<SquareProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
{
	// compute exact solution

	for (amrex::MFIter iter(state_old_cc_[0]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = ref.array(iter);
		const int ncomp_cc = Physics_Indices<SquareProblem>::nvarTotal_cc;

		amrex::ParallelFor(indexRange, ncomp_cc,
				   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { state(i, j, k, n) = exactSolutionAtIndex(i, j, prob_lo, prob_hi, dx); });
	}
}

template <> void AdvectionSimulation<SquareProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const Real eta_threshold = 0.5; // gradient refinement threshold
	const Real rho_min = 0.1;	// minimum rho for refinement
	auto const &dx = geom[lev].CellSizeArray();

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			int const n = 0;
			Real const rho = state(i, j, k, n);

			Real const del_x = (state(i + 1, j, k, n) - state(i - 1, j, k, n)) / (2.0 * dx[0]);
			Real const del_y = (state(i, j + 1, k, n) - state(i, j - 1, k, n)) / (2.0 * dx[1]);
			Real const gradient_indicator = std::sqrt(del_x * del_x + del_y * del_y) / rho;

			if (gradient_indicator > eta_threshold && rho >= rho_min) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// check that we are in strict IEEE 754 mode
	// (If we are, then the results should be symmetric [about the diagonal of the
	// grid] not only to machine epsilon but to every last digit! BUT not
	// necessarily true when refinement is enabled.)
	static_assert(std::numeric_limits<double>::is_iec559);

	// Problem parameters
	// const int nx = 100;
	// const double Lx = 1.0;
	const double advection_velocity = 1.0; // same for x- and y- directions
	const double CFL_number = 0.4;
	const double max_time = 1.0;
	const int max_timesteps = 1e4;
	const int nvars = 1;

	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	AdvectionSimulation<SquareProblem> sim(BCs_cc);

	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = 1000;
	sim.checkpointInterval_ = -1;

	sim.advectionVx_ = advection_velocity;
	sim.advectionVy_ = advection_velocity;

	// set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	int status;
	if (sim.errorNorm_ < 0.15) {
		status = 0;
	} else {
		status = 1;
	}
	return status;
}
