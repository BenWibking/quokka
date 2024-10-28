//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include <limits>
#include <vector>

#include "AMReX_Array.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "hyperbolic_system.hpp"
#include "linear_advection/AdvectionSimulation.hpp"
#include "linear_advection/linear_advection.hpp"
#include "test_advection_semiellipse.hpp"
#include "util/fextract.hpp"

struct SemiellipseProblem {
};

template <> struct Physics_Traits<SemiellipseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

AMREX_GPU_DEVICE void ComputeExactSolution(int i, int j, int k, int n, amrex::Array4<amrex::Real> const &exact_arr,
					   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	amrex::Real x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
	double dens = 0.0;
	if (std::abs(x - 0.2) <= 0.15) {
		dens = std::sqrt(1.0 - std::pow((x - 0.2) / 0.15, 2));
	}
	exact_arr(i, j, k, n) = dens;
}

template <> void AdvectionSimulation<SemiellipseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const int ncomp_cc = Physics_Indices<SemiellipseProblem>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, ncomp_cc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) { ComputeExactSolution(i, j, k, n, state_cc, dx, prob_lo); });
}

template <>
void AdvectionSimulation<SemiellipseProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
								       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
{

	// fill reference solution multifab
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, ncomp,
				   [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept { ComputeExactSolution(i, j, k, n, stateExact, dx, prob_lo); });
	}

	// Plot results
	auto [position, values] = fextract(state_new_cc_[0], geom[0], 0, 0.5);
	auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.5);

	// interpolate exact solution onto coarse grid
	int nx = static_cast<int>(position.size());
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		xs.at(i) = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
	}

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// extract values
		std::vector<double> d(nx);
		std::vector<double> d_exact(nx);
		for (int i = 0; i < nx; ++i) {
			amrex::Real rho = values.at(0)[i];
			amrex::Real rho_exact = val_exact.at(0)[i];
			d.at(i) = rho;
			d_exact.at(i) = rho_exact;
		}

#ifdef HAVE_PYTHON
		// Plot results
		std::map<std::string, std::string> d_initial_args;
		std::map<std::string, std::string> d_final_args;
		d_initial_args["label"] = "density (exact solution)";
		d_final_args["label"] = "density";

		matplotlibcpp::plot(xs, d_exact, d_initial_args);
		matplotlibcpp::plot(xs, d, d_final_args);
		matplotlibcpp::legend();
		matplotlibcpp::save(std::string("./advection_semiellipse.pdf"));
#endif
	}
}

auto problem_main() -> int
{
	// Based on
	// https://www.mathematik.uni-dortmund.de/~kuzmin/fcttvd.pdf
	// Section 6.2: Convection of a semi-ellipse

	// Problem parameters
	// const int nx = 400;
	// const double Lx = 1.0;
	const double advection_velocity = 1.0;
	const double max_time = 1.0;
	const double max_dt = 1e-4;
	const int max_timesteps = 1e4;

	const int nvars = 1; // only density
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	AdvectionSimulation<SemiellipseProblem> sim(BCs_cc);
	sim.maxDt_ = max_dt;
	sim.advectionVx_ = advection_velocity;
	sim.advectionVy_ = 0.;
	sim.advectionVz_ = 0.;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = -1;

	// set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Compute reference solution
	int status = 0;
	const double err_tol = 0.015;
	if (sim.errorNorm_ > err_tol) {
		status = 1;
	}

	return status;
}
