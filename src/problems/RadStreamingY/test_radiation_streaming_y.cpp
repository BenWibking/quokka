//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_streaming.cpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

#include "test_radiation_streaming_y.hpp"
#include "AMReX.H"
#include "AMReX_BLassert.H"
#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"

struct StreamingProblem {
};

constexpr int direction = 1;

constexpr double initial_Erad = 1.0e-5;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = 0.3;	   // reduced speed of light
constexpr double kappa0 = 1.0e-10; // opacity
constexpr double rho = 1.0;

template <> struct quokka::EOS_Traits<StreamingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StreamingProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = 1.0;
};

template <> struct RadSystem_Traits<StreamingProblem> {
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 0;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;
	const auto Egas0 = initial_Egas;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<StreamingProblem>::radEnergy_index) = Erad0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x3GasMomentum_index) = 0.;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<StreamingProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
							     int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
							     const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int j = 0;
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();

	if constexpr (direction == 0) {
		if (i < lo[0]) {
			// streaming inflow boundary
			const double Erad = 1.0;
			const double Frad = c * Erad;

			// x1 left side boundary (Marshak)
			consVar(i, j, k, RadSystem<StreamingProblem>::radEnergy_index) = Erad;
			consVar(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index) = Frad;
			consVar(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index) = 0.;
			consVar(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index) = 0.;
		} else if (i >= hi[0]) {
			// right-side boundary -- constant
			const double Erad = initial_Erad;
			consVar(i, j, k, RadSystem<StreamingProblem>::radEnergy_index) = Erad;
			consVar(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index) = 0;
			consVar(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index) = 0;
			consVar(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index) = 0;
		}
	} else {
		if (j < lo[1]) {
			// streaming inflow boundary
			const double Erad = 1.0;
			const double Frad = c * Erad;

			// x1 left side boundary (Marshak)
			consVar(i, j, k, RadSystem<StreamingProblem>::radEnergy_index) = Erad;
			consVar(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index) = 0.;
			consVar(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index) = Frad;
			consVar(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index) = 0.;
		} else if (j >= hi[1]) {
			// right-side boundary -- constant
			const double Erad = initial_Erad;
			consVar(i, j, k, RadSystem<StreamingProblem>::radEnergy_index) = Erad;
			consVar(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index) = 0;
			consVar(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index) = 0;
			consVar(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index) = 0;
		}
	}

	// gas boundary conditions are the same everywhere
	const double Egas = initial_Egas;
	consVar(i, j, k, RadSystem<StreamingProblem>::gasEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<StreamingProblem>::gasDensity_index) = rho;
	consVar(i, j, k, RadSystem<StreamingProblem>::gasInternalEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<StreamingProblem>::x1GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<StreamingProblem>::x2GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<StreamingProblem>::x3GasMomentum_index) = 0.;
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double dt_max = 1e-2;
	double tmax = 1.0;
	const int max_timesteps = 5000;

	// Boundary conditions
	constexpr int nvars = RadSystem<StreamingProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		// assert at compile time
		if constexpr (direction == 0) {
			BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // Dirichlet x1
			BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
			BCs_cc[n].setLo(1, amrex::BCType::int_dir);  // periodic
			BCs_cc[n].setHi(1, amrex::BCType::int_dir);
		} else {
			BCs_cc[n].setLo(0, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(0, amrex::BCType::int_dir);
			BCs_cc[n].setLo(1, amrex::BCType::ext_dir);  // Dirichlet x1
			BCs_cc[n].setHi(1, amrex::BCType::foextrap); // extrapolate x1
		}
	}

	// Problem initialization
	QuokkaSimulation<StreamingProblem> sim(BCs_cc);

	// read tmax from inputs file
	amrex::ParmParse pp; // NOLINT
	pp.query("max_time", tmax);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;
	sim.chat_over_c_ = chat / c;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), direction, 0.0);
	const int nx = static_cast<int>(position.size());

	// compute error norm
	std::vector<double> erad(nx);
	std::vector<double> erad_exact(nx);
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		erad_exact.at(i) = (x <= chat * tmax) ? 1.0 : 0.0;
		erad.at(i) = values.at(RadSystem<StreamingProblem>::radEnergy_index)[i];
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(erad[i] - erad_exact[i]);
		sol_norm += std::abs(erad_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.05;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	matplotlibcpp::ylim(-0.1, 1.1);

	std::map<std::string, std::string> erad_args;
	std::map<std::string, std::string> erad_exact_args;
	erad_args["label"] = "numerical solution";
	erad_exact_args["label"] = "exact solution";
	erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, erad, erad_args);
	matplotlibcpp::plot(xs, erad_exact, erad_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	if constexpr (direction == 0) {
		matplotlibcpp::save("./radiation_streaming_x.pdf");
	} else {
		matplotlibcpp::save("./radiation_streaming_y.pdf");
	}
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}
