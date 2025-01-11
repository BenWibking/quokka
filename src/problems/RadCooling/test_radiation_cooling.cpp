//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_streaming.cpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

#include "test_radiation_cooling.hpp"
#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"

struct CoolingProblem {
};

constexpr double Erad_floor_ = 1.0e-20;
constexpr double c = 1.0; // speed of light
constexpr double chat = 1.0;
constexpr double chi0 = 1.0e5;
constexpr double rho0 = 1.0;
constexpr double T0 = 1.0;
constexpr double mu = 1.0;
constexpr double k_B = 1.0;
constexpr double Tgas_exact_ = 0.768032502191;
constexpr double a_rad_ = 1.0;

template <> struct quokka::EOS_Traits<CoolingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<CoolingProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = a_rad_;
};

template <> struct RadSystem_Traits<CoolingProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 0;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
{
	return chi0 / rho;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::ComputeFluxMeanOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
{
	return chi0 / rho;
}

template <> void QuokkaSimulation<CoolingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = Erad_floor_;
	const auto Egas0 = 1.5 * (rho0 / mu) * k_B * T0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < 1; ++g) {
			state_cc(i, j, k, RadSystem<CoolingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad0;
			state_cc(i, j, k, RadSystem<CoolingProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<CoolingProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<CoolingProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<CoolingProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<CoolingProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<CoolingProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	// const double dt_max = 1e-2;
	const double tmax = 10.0;
	const int max_timesteps = 5000;

	// Boundary conditions
	constexpr int nvars = RadSystem<CoolingProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<CoolingProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
	sim.radiationCflNumber_ = CFL_number;
	// sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// compute error norm
	std::vector<double> Tgas(nx);
	std::vector<double> Tgas_exact(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Trad_exact(nx);
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		Tgas_exact.at(i) = Tgas_exact_;
		Trad_exact.at(i) = Tgas_exact_;
		const auto Egas = values.at(RadSystem<CoolingProblem>::gasInternalEnergy_index)[i];
		const auto rho = values.at(RadSystem<CoolingProblem>::gasDensity_index)[i];
		Tgas.at(i) = quokka::EOS<CoolingProblem>::ComputeTgasFromEint(rho, Egas);
		double Erad = 0.0;
		for (int g = 0; g < Physics_Traits<CoolingProblem>::nGroups; ++g) {
			Erad += values.at(RadSystem<CoolingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		Trad.at(i) = std::pow(Erad / a_rad_, 1. / 4.);
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(Tgas[i] - Tgas_exact[i]);
		sol_norm += std::abs(Tgas_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 1.0e-4;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << "\n";

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	matplotlibcpp::ylim(0.0, 1.1);

	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Tgas_exact_args;
	Tgas_args["label"] = "numerical solution";
	Tgas_exact_args["label"] = "exact solution";
	Tgas_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::plot(xs, Tgas_exact, Tgas_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./radiation_cooling.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << "\n";
	return status;
}
