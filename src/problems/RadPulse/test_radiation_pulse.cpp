//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_pulse.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include "test_radiation_pulse.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "util/fextract.hpp"

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double kappa0 = 1.0e5;  // cm^-1 (opacity at temperature T0)
constexpr double T0 = 1.0;	  // K (temperature)
constexpr double rho0 = 1.0;	  // g cm^-3 (matter density)
constexpr double a_rad = 4.0e-10; // radiation constant == 4sigma_SB/c (dimensionless)
constexpr double c = 1.0e8;	  // speed of light (dimensionless)
constexpr double chat = 1.0e7;
constexpr double erad_floor = a_rad * (1.0e-10);

// constexpr double Lx = 1.0; // dimensionless length
constexpr double initial_time = 1.0e-8;

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = (2. / 3.);
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = a_rad;
};

AMREX_GPU_HOST_DEVICE
auto compute_exact_Trad(const double x, const double t) -> double
{
	// compute exact solution for Gaussian radiation pulse
	// 		assuming diffusion approximation
	const double sigma = 0.025;
	const double D = 4.0 * c * a_rad * std::pow(T0, 3) / (3.0 * kappa0);
	const double width_sq = (sigma * sigma + D * t);
	const double normfac = 1.0 / (2.0 * std::sqrt(M_PI * width_sq));
	return 0.5 * normfac * std::exp(-(x * x) / (4.0 * width_sq));
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
{
	auto kappa = (kappa0 / rho) * std::max(std::pow(Tgas / T0, 3), 1.0);
	return kappa;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		const double Trad = compute_exact_Trad(x - x0, initial_time);
		const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, Trad);

		state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = erad_floor;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is a *linear* radiation diffusion problem, i.e.
	// parameters are chosen such that the radiation and gas temperatures
	// should be near equilibrium, and the opacity is chosen to go as
	// T^3, such that the radiation diffusion equation becomes linear in T.

	// This makes this problem a stringent test of the asymptotic-
	// preserving property of the computational method, since the
	// optical depth per cell at the peak of the temperature profile is
	// of order 10^5.

	// Problem parameters
	const int max_timesteps = 1e5;
	const double CFL_number = 0.8;
	// const int nx = 32;

	const double max_dt = 1e-3;	// dimensionless time
	const double max_time = 1.0e-4; // dimensionless time

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::foextrap); // extrapolate
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = max_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();
	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> T_initial(nx);
	std::vector<double> xs_exact;
	std::vector<double> Trad_exact;
	std::vector<double> Erad_exact;

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index)[i];
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		Erad.at(i) = Erad_t;
		Trad.at(i) = Trad_t;
		Egas.at(i) = values.at(RadSystem<PulseProblem>::gasEnergy_index)[i];
		Tgas.at(i) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho0, Egas.at(i));

		auto Trad_val = compute_exact_Trad(x - x0, initial_time + sim.tNew_[0]);
		auto Erad_val = a_rad * std::pow(Trad_val, 4);
		xs_exact.push_back(x);
		Trad_exact.push_back(Trad_val);
		Erad_exact.push_back(Erad_val);
	}

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < xs.size(); ++i) {
		err_norm += std::abs(Trad[i] - Trad_exact[i]);
		sol_norm += std::abs(Trad_exact[i]);
	}
	const double error_tol = 0.01;
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();

	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Tinit_args;
	std::map<std::string, std::string> Trad_exact_args;
	Trad_args["label"] = "radiation temperature";
	Trad_args["linestyle"] = "-.";
	Tgas_args["label"] = "gas temperature";
	Tgas_args["linestyle"] = "--";
	// Tinit_args["label"] = "initial temperature";
	// Tinit_args["color"] = "grey";
	Trad_exact_args["label"] = "radiation temperature (exact)";
	Trad_exact_args["linestyle"] = ":";

	// matplotlibcpp::plot(xs, T_initial, Tinit_args);
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::plot(xs, Trad_exact, Trad_exact_args);

	matplotlibcpp::xlabel("length x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(std::format("time ct = {:.4g}", initial_time + sim.tNew_[0] * c));
	matplotlibcpp::save("./radiation_pulse_temperature.pdf");
#endif

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
