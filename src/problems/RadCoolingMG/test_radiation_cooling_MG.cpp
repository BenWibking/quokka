/// \file test_radiation_cooling_MG.cpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

#include "test_radiation_cooling_MG.hpp"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_Box.H"
#include "AMReX_Extension.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "fmt/format.h"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"
#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct CoolingProblem {
};

constexpr double Erad_floor_ = 1.0e-20;
constexpr double c = 1.0; // speed of light
constexpr double chat = 1.0;
constexpr double chi0 = 1.0e-8;
constexpr double rho0 = 1.0;
constexpr double T0 = 1.0;
constexpr double mu = 1.0;
constexpr double k_B = 1.0;
constexpr double Tgas_exact_ = 0.768032502191;
constexpr double a_rad_ = 1.0e30;

constexpr int n_groups_ = 2;

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
	static constexpr int nGroups = n_groups_; // number of radiation groups
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
	static constexpr double energy_unit = 1.;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries{1.0e-3, 3.0, 1.0e3};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <> void QuokkaSimulation<CoolingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Egas0 = 1.5 * (rho0 / mu) * k_B * T0;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < n_groups_; ++g) {
			state_cc(i, j, k, RadSystem<CoolingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_floor_;
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

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<CoolingProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, n_groups_ + 1> /*rad_boundaries*/, const double rho,
								const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, n_groups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, n_groups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < n_groups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = chi0 / rho;
	}
	return exponents_and_values;
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
	amrex::Print() << "T = " << Tgas[0] << "\n";
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << "\n";

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	matplotlibcpp::ylim(-0.1, 1.1);

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
	amrex::Print() << "Finished."
		       << "\n";
	return status;
}
