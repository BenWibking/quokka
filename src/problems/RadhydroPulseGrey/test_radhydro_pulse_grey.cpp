/// \file test_radhydro_pulse_grey.cpp
/// \brief Defines a test problem for radiation in the diffusion regime with advection in medium with variable opacity under grey approximation.
///

#include "test_radhydro_pulse_grey.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "util/fextract.hpp"

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization
struct AdvPulseProblem {
};

constexpr double T0 = 1.0e7; // K (temperature)
constexpr double T1 = 2.0e7; // K (temperature)
constexpr double rho0 = 1.2; // g cm^-3 (matter density)
constexpr double a_rad = C::a_rad;
constexpr double width = 24.0; // cm, width of the pulse
constexpr double erad_floor = a_rad * T0 * T0 * T0 * T0 * 1.0e-10;
constexpr double mu = 2.33 * C::m_u;
constexpr double k_B = C::k_B;

// Default parameters: static diffusion, tau = 2e3, beta = 3e-5, beta tau = 6e-2
AMREX_GPU_MANAGED double kappa0 = 100.;	 // NOLINT
AMREX_GPU_MANAGED double v0_adv = 1.0e6; // NOLINT
// AMREX_GPU_MANAGED double max_time = 4.8e-5; // max_time = 2.0 * width / v1;

// dynamic diffusion: tau = 2e4, beta = 3e-3, beta tau = 60
// constexpr double kappa0 = 1000.; // cm^2 g^-1
// constexpr double v0_adv = 1.0e8;    // advecting pulse
// constexpr double max_time = 1.2e-4; // max_time = 2.0 * width / v1;

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};
template <> struct quokka::EOS_Traits<AdvPulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 1;
};
template <> struct RadSystem_Traits<AdvPulseProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 2;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};
template <> struct Physics_Traits<AdvPulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

AMREX_GPU_HOST_DEVICE
auto compute_initial_Tgas(const double x) -> double
{
	// compute temperature profile for Gaussian radiation pulse
	const double sigma = width;
	return T0 + (T1 - T0) * std::exp(-x * x / (2.0 * sigma * sigma));
}

AMREX_GPU_HOST_DEVICE
auto compute_exact_rho(const double x) -> double
{
	// compute density profile for Gaussian radiation pulse
	auto T = compute_initial_Tgas(x);
	return rho0 * T0 / T + (a_rad * mu / 3. / k_B) * (std::pow(T0, 4) / T - std::pow(T, 3));
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
{
	const double sigma = 3063.96 * std::pow(Tgas / T0, -3.5);
	return sigma / rho;
}
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
{
	const double sigma = 3063.96 * std::pow(Tgas / T0, -3.5);
	return sigma / rho;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	const double sigma = 101.248 * std::pow(Tgas / T0, -3.5);
	return sigma / rho;
}
template <> AMREX_GPU_HOST_DEVICE auto RadSystem<AdvPulseProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	const double sigma = 101.248 * std::pow(Tgas / T0, -3.5);
	return sigma / rho;
}

template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		const double Trad = compute_initial_Tgas(x - x0);
		const double Erad = a_rad * std::pow(Trad, 4);
		const double rho = compute_exact_rho(x - x0);
		const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho, Trad);

		state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}
template <> void QuokkaSimulation<AdvPulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		const double Trad = compute_initial_Tgas(x - x0);
		const double Erad = a_rad * std::pow(Trad, 4);
		const double rho = compute_exact_rho(x - x0);
		const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho, Trad);
		const double v0 = v0_adv;

		state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index) = 4. / 3. * v0 * Erad;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho * v0 * v0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = v0 * rho;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is a test of grey radiation diffusion plus advection by gas.
	// This makes this problem a stringent test of the radiation advection
	// in the diffusion limit under grey approximation.

	// Problem parameters
	const int64_t max_timesteps = 1e8;
	const double CFL_number = 0.8;
	// const int nx = 32;

	const double max_dt = 1e-3; // t_cr = 2 cm / cs = 7e-8 s

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem 1: non-advecting pulse

	// Problem initialization
	QuokkaSimulation<PulseProblem> sim(BCs_cc);

	double max_time = 4.8e-5;
	amrex::ParmParse pp; // NOLINT
	pp.query("kappa0", kappa0);
	pp.query("v0_adv", v0_adv);
	pp.query("max_time", max_time);

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

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> rhogas(nx);

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index)[i];
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values.at(RadSystem<PulseProblem>::gasDensity_index)[i];
		const auto v_t = values.at(RadSystem<PulseProblem>::x1GasMomentum_index)[i] / rho_t;
		const auto Egas = values.at(RadSystem<PulseProblem>::gasInternalEnergy_index)[i];
		rhogas.at(i) = rho_t;
		Trad.at(i) = Trad_t;
		Tgas.at(i) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho_t, Egas);
		Vgas.at(i) = 1e-5 * v_t;
	}
	// END OF PROBLEM 1

	// Problem 2: advecting radiation

	// Problem initialization
	QuokkaSimulation<AdvPulseProblem> sim2(BCs_cc);

	sim2.radiationReconstructionOrder_ = 3; // PPM
	sim2.stopTime_ = max_time;
	sim2.radiationCflNumber_ = CFL_number;
	sim2.maxDt_ = max_dt;
	sim2.maxTimesteps_ = max_timesteps;
	sim2.plotfileInterval_ = -1;

	// initialize
	sim2.setInitialConditions();

	// evolve
	sim2.evolve();

	// read output variables
	auto [position2, values2] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0.0);
	prob_lo = sim2.geom[0].ProbLoArray();
	prob_hi = sim2.geom[0].ProbHiArray();
	// compute the pixel size
	const double dx = (prob_hi[0] - prob_lo[0]) / static_cast<double>(nx);
	const double move = v0_adv * sim2.tNew_[0];
	const int n_p = static_cast<int>(move / dx);
	const int half = static_cast<int>(nx / 2.0);
	const double drift = move - static_cast<double>(n_p) * dx;
	const int shift = n_p - static_cast<int>((n_p + half) / nx) * nx;

	std::vector<double> xs2(nx);
	std::vector<double> Trad2(nx);
	std::vector<double> Tgas2(nx);
	std::vector<double> Vgas2(nx);
	std::vector<double> rhogas2(nx);

	for (int i = 0; i < nx; ++i) {
		int index_ = 0;
		if (shift >= 0) {
			if (i < shift) {
				index_ = nx - shift + i;
			} else {
				index_ = i - shift;
			}
		} else {
			if (i <= nx - 1 + shift) {
				index_ = i - shift;
			} else {
				index_ = i - (nx + shift);
			}
		}
		const amrex::Real x = position2[i];
		const auto Erad_t = values2.at(RadSystem<PulseProblem>::radEnergy_index)[i];
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values2.at(RadSystem<PulseProblem>::gasDensity_index)[i];
		const auto v_t = values2.at(RadSystem<PulseProblem>::x1GasMomentum_index)[i] / rho_t;
		const auto Egas = values2.at(RadSystem<PulseProblem>::gasInternalEnergy_index)[i];
		xs2.at(i) = x - drift;
		rhogas2.at(index_) = rho_t;
		Trad2.at(index_) = Trad_t;
		Tgas2.at(index_) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho_t, Egas);
		Vgas2.at(index_) = 1e-5 * (v_t - v0_adv);
	}
	// END OF PROBLEM 2

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	// double Tmax = 0.;
	for (size_t i = 0; i < xs2.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Trad[i]);
		err_norm += std::abs(Trad2[i] - Trad[i]);
		err_norm += std::abs(Tgas2[i] - Trad[i]);
		sol_norm += std::abs(Trad[i]) * 3.0;
		// Tmax = std::max(Tmax, Tgas2[i]);
	}
	// const double Tmax_tol = 1.37e7;
	const double error_tol = 8e-3;
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	// symmetry check
	double symm_err = 0.;
	double symm_norm = 0.;
	const double symm_err_tol = 0.02;
	for (size_t i = 0; i < xs2.size(); ++i) {
		symm_err += std::abs(Tgas2[i] - Tgas2[xs2.size() - 1 - i]);
		symm_norm += std::abs(Tgas2[i]);
	}
	const double symm_rel_error = symm_err / symm_norm;
	amrex::Print() << "Symmetry L1 error norm = " << symm_rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	Trad_args["label"] = "Trad (nonadvecting)";
	Trad_args["linestyle"] = "-.";
	Tgas_args["label"] = "Tgas (nonadvecting)";
	Tgas_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	Trad_args["label"] = "Trad (advecting)";
	Tgas_args["label"] = "Tgas (advecting)";
	matplotlibcpp::plot(xs2, Trad2, Trad_args);
	matplotlibcpp::plot(xs2, Tgas2, Tgas_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (K)");
	matplotlibcpp::ylim(0.98e7, 2.02e7);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim2.tNew_[0]));
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./radhydro_pulse_temperature_greynew.pdf");
	// save to file with tNew_[0] in the name
	matplotlibcpp::save(fmt::format("./radhydro_pulse_grey_temperature.pdf", sim2.tNew_[0]));

	// plot gas density profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> rho_args;
	rho_args["label"] = "gas density (non-advecting)";
	rho_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, rhogas, rho_args);
	rho_args["label"] = "gas density (advecting))";
	matplotlibcpp::plot(xs2, rhogas2, rho_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("density (g cm^-3)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_grey_density.pdf");

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "gas velocity (non-advecting)";
	vgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Vgas, vgas_args);
	vgas_args["label"] = "gas velocity (advecting)";
	matplotlibcpp::plot(xs2, Vgas2, vgas_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("velocity (km s^-1)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_grey_velocity.pdf");

#endif

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error) || (symm_rel_error > symm_err_tol) || std::isnan(symm_rel_error)) {
		status = 1;
	}
	return status;
}
