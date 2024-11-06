/// \file test_radhydro_pulse_MG_const_kappa.cpp
/// \brief Defines a test problem for multigroup radiation in the diffusion regime with advection by gas, running
/// with PPL_opacity_fixed_slope_spectrum opacity model.
///

#include "test_radhydro_pulse_MG_const_kappa.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "util/fextract.hpp"

// Single-group problem
struct SGProblem {
}; // dummy type to allow compile-type polymorphism via template specialization
// Multi-group problem
struct MGproblem {
};

// constexpr int n_groups_ = 1;
// constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{0., inf};
// constexpr int n_groups_ = 2;
// constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1e15, 1e17, 1e19};
constexpr int n_groups_ = 4;
constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1e15, 1e16, 1e17, 1e18, 1e19};
// constexpr int n_groups_ = 64;
// constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_{1.00000000e+15, 1.15478198e+15, 1.33352143e+15, 1.53992653e+15,
//        1.77827941e+15, 2.05352503e+15, 2.37137371e+15, 2.73841963e+15,
//        3.16227766e+15, 3.65174127e+15, 4.21696503e+15, 4.86967525e+15,
//        5.62341325e+15, 6.49381632e+15, 7.49894209e+15, 8.65964323e+15,
//        1.00000000e+16, 1.15478198e+16, 1.33352143e+16, 1.53992653e+16,
//        1.77827941e+16, 2.05352503e+16, 2.37137371e+16, 2.73841963e+16,
//        3.16227766e+16, 3.65174127e+16, 4.21696503e+16, 4.86967525e+16,
//        5.62341325e+16, 6.49381632e+16, 7.49894209e+16, 8.65964323e+16,
//        1.00000000e+17, 1.15478198e+17, 1.33352143e+17, 1.53992653e+17,
//        1.77827941e+17, 2.05352503e+17, 2.37137371e+17, 2.73841963e+17,
//        3.16227766e+17, 3.65174127e+17, 4.21696503e+17, 4.86967525e+17,
//        5.62341325e+17, 6.49381632e+17, 7.49894209e+17, 8.65964323e+17,
//        1.00000000e+18, 1.15478198e+18, 1.33352143e+18, 1.53992653e+18,
//        1.77827941e+18, 2.05352503e+18, 2.37137371e+18, 2.73841963e+18,
//        3.16227766e+18, 3.65174127e+18, 4.21696503e+18, 4.86967525e+18,
//        5.62341325e+18, 6.49381632e+18, 7.49894209e+18, 8.65964323e+18,
//        1.00000000e+19};

constexpr double kappa0 = 100.; // cm^2 g^-1
constexpr double T0 = 1.0e7;	// K (temperature)
constexpr double T1 = 2.0e7;	// K (temperature)
constexpr double rho0 = 1.2;	// g cm^-3 (matter density)
constexpr double a_rad = C::a_rad;
constexpr double c = C::c_light; // speed of light (cgs)
constexpr double chat = c;
constexpr double width = 24.0; // cm, width of the pulse
constexpr double Erad0 = a_rad * T0 * T0 * T0 * T0;
constexpr double erad_floor = Erad0 * 1.0e-14;
constexpr double mu = 2.33 * C::m_u;
constexpr double h_planck = C::hplanck;
constexpr double k_B = C::k_B;

// testing parameters:
// dynamic diffusion limit: tau = 2e3, beta = 7e-3, beta tau = 14
constexpr double v0 = 2.0e8;	       // advecting speed
constexpr double max_time = 4.8e-5;    // diffusion distance is about a few pulse width
constexpr int64_t max_timesteps = 1e2; // for fast testing

// dynamic diffusion: tau = 2e4, beta = 3e-3, beta tau = 60
// constexpr double kappa0 = 1000.; // cm^2 g^-1
// constexpr double v0 = 1.0e8;    // advecting pulse
// constexpr double max_time = 1.2e-4; // max_time = 2.0 * width / v1;

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

template <> struct quokka::EOS_Traits<SGProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<SGProblem> {
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

template <> struct RadSystem_Traits<SGProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 1;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SGProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real { return kappa0; }

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SGProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <> void QuokkaSimulation<SGProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
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
		const double rho = compute_exact_rho(x - x0);
		const double Egas = quokka::EOS<SGProblem>::ComputeEintFromTgas(rho, Trad);
		const double Erad = a_rad * Trad * Trad * Trad * Trad;

		state_cc(i, j, k, RadSystem<SGProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<SGProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<SGProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<SGProblem>::x3RadFlux_index) = 0;

		state_cc(i, j, k, RadSystem<SGProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SGProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<SGProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SGProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SGProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SGProblem>::x3GasMomentum_index) = 0.;
	});
}

template <> struct quokka::EOS_Traits<MGproblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<MGproblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<MGproblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = erad_floor;
	static constexpr double energy_unit = h_planck;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr int beta_order = 1;
	// static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_fixed_slope_spectrum;
	// static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_full_spectrum;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<MGproblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> const /* rad_boundaries */, const double /* rho */,
							   const double /* Tgas */) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<double, nGroups_ + 1> exponents{};
	amrex::GpuArray<double, nGroups_ + 1> kappa_lower{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents[g] = 0.;
		kappa_lower[g] = kappa0;
	}
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const exponents_and_values{exponents, kappa_lower};
	return exponents_and_values;
}

template <> void QuokkaSimulation<MGproblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	const auto radBoundaries_g = RadSystem_Traits<MGproblem>::radBoundaries;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		const double Trad = compute_initial_Tgas(x - x0);
		const double rho = compute_exact_rho(x - x0);
		const double Egas = quokka::EOS<MGproblem>::ComputeEintFromTgas(rho, Trad);

		auto Erad_g = RadSystem<MGproblem>::ComputeThermalRadiationMultiGroup(Trad, radBoundaries_g);

		for (int g = 0; g < Physics_Traits<MGproblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<MGproblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			state_cc(i, j, k, RadSystem<MGproblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 4. / 3. * v0 * Erad_g[g];
			state_cc(i, j, k, RadSystem<MGproblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<MGproblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}

		state_cc(i, j, k, RadSystem<MGproblem>::gasEnergy_index) = Egas + 0.5 * rho * v0 * v0;
		state_cc(i, j, k, RadSystem<MGproblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<MGproblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<MGproblem>::x1GasMomentum_index) = v0 * rho;
		state_cc(i, j, k, RadSystem<MGproblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<MGproblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is a test of radiation diffusion plus advection by gas.
	// This makes this problem a stringent test of the radiation advection
	// in the diffusion limit.

	// Problem parameters
	const double CFL_number = 0.8;
	// const int nx = 32;

	const double max_dt = 1e-3; // t_cr = 2 cm / cs = 7e-8 s

	// Problem 1: pulse with grey radiation

	// Boundary conditions
	constexpr int nvars = RadSystem<SGProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		// periodic boundary condition in the x-direction will not work
		BCs_cc[n].setLo(0, amrex::BCType::foextrap); // extrapolate
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<SGProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.cflNumber_ = CFL_number;
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

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> rhogas(nx);

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		// const auto Erad_t = values.at(RadSystem<SGProblem>::radEnergy_index)[i];
		double Erad_t = 0.0;
		for (int g = 0; g < Physics_Traits<SGProblem>::nGroups; ++g) {
			Erad_t += values.at(RadSystem<SGProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values.at(RadSystem<SGProblem>::gasDensity_index)[i];
		const auto v_t = values.at(RadSystem<SGProblem>::x1GasMomentum_index)[i] / rho_t;
		const auto Egas = values.at(RadSystem<SGProblem>::gasInternalEnergy_index)[i];
		rhogas.at(i) = rho_t;
		Trad.at(i) = Trad_t;
		Tgas.at(i) = quokka::EOS<SGProblem>::ComputeTgasFromEint(rho_t, Egas);
		Vgas.at(i) = 1e-5 * v_t;
	}
	// END OF PROBLEM 1

	// Problem 2: advecting pulse

	// Boundary conditions
	constexpr int nvars2 = RadSystem<MGproblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc2(nvars2);
	for (int n = 0; n < nvars2; ++n) {
		// periodic boundary condition in the x-direction will not work
		BCs_cc2[n].setLo(0, amrex::BCType::foextrap); // extrapolate
		BCs_cc2[n].setHi(0, amrex::BCType::foextrap);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc2[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc2[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<MGproblem> sim2(BCs_cc2);

	sim2.radiationReconstructionOrder_ = 3; // PPM
	sim2.stopTime_ = max_time;
	sim2.radiationCflNumber_ = CFL_number;
	sim2.cflNumber_ = CFL_number;
	sim2.maxDt_ = max_dt;
	sim2.maxTimesteps_ = max_timesteps;
	sim2.plotfileInterval_ = -1;

	// initialize
	sim2.setInitialConditions();

	// evolve
	sim2.evolve();

	// read output variables
	auto [position2, values2] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0.0);
	const int nx2 = static_cast<int>(position2.size());

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim2.geom[0].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim2.geom[0].ProbHiArray();
	// compute the pixel size
	const double dx = (prob_hi[0] - prob_lo[0]) / static_cast<double>(nx2);
	const double move = v0 * sim2.tNew_[0];
	const int n_p = static_cast<int>(move / dx);
	const int half = static_cast<int>(nx2 / 2.0);
	const double drift = move - static_cast<double>(n_p) * dx;
	const int shift = n_p - static_cast<int>((n_p + half) / nx2) * nx2;

	std::vector<double> xs2(nx2);
	std::vector<double> Trad2(nx2);
	std::vector<double> Tgas2(nx2);
	std::vector<double> Vgas2(nx2);
	std::vector<double> rhogas2(nx2);

	for (int i = 0; i < nx2; ++i) {
		int index_ = 0;
		if (shift >= 0) {
			if (i < shift) {
				index_ = nx2 - shift + i;
			} else {
				index_ = i - shift;
			}
		} else {
			if (i <= nx2 - 1 + shift) {
				index_ = i - shift;
			} else {
				index_ = i - (nx2 + shift);
			}
		}
		const amrex::Real x = position2[i];
		// const auto Erad_t = values2.at(RadSystem<MGproblem>::radEnergy_index)[i];
		double Erad_t = 0.0;
		for (int g = 0; g < Physics_Traits<MGproblem>::nGroups; ++g) {
			Erad_t += values2.at(RadSystem<MGproblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values2.at(RadSystem<MGproblem>::gasDensity_index)[i];
		const auto v_t = values2.at(RadSystem<MGproblem>::x1GasMomentum_index)[i] / rho_t;
		const auto Egas = values2.at(RadSystem<MGproblem>::gasInternalEnergy_index)[i];
		xs2.at(i) = x - drift;
		rhogas2.at(index_) = rho_t;
		Trad2.at(index_) = Trad_t;
		Tgas2.at(index_) = quokka::EOS<MGproblem>::ComputeTgasFromEint(rho_t, Egas);
		Vgas2.at(index_) = 1e-5 * (v_t - v0);
	}
	// END OF PROBLEM 2

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < xs2.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Trad[i]);
		err_norm += std::abs(Trad2[i] - Trad[i]);
		err_norm += std::abs(Tgas2[i] - Trad[i]);
		sol_norm += std::abs(Trad[i]) * 3.0;
	}
	const double error_tol = 0.006;
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	Trad_args["label"] = "Trad (non-advecting)";
	Trad_args["linestyle"] = "-.";
	Tgas_args["label"] = "Tgas (non-advecting)";
	Tgas_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	Trad_args["label"] = "Trad (advecting)";
	Tgas_args["label"] = "Tgas (advecting)";
	matplotlibcpp::plot(xs2, Trad2, Trad_args);
	matplotlibcpp::plot(xs2, Tgas2, Tgas_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (K)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_MG_temperature.pdf");

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
	matplotlibcpp::save("./radhydro_pulse_MG_density.pdf");

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
	matplotlibcpp::save("./radhydro_pulse_MG_velocity.pdf");

#endif

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
