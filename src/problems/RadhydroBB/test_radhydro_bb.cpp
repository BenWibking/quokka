/// \file test_radhydro_bb.cpp
/// \brief Defines a test problem for blackbody spectrum in a uniform advecting medium.
///

#include <cmath>
#include <unordered_map>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"

#include "AMReX_BLassert.H"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"

#include "AMReX_Print.H"
#include "physics_info.hpp"
#include "test_radhydro_bb.hpp"

static constexpr bool export_csv = true;

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int n_groups_ = 4;
// constexpr int n_groups_ = 8;
// constexpr int n_groups_ = 16;
// constexpr int n_groups_ = 64;

constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_ = []() constexpr {
	if constexpr (n_groups_ == 1) {
		return amrex::GpuArray<double, 2>{0.0, inf};
	} else if constexpr (n_groups_ == 4) {
		return amrex::GpuArray<double, 5>{1.00000000e-03, 1.77827941e-02, 3.16227766e-01, 5.62341325e+00, 1.00000000e+02};
	} else if constexpr (n_groups_ == 8) {
		return amrex::GpuArray<double, 9>{1.00000000e-03, 4.21696503e-03, 1.77827941e-02, 7.49894209e-02, 3.16227766e-01,
						  1.33352143e+00, 5.62341325e+00, 2.37137371e+01, 1.00000000e+02};
	} else if constexpr (n_groups_ == 16) {
		return amrex::GpuArray<double, 17>{1.00000000e-03, 2.05352503e-03, 4.21696503e-03, 8.65964323e-03, 1.77827941e-02, 3.65174127e-02,
						   7.49894209e-02, 1.53992653e-01, 3.16227766e-01, 6.49381632e-01, 1.33352143e+00, 2.73841963e+00,
						   5.62341325e+00, 1.15478198e+01, 2.37137371e+01, 4.86967525e+01, 1.00000000e+02};
	} else if constexpr (n_groups_ == 64) {
		return amrex::GpuArray<double, 65>{
		    1.00000000e-03, 1.19708503e-03, 1.43301257e-03, 1.71543790e-03, 2.05352503e-03, 2.45824407e-03, 2.94272718e-03, 3.52269465e-03,
		    4.21696503e-03, 5.04806572e-03, 6.04296390e-03, 7.23394163e-03, 8.65964323e-03, 1.03663293e-02, 1.24093776e-02, 1.48550802e-02,
		    1.77827941e-02, 2.12875166e-02, 2.54829675e-02, 3.05052789e-02, 3.65174127e-02, 4.37144481e-02, 5.23299115e-02, 6.26433537e-02,
		    7.49894209e-02, 8.97687132e-02, 1.07460783e-01, 1.28639694e-01, 1.53992653e-01, 1.84342299e-01, 2.20673407e-01, 2.64164832e-01,
		    3.16227766e-01, 3.78551525e-01, 4.53158364e-01, 5.42469094e-01, 6.49381632e-01, 7.77365030e-01, 9.30572041e-01, 1.11397386e+00,
		    1.33352143e+00, 1.59633854e+00, 1.91095297e+00, 2.28757320e+00, 2.73841963e+00, 3.27812115e+00, 3.92418976e+00, 4.69758882e+00,
		    5.62341325e+00, 6.73170382e+00, 8.05842188e+00, 9.64661620e+00, 1.15478198e+01, 1.38237223e+01, 1.65481710e+01, 1.98095678e+01,
		    2.37137371e+01, 2.83873596e+01, 3.39820833e+01, 4.06794432e+01, 4.86967525e+01, 5.82941535e+01, 6.97830585e+01, 8.35362547e+01,
		    1.00000000e+02};
	}
}();

constexpr double c = 1.0e8;
// model 0
// constexpr int beta_order_ = 1; // order of beta in the radiation four-force
// constexpr double v0 = 1e-4 * c;
// constexpr double kappa0 = 1.0e4; // dx = 1, tau = kappa0 * dx = 1e4
// constexpr double chat = 1.0e7;
// model 1
// constexpr int beta_order_ = 1; // order of beta in the radiation four-force
// constexpr double v0 = 1e-4 * c;
// constexpr double kappa0 = 1.0e4; // dx = 1, tau = kappa0 * dx = 1e4
// constexpr double chat = 1.0e8;
// model 2
// constexpr int beta_order_ = 1; // order of beta in the radiation four-force
// constexpr double v0 = 1e-2 * c;
// constexpr double kappa0 = 1.0e5;
// constexpr double chat = 1.0e8;
// model 3
constexpr int beta_order_ = 1; // order of beta in the radiation four-force
// constexpr double v0 = 0.0;
// constexpr double v0 = 1e-2 * c;
// constexpr double v0 = 0.3 * c;
constexpr double v0 = 0.001 * c;
constexpr double kappa0 = 1.0e5;
constexpr double chat = c;

constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.0;
constexpr double k_B = 1.0;

constexpr double nu_unit = 1.0;
constexpr double T_equilibrium = 0.768032502191;

// static diffusion, beta = 1e-4, tau_cell = kappa0 * dx = 100, beta tau_cell = 1e-2
// constexpr double kappa0 = 100.; // cm^2 g^-1
// constexpr double v0 = 1.0e-4 * c; // advecting pulse

// dynamic diffusion, beta = 1e-3, tau = kappa0 * dx = 1e5, beta tau = 100
// constexpr double max_time = 10.0 / v0;
// constexpr double max_time = 1000.0 / (c * rho0 * kappa0); // dt >> 1 / (c * chi)
constexpr double max_time = 10.0 / (1e-2 * c);

constexpr double erad_floor = a_rad * 1e-30;

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<PulseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = a_rad;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = beta_order_;
	static constexpr double energy_unit = nu_unit;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<PulseProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							      const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
	}
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[1][i] = kappa0;
	}
	return exponents_and_values;
}

AMREX_GPU_HOST_DEVICE
auto compute_exact_bb(const double nu, const double T) -> double
{
	double const x = nu_unit * nu / (k_B * T);
	double const coeff = nu_unit / (k_B * T);
	double const planck_integral = std::pow(x, 3) / (std::exp(x) - 1.0);
	return coeff * planck_integral / (std::pow(PI, 4) / 15.0) * (a_rad * std::pow(T, 4));
}

template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, T0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < n_groups_; ++g) {
			state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = erad_floor;
			state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = v0 * rho0;
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
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;
	const double CFL_number_rad = 8.0;

	const double max_dt = 1.0;

	// Boundary conditions
	constexpr int nvars = RadSystem<PulseProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<PulseProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number_rad;
	sim.cflNumber_ = CFL_number_gas;
	sim.maxDt_ = max_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;
	sim.chat_over_c_ = chat / c;

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
	std::vector<double> Erad(nx);
	std::vector<double> Egas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> Vgas_exact(nx);
	std::vector<double> rhogas(nx);
	std::vector<double> Trad_exact{};
	std::vector<double> Tgas_exact{};
	std::vector<double> Erad_exact{};

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		// const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index)[i];
		double Erad_t = 0.0;
		for (int g = 0; g < n_groups_; ++g) {
			Erad_t += values.at(RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values.at(RadSystem<PulseProblem>::gasDensity_index)[i];
		const auto v_t = values.at(RadSystem<PulseProblem>::x1GasMomentum_index)[i] / rho_t;
		rhogas.at(i) = rho_t;
		Erad.at(i) = Erad_t;
		Trad.at(i) = Trad_t / T0;
		Egas.at(i) = values.at(RadSystem<PulseProblem>::gasInternalEnergy_index)[i];
		Tgas.at(i) = quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho_t, Egas.at(i)) / T0;
		Tgas_exact.push_back(1.0);
		Vgas.at(i) = v_t;
		Vgas_exact.at(i) = v0;

		auto const Erad_val = a_rad * std::pow(T0, 4);
		Trad_exact.push_back(T_equilibrium);
		Erad_exact.push_back(Erad_val);
	}

	// compute spectrum
	std::vector<double> E_r{};	// radiation energy density
	std::vector<double> F_r{};	// flux density
	std::vector<double> E_r_spec{}; // specific radiation energy density
	std::vector<double> F_r_spec{}; // specific flux density
	std::vector<double> bin_center{};
	int const ii = 3; // a random grid
	for (int g = 0; g < n_groups_; ++g) {
		// bin_center.push_back(std::sqrt(rad_boundaries_[g] * rad_boundaries_[g + 1]));
		bin_center.push_back(rad_boundaries_[g]);
		const auto Erad_t = values.at(RadSystem<PulseProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[ii];
		const auto Frad_t = values.at(RadSystem<PulseProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g)[ii];
		const double bin_width = rad_boundaries_[g + 1] - rad_boundaries_[g];
		E_r.push_back(Erad_t);
		F_r.push_back(Frad_t);
		F_r_spec.push_back(Frad_t / bin_width);
		E_r_spec.push_back(Erad_t / bin_width);
	}

	// Read in exact solution
	std::vector<double> nu_exact;
	std::vector<double> F_read;
	std::vector<double> E_read;

	std::string const filename = "../extern/Doppler-spectrum/exact_flux_density.csv";
	int const nbins = 1024;
	int const bins_per_group = nbins / n_groups_;
	std::ifstream fstream(filename, std::ios::in);
	AMREX_ASSERT(fstream.is_open());

	double err_norm = 0.;
	double sol_norm = 0.;

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values_line;
		std::string value;

		while (std::getline(iss, value, ',')) {
			values_line.push_back(std::stod(value));
		}
		double const nu_val = values_line.at(0);  // dimensionless
		double const Fnu_val = values_line.at(1); // dimensionless
		nu_exact.push_back(nu_val);
		F_read.push_back(Fnu_val);
		double const enu = compute_exact_bb(nu_val, T_equilibrium);
		E_read.push_back(enu);
	}
	// insert a dummy breakpoint
	int aa = 0;
	aa += 1;
	std::cout << aa << std::endl;

	// assert nu_exact[0] = 0.001 and nu_exact[-1] = 100
	AMREX_ASSERT(nu_exact[0] == 0.001);
	AMREX_ASSERT(nu_exact.back() == 100.0);
	AMREX_ASSERT(nu_exact.size() == nbins + 1);
	// compute group-integrated exact solution
	std::vector<double> Fnu_exact(n_groups_);
	std::vector<double> Enu_exact(n_groups_);
	int count = 0;
	double sum_F = 0.0;
	double sum_E = 0.0;
	for (int i = 0; i < nbins; ++i) {
		sum_F += F_read[i] * (nu_exact[i + 1] - nu_exact[i]);
		sum_E += E_read[i] * (nu_exact[i + 1] - nu_exact[i]);
		if ((i + 1) % bins_per_group == 0) {
			Fnu_exact[count] = sum_F / (rad_boundaries_[count + 1] - rad_boundaries_[count]);
			Enu_exact[count] = sum_E / (rad_boundaries_[count + 1] - rad_boundaries_[count]);
			sum_F = 0.0;
			sum_E = 0.0;
			count++;
		}
	}

	// compute error norm

	// compare temperature with T_equilibrium
	double err_norm_T = 0.;
	double sol_norm_T = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm_T += std::abs(Trad[i] - Trad_exact[i]);
		sol_norm_T += std::abs(Trad_exact[i]);
	}
	const double rel_error_T = err_norm_T / sol_norm_T;
	amrex::Print() << "Relative L1 error norm for T_gas = " << rel_error_T << std::endl;

	for (int g = 0; g < n_groups_; ++g) {
		err_norm += std::abs(Fnu_exact[g] - F_r_spec[g]);
		sol_norm += std::abs(Fnu_exact[g]);
	}

	const double error_tol = 0.1;
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Tradexact_args;
	Trad_args["label"] = "T_rad (numerical)";
	Trad_args["linestyle"] = "-";
	Tradexact_args["label"] = "T (exact)";
	Tradexact_args["linestyle"] = "--";
	Tgas_args["label"] = "T_gas (numerical)";
	Tgas_args["linestyle"] = "-";
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Trad_exact, Tradexact_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	matplotlibcpp::xlabel("x (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::ylim(0.0, 1.0);
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	// if constexpr (beta_order_ == 1) {
	// 	matplotlibcpp::ylim(1.0 - 1.0e-7, 1.0 + 1.0e-7);
	// }
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./adv_temp.pdf");
	// save to adv_temp_{n_groups_}bins.pdf
	matplotlibcpp::save(fmt::format("./adv_temp_{}_bins.pdf", n_groups_));

	// plot spectrum
	matplotlibcpp::clf();
	std::unordered_map<std::string, std::string> spec_args;
	spec_args["label"] = "spectrum";
	spec_args["color"] = "C0";
	matplotlibcpp::scatter(bin_center, E_r_spec, 10.0, spec_args);
	std::map<std::string, std::string> spec_exact_args;
	spec_exact_args["label"] = "spectrum (exact)";
	spec_exact_args["linestyle"] = "-";
	spec_exact_args["color"] = "C1";
	matplotlibcpp::plot(bin_center, Enu_exact, spec_exact_args);
	// log-log
	matplotlibcpp::xscale("log");
	matplotlibcpp::yscale("log");
	matplotlibcpp::ylim(1.0e-8, 1.0e0);
	matplotlibcpp::xlabel("frequency (dimensionless)");
	matplotlibcpp::ylabel("spectrum density (dimensionless)");
	// matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./adv_spectrum.pdf");
	// save to adv_spectrum_{n_groups_}bins.pdf
	matplotlibcpp::save(fmt::format("./adv_spectrum_{}_bins.pdf", n_groups_));

	// plot flux spectrum
	matplotlibcpp::clf();
	std::unordered_map<std::string, std::string> Frad_args;
	Frad_args["label"] = "flux (exact)";
	Frad_args["color"] = "C0";
	matplotlibcpp::scatter(bin_center, Fnu_exact, 10.0, Frad_args);
	std::map<std::string, std::string> Frad_exact_args;
	Frad_exact_args["label"] = "flux (numerical)";
	Frad_exact_args["linestyle"] = "-";
	Frad_exact_args["color"] = "C1";
	matplotlibcpp::plot(bin_center, F_r_spec, Frad_exact_args);
	// log-log
	matplotlibcpp::xscale("log");
	matplotlibcpp::yscale("log");
	matplotlibcpp::ylim(1.0e-3, 1.0e5);
	matplotlibcpp::xlabel("frequency (dimensionless)");
	matplotlibcpp::ylabel("flux density (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("time ct = {:.4g}", sim.tNew_[0] * c));
	matplotlibcpp::tight_layout();
	// matplotlibcpp::save("./adv_flux_spectrum.pdf");
	// save to adv_flux_spectrum_{n_groups_}bins.pdf
	matplotlibcpp::save(fmt::format("./adv_flux_spectrum_{}_bins.pdf", n_groups_));
#endif

	if (export_csv) {
		std::ofstream file;
		file.open("adv_spectrum.csv");
		file << "nu_Left, E_r (erg/cm^3/Hz)\n";
		for (int g = 0; g < n_groups_; ++g) {
			file << std::scientific << std::setprecision(12) << rad_boundaries_[g] << "," << E_r_spec[g] << "\n";
		}
		file.close();

		file.open("adv_flux_spectrum.csv");
		file << "nu_Left, F_r (erg/cm^2/s/Hz)\n";
		for (int g = 0; g < n_groups_; ++g) {
			file << std::scientific << std::setprecision(12) << rad_boundaries_[g] << "," << F_r_spec[g] << "\n";
		}
		file.close();

		file.open("adv_temp.csv");
		file << "xs,Tgas,Trad\n";
		for (size_t i = 0; i < xs.size(); ++i) {
			file << std::scientific << std::setprecision(12) << xs[i] << "," << Tgas[i] << "," << Trad[i] << "\n";
		}
		file.close();
	}

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error) || (rel_error_T > error_tol) || std::isnan(rel_error_T)) {
		status = 1;
	}
	return status;
}
