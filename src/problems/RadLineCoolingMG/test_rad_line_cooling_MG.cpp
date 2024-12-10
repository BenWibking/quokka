/// \file test_rad_line_cooling_MG.cpp
/// \brief Defines a test problem for line cooling and cosmic-ray heating in a uniform medium.
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "physics_info.hpp"
#include "test_rad_line_cooling_MG.hpp"
#include "util/fextract.hpp"

static constexpr bool export_csv = true;

struct CoolingProblemMG {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int n_groups_ = 4;
constexpr amrex::GpuArray<double, 5> rad_boundaries_ = {1.00000000e-03, 1.77827941e-02, 3.16227766e-01, 5.62341325e+00, 1.00000000e+02};

constexpr double chat_over_c = 1.0;
constexpr double v0 = 0.0;
constexpr double kappa0 = 0.0;

constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.5; // mean molecular weight; so that C_V = 1.0
constexpr double C_V = 1.0;

constexpr double nu_unit = 1.0;
constexpr double Erad_bar = a_rad * T0 * T0 * T0 * T0;
constexpr double Erad_floor_ = a_rad * 1e-20;
constexpr double Erad_FUV = Erad_bar; // = 1.0

const double max_time = 10.0;
const int line_index = 0; // last group
const double cooling_rate = 0.1;
const double CR_heating_rate = 0.03;
const double PE_rate = 0.02;

template <> struct SimulationData<CoolingProblemMG> {
	std::vector<double> t_vec_;
	std::vector<double> Tgas_vec_;
	std::vector<double> Erad_line_vec_;
};

template <> struct quokka::EOS_Traits<CoolingProblemMG> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<CoolingProblemMG> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = a_rad;
};

template <> struct RadSystem_Traits<CoolingProblemMG> {
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = nu_unit;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
};

template <> struct ISM_Traits<CoolingProblemMG> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = 1;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = 1;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/,
											       amrex::Real const /*num_density*/) -> amrex::Real
{
	return PE_rate / Erad_FUV;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[line_index] = cooling_rate * temperature;
	return cooling;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineNetCoolingRateTempDerivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[line_index] = cooling_rate;
	return cooling;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblemMG>::DefineCosmicRayHeatingRate(amrex::Real const /*num_density*/) -> double
{
	return CR_heating_rate;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<CoolingProblemMG>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
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

template <> void QuokkaSimulation<CoolingProblemMG>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Egas = quokka::EOS<CoolingProblemMG>::ComputeEintFromTgas(rho0, T0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < n_groups_; ++g) {
			const auto Erad = g == n_groups_ - 1 ? Erad_FUV : Erad_floor_;
			state_cc(i, j, k, RadSystem<CoolingProblemMG>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad;
			state_cc(i, j, k, RadSystem<CoolingProblemMG>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			state_cc(i, j, k, RadSystem<CoolingProblemMG>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			state_cc(i, j, k, RadSystem<CoolingProblemMG>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}
		state_cc(i, j, k, RadSystem<CoolingProblemMG>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<CoolingProblemMG>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<CoolingProblemMG>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<CoolingProblemMG>::x1GasMomentum_index) = v0 * rho0;
		state_cc(i, j, k, RadSystem<CoolingProblemMG>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<CoolingProblemMG>::x3GasMomentum_index) = 0.;
	});
}

template <> void QuokkaSimulation<CoolingProblemMG>::computeAfterTimestep()
{
	auto [_, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5); // NOLINT [[maybe_unused]]

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);

		const amrex::Real Etot_i = values.at(RadSystem<CoolingProblemMG>::gasEnergy_index)[0];
		const amrex::Real x1GasMom = values.at(RadSystem<CoolingProblemMG>::x1GasMomentum_index)[0];
		const amrex::Real x2GasMom = values.at(RadSystem<CoolingProblemMG>::x2GasMomentum_index)[0];
		const amrex::Real x3GasMom = values.at(RadSystem<CoolingProblemMG>::x3GasMomentum_index)[0];
		const amrex::Real rho = values.at(RadSystem<CoolingProblemMG>::gasDensity_index)[0];
		const amrex::Real Egas_i = RadSystem<CoolingProblemMG>::ComputeEintFromEgas(rho, x1GasMom, x2GasMom, x3GasMom, Etot_i);
		userData_.Tgas_vec_.push_back(quokka::EOS<CoolingProblemMG>::ComputeTgasFromEint(rho, Egas_i));
		const double Erad_line_i = values.at(RadSystem<CoolingProblemMG>::radEnergy_index + Physics_NumVars::numRadVars * line_index)[0];
		userData_.Erad_line_vec_.push_back(Erad_line_i);
	}
}

auto problem_main() -> int
{
	// This problem is a test of photoelectric heating, line cooling, and cosmic-ray heating in a uniform medium with multigroup radiation. The gas/dust
	// opacity is set to zero, so that the radiation does not interact with matter. The initial conditions are set to a constant temperature and radiation
	// energy density Erad_FUV = 1. The gas cools at a rate of 0.1 per unit time, and is heated by cosmic rays at a rate of 0.03 per unit time. The
	// photoelectric heating rate is 0.02 * Erad_FUV per unit time. The exact solution is given by the following system of equations: dTgas/dt = -0.1 * Tgas
	// + 0.03 + 0.02 * Erad_FUV where Erad_FUV = 1.0 is constant over time.

	// Problem parameters
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;
	const double CFL_number_rad = 0.8;

	const double the_dt = 1.0e-2;

	// Boundary conditions
	constexpr int nvars = RadSystem<CoolingProblemMG>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<CoolingProblemMG> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number_rad;
	sim.cflNumber_ = CFL_number_gas;
	sim.initDt_ = the_dt;
	sim.maxDt_ = the_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;
	sim.chat_over_c_ = chat_over_c;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const bool is_coupled = sim.dustGasInteractionCoeff_ > 1.0;

	const auto t_end = sim.tNew_[0];

	const double heating_rate_ = ISM_Traits<CoolingProblemMG>::enable_photoelectric_heating ? PE_rate + CR_heating_rate : CR_heating_rate;

	// compute exact solution from t = 0 to t = t_end
	const double N_dt = 1000.;
	double t_exact = 0.0;
	std::vector<double> t_exact_vec{};
	std::vector<double> Tgas_exact_vec{};
	std::vector<double> Erad_line_exact_vec{};
	while (true) {
		const double Egas_exact_solution =
		    std::exp(-cooling_rate * t_exact) * (cooling_rate * T0 - heating_rate_ + heating_rate_ * std::exp(cooling_rate * t_exact)) / cooling_rate;
		const double T_exact_solution = Egas_exact_solution / C_V;
		Tgas_exact_vec.push_back(T_exact_solution);
		const double Erad_line_exact_solution = -(Egas_exact_solution - C_V * T0 - heating_rate_ * t_exact) * chat_over_c;
		Erad_line_exact_vec.push_back(Erad_line_exact_solution);
		t_exact_vec.push_back(t_exact);
		t_exact += t_end / N_dt;
		if (t_exact > 1.01 * t_end) {
			break; // note that we need the last time to be greater than t_end
		}
	}

	std::vector<double> &Tgas = sim.userData_.Tgas_vec_;
	std::vector<double> &Erad_line = sim.userData_.Erad_line_vec_;
	std::vector<double> &t = sim.userData_.t_vec_;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Tgas_args;
	Tgas_args["label"] = "T_gas (numerical)";
	Tgas_args["linestyle"] = "-";
	matplotlibcpp::plot(t, Tgas, Tgas_args);
	Tgas_args["label"] = "T_gas (exact)";
	Tgas_args["linestyle"] = "--";
	matplotlibcpp::plot(t_exact_vec, Tgas_exact_vec, Tgas_args);
	matplotlibcpp::xlabel("t (dimensionless)");
	matplotlibcpp::ylabel("temperature (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::ylim(-0.05, 2.05);
	matplotlibcpp::tight_layout();
	if (is_coupled) {
		matplotlibcpp::save("./rad_line_cooling_MG_coupled_temperature.pdf");
	} else {
		matplotlibcpp::save("./rad_line_cooling_MG_decoupled_temperature.pdf");
	}

	// plot Erad_line
	matplotlibcpp::clf();
	std::map<std::string, std::string> Erad_args;
	std::map<std::string, std::string> Erad_exact_args;
	Erad_args["label"] = "E_rad (numerical)";
	Erad_args["linestyle"] = "-";
	Erad_exact_args["label"] = "E_rad (exact)";
	Erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(t, Erad_line, Erad_args);
	matplotlibcpp::plot(t_exact_vec, Erad_line_exact_vec, Erad_exact_args);
	matplotlibcpp::xlabel("t (dimensionless)");
	matplotlibcpp::ylabel("radiation energy density (dimensionless)");
	matplotlibcpp::ylim(-0.05, 1.05);
	matplotlibcpp::legend();
	matplotlibcpp::tight_layout();
	if (is_coupled) {
		matplotlibcpp::save("./rad_line_cooling_MG_coupled_radiation_energy_density.pdf");
	} else {
		matplotlibcpp::save("./rad_line_cooling_MG_decoupled_radiation_energy_density.pdf");
	}
#endif

	std::vector<double> Tgas_interp(t.size());
	std::vector<double> Erad_line_interp(t.size());
	interpolate_arrays(t.data(), Tgas_interp.data(), static_cast<int>(t.size()), t_exact_vec.data(), Tgas_exact_vec.data(),
			   static_cast<int>(t_exact_vec.size()));
	interpolate_arrays(t.data(), Erad_line_interp.data(), static_cast<int>(t.size()), t_exact_vec.data(), Erad_line_exact_vec.data(),
			   static_cast<int>(t_exact_vec.size()));

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < t.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Tgas_interp[i]);
		err_norm += std::abs(Erad_line[i] - Erad_line_interp[i]);
		sol_norm += std::abs(Tgas_interp[i]) + std::abs(Erad_line_interp[i]);
	}
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	if (export_csv) {
		std::ofstream file;
		file.open("rad_line_cooling_temp.csv");
		file << "t,Tgas,Tgas_exact\n";
		for (size_t i = 0; i < t.size(); ++i) {
			file << std::scientific << std::setprecision(12) << t[i] << "," << Tgas[i] << "," << Tgas_exact_vec[i] << "\n";
		}
		file.close();

		file.open("rad_line_cooling_rad_energy_density.csv");
		file << "t,Erad_line,Erad_line_exact\n";
		for (size_t i = 0; i < t.size(); ++i) {
			file << std::scientific << std::setprecision(12) << t[i] << "," << Erad_line[i] << "," << Erad_line_exact_vec[i] << "\n";
		}
		file.close();
	}

	// exit
	int status = 0;
	const double error_tol = 0.0005;
	if (rel_error > error_tol) {
		status = 1;
	}
	return status;
}
