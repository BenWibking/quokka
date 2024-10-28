/// \file test_rad_line_cooling.cpp
/// \brief Defines a test problem for line cooling and cosmic-ray heating in a uniform medium.
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "physics_info.hpp"
#include "test_rad_line_cooling.hpp"
#include "util/fextract.hpp"

static constexpr bool export_csv = true;

struct CoolingProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

const double cooling_rate = 0.1;
const double CR_heating_rate = 0.03;

constexpr double c = 1.0;
constexpr double chat = c;
constexpr double v0 = 0.0;
constexpr double kappa0 = 0.0;

constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.5; // mean molecular weight; so that C_V = 1.0
constexpr double C_V = 1.0;
constexpr double k_B = 1.0;

constexpr double nu_unit = 1.0;
constexpr double erad_floor = a_rad * 1e-20;

const double max_time = 10.0;

template <> struct SimulationData<CoolingProblem> {
	std::vector<double> t_vec_;
	std::vector<double> Tgas_vec_;
	std::vector<double> Erad_vec_;
};

template <> struct quokka::EOS_Traits<CoolingProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<CoolingProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	// A custom unit system is used here to replicate a dimentionless unit system (c = k_B = a_rad = G = 1), for testing units conversion
	static constexpr UnitSystem unit_system = UnitSystem::CUSTOM;
	static constexpr double c_hat = chat;
	static constexpr double unit_length = 1.733039549e-33;
	static constexpr double unit_mass = 2.333695323e-05;
	static constexpr double unit_time = 5.780797690e-44;
	static constexpr double unit_temperature = 1.519155670e+32;
	// Equivalently, set
	// static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	// static constexpr double boltzmann_constant = k_B;
	// static constexpr double gravitational_constant = 1.0;
};

template <> struct RadSystem_Traits<CoolingProblem> {
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = nu_unit;
	// Equivalently, set
	// static constexpr double c_light = c;
	// static constexpr double radiation_constant = a_rad;
};

template <> struct ISM_Traits<CoolingProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr bool enable_linear_cooling_heating = true;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineNetCoolingRate(amrex::Real const temperature, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[0] = cooling_rate * temperature;
	return cooling;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineNetCoolingRateTempDerivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	cooling[0] = cooling_rate;
	return cooling;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::DefineCosmicRayHeatingRate(amrex::Real const /*num_density*/) -> double
{
	return CR_heating_rate;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<CoolingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CoolingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<CoolingProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
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

template <> void QuokkaSimulation<CoolingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Egas = quokka::EOS<CoolingProblem>::ComputeEintFromTgas(rho0, T0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<CoolingProblem>::radEnergy_index) = erad_floor;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x1RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x2RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x3RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<CoolingProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<CoolingProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<CoolingProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x1GasMomentum_index) = v0 * rho0;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<CoolingProblem>::x3GasMomentum_index) = 0.;
	});
}

template <> void QuokkaSimulation<CoolingProblem>::computeAfterTimestep()
{
	auto values = std::get<1>(fextract(state_new_cc_[0], Geom(0), 0, 0.5)); // NOLINT

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);

		const amrex::Real Etot_i = values.at(RadSystem<CoolingProblem>::gasEnergy_index)[0];
		const amrex::Real x1GasMom = values.at(RadSystem<CoolingProblem>::x1GasMomentum_index)[0];
		const amrex::Real x2GasMom = values.at(RadSystem<CoolingProblem>::x2GasMomentum_index)[0];
		const amrex::Real x3GasMom = values.at(RadSystem<CoolingProblem>::x3GasMomentum_index)[0];
		const amrex::Real rho = values.at(RadSystem<CoolingProblem>::gasDensity_index)[0];
		const amrex::Real Egas_i = RadSystem<CoolingProblem>::ComputeEintFromEgas(rho, x1GasMom, x2GasMom, x3GasMom, Etot_i);
		userData_.Tgas_vec_.push_back(quokka::EOS<CoolingProblem>::ComputeTgasFromEint(rho, Egas_i));
		const double Erad_i = values.at(RadSystem<CoolingProblem>::radEnergy_index)[0];
		userData_.Erad_vec_.push_back(Erad_i);
	}
}

auto problem_main() -> int
{
	// This problem is a test of line cooling and cosmic-ray heating in a uniform medium. The gas/dust opacity is set to zero, so that the radiation does
	// not interact with matter. The initial conditions are set to a constant temperature and zero radiation energy density. The gas cools at a rate of 0.1
	// per unit time, and is heated by cosmic rays at a rate of 0.03 per unit time. The exact solution is given by the following system of equations:
	// dTgas/dt = -0.1 * Tgas + 0.03,

	// Problem parameters
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;
	const double CFL_number_rad = 0.8;

	const double the_dt = 1.0e-2;

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
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number_rad;
	sim.cflNumber_ = CFL_number_gas;
	sim.initDt_ = the_dt;
	sim.maxDt_ = the_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	std::vector<double> &Tgas = sim.userData_.Tgas_vec_;
	std::vector<double> &Erad = sim.userData_.Erad_vec_;
	std::vector<double> &t_sim = sim.userData_.t_vec_;

	std::vector<double> Tgas_exact_vec{};
	std::vector<double> Erad_exact_vec{};
	for (const auto &t_exact : t_sim) {
		const double Egas_exact_solution = std::exp(-cooling_rate * t_exact) *
						   (cooling_rate * T0 - CR_heating_rate + CR_heating_rate * std::exp(cooling_rate * t_exact)) / cooling_rate;
		const double T_exact_solution = Egas_exact_solution / C_V;
		Tgas_exact_vec.push_back(T_exact_solution);
		const double Erad_exact_solution = -(Egas_exact_solution - C_V * T0 - CR_heating_rate * t_exact) * (chat / c);
		Erad_exact_vec.push_back(Erad_exact_solution);
	}

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> args;
	args["label"] = "T_gas (numerical)";
	args["linestyle"] = "-";
	matplotlibcpp::plot(t_sim, Tgas, args);
	args["label"] = "T_gas (exact)";
	args["linestyle"] = "--";
	matplotlibcpp::plot(t_sim, Tgas_exact_vec, args);
	args["label"] = "E_rad (numerical)";
	args["linestyle"] = "-";
	matplotlibcpp::plot(t_sim, Erad, args);
	args["label"] = "E_rad (exact)";
	args["linestyle"] = "--";
	matplotlibcpp::plot(t_sim, Erad_exact_vec, args);
	matplotlibcpp::xlabel("t (dimensionless)");
	matplotlibcpp::ylabel("T or Erad (dimensionless)");
	matplotlibcpp::legend();
	// matplotlibcpp::ylim(0.0, 2.0);
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./rad_line_cooling_single_group.pdf");
#endif

	// compute L1 error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < t_sim.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Tgas_exact_vec[i]);
		err_norm += std::abs(Erad[i] - Erad_exact_vec[i]);
		sol_norm += std::abs(Tgas_exact_vec[i]) + std::abs(Erad_exact_vec[i]);
	}
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	if (export_csv) {
		std::ofstream file;
		file.open("rad_line_cooling_temp.csv");
		file << "t,Tgas,Tgas_exact\n";
		for (size_t i = 0; i < t_sim.size(); ++i) {
			file << std::scientific << std::setprecision(12) << t_sim[i] << "," << Tgas[i] << "," << Tgas_exact_vec[i] << "\n";
		}
		file.close();

		file.open("rad_line_cooling_rad_energy_density.csv");
		file << "t,Erad,Erad_exact\n";
		for (size_t i = 0; i < t_sim.size(); ++i) {
			file << std::scientific << std::setprecision(12) << t_sim[i] << "," << Erad[i] << "," << Erad_exact_vec[i] << "\n";
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
