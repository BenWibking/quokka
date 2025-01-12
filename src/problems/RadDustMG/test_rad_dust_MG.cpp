/// \file test_rad_dust_MG.cpp
/// \brief Defines a multigroup test problem for gas-dust-radiation coupling in uniform medium.
///

#include "test_rad_dust_MG.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "util/fextract.hpp"
#include <vector>

struct DustProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// In this test, the hydro time step is dt = CFL * dx / (chat / 10) = 0.8 * (1/8) / (1e7 / 10) = 1e-8

AMREX_GPU_MANAGED double C_V = 1.5;
AMREX_GPU_MANAGED double chi0 = 10000.0;

constexpr int beta_order_ = 1; // order of beta in the radiation four-force
constexpr double c = 1.0e8;
constexpr double chat = 0.1 * c;
constexpr double v0 = 0.0;

constexpr double T0 = 1.0;
constexpr double a_rad = 1.0;
constexpr double mu = 1.0;
constexpr double k_B = 1.0;

constexpr double max_time = 3.0e-5;

constexpr double erad_floor = 1.0e-20 * a_rad * T0 * T0 * T0 * T0;

template <> struct SimulationData<DustProblem> {
	std::vector<double> t_vec_;
	std::vector<double> Trad_vec_;
	std::vector<double> Tgas_vec_;
};

template <> struct quokka::EOS_Traits<DustProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<DustProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 4;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = a_rad;
};

template <> struct RadSystem_Traits<DustProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = beta_order_;
	static constexpr double energy_unit = 1.;
	static constexpr amrex::GpuArray<double, Physics_Traits<DustProblem>::nGroups + 1> radBoundaries{1.0e-3, 0.1, 1.0, 10.0, 1.0e3};
	// static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_fixed_slope_spectrum;
};

template <> struct ISM_Traits<DustProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
	static constexpr bool enable_photoelectric_heating = false;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<DustProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double rho, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = chi0 / rho;
	}
	return exponents_and_values;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationMultiGroup(amrex::Real temperature,
										     amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<amrex::Real, nGroups_> Erad_g{};
	const double radEnergyFractions = 1.0 / nGroups_;
	const double power = radiation_constant_ * temperature;
	Erad_g.fillin(power * radEnergyFractions);
	return Erad_g;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<DustProblem>::ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature,
												   amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<amrex::Real, nGroups_> d_power_dt{};
	const double radEnergyFractions = 1.0 / nGroups_;
	d_power_dt.fillin(radiation_constant_ * radEnergyFractions);
	return d_power_dt;
}

template <> void QuokkaSimulation<DustProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double rho0 = C_V * mu / (1.5 * k_B);
	const double Egas = quokka::EOS<DustProblem>::ComputeEintFromTgas(rho0, T0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<DustProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<DustProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = erad_floor;
			state_cc(i, j, k, RadSystem<DustProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<DustProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<DustProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<DustProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<DustProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<DustProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<DustProblem>::x1GasMomentum_index) = v0 * rho0;
		state_cc(i, j, k, RadSystem<DustProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<DustProblem>::x3GasMomentum_index) = 0.;
	});
}

template <> void QuokkaSimulation<DustProblem>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5); // NOLINT

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);

		const amrex::Real Etot_i = values.at(RadSystem<DustProblem>::gasEnergy_index)[0];
		const amrex::Real x1GasMom = values.at(RadSystem<DustProblem>::x1GasMomentum_index)[0];
		const amrex::Real x2GasMom = values.at(RadSystem<DustProblem>::x2GasMomentum_index)[0];
		const amrex::Real x3GasMom = values.at(RadSystem<DustProblem>::x3GasMomentum_index)[0];
		const amrex::Real rho = values.at(RadSystem<DustProblem>::gasDensity_index)[0];
		const amrex::Real Egas_i = RadSystem<DustProblem>::ComputeEintFromEgas(rho, x1GasMom, x2GasMom, x3GasMom, Etot_i);
		double Erad_i = 0.0;
		for (int g = 0; g < Physics_Traits<DustProblem>::nGroups; ++g) {
			Erad_i += values.at(RadSystem<DustProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[0];
		}
		// userData_.Trad_vec_.push_back(std::pow(Erad_i / a_rad, 1. / 4.));
		userData_.Trad_vec_.push_back(Erad_i / a_rad);
		userData_.Tgas_vec_.push_back(quokka::EOS<DustProblem>::ComputeTgasFromEint(rho, Egas_i));
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const int max_timesteps = 1e6;
	const double CFL_number_gas = 0.8;

	// Boundary conditions
	constexpr int nvars = RadSystem<DustProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// read C_V from input file
	amrex::ParmParse pp("problem");
	pp.query("C_V", C_V);
	pp.query("chi0", chi0);

	// Problem initialization
	QuokkaSimulation<DustProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.cflNumber_ = CFL_number_gas;
	sim.radiationCflNumber_ = CFL_number_gas;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	const bool perfect_coupling = sim.dustGasInteractionCoeff_ > 1.0e15;

	// read in exact solution
	std::vector<double> ts_exact{};
	std::vector<double> Trad_exact{};
	std::vector<double> Tgas_exact{};

	std::ifstream fstream("../extern/data/dust/rad_dust_exact_chat0.1.csv", std::ios::in);
	AMREX_ALWAYS_ASSERT(fstream.is_open());
	std::string header;
	std::getline(fstream, header);

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;
		std::string value;

		while (std::getline(iss, value, ',')) {
			values.push_back(std::stod(value));
		}
		auto t_val = values.at(0);
		auto Tmat_val = values.at(1);
		auto Trad_val = values.at(2);
		ts_exact.push_back(t_val);
		Tgas_exact.push_back(Tmat_val);
		Trad_exact.push_back(Trad_val);
	}

	std::vector<double> &Tgas = sim.userData_.Tgas_vec_;
	std::vector<double> &Trad = sim.userData_.Trad_vec_;
	std::vector<double> &t = sim.userData_.t_vec_;

	std::vector<double> Tgas_interp(t.size());
	std::vector<double> Trad_interp(t.size());
	interpolate_arrays(t.data(), Tgas_interp.data(), static_cast<int>(t.size()), ts_exact.data(), Tgas_exact.data(), static_cast<int>(ts_exact.size()));
	interpolate_arrays(t.data(), Trad_interp.data(), static_cast<int>(t.size()), ts_exact.data(), Trad_exact.data(), static_cast<int>(ts_exact.size()));

	// compute error norm
	const double error_tol = 0.003;
	double err_norm = 0.;
	double sol_norm = 0.;
	if (!perfect_coupling) {
		for (size_t i = 0; i < t.size(); ++i) {
			err_norm += std::abs(Tgas[i] - Tgas_interp[i]);
			err_norm += std::abs(Trad[i] - Trad_interp[i]);
			sol_norm += std::abs(Tgas_interp[i]) + std::abs(Trad_interp[i]);
		}
	} else {
		// compare the last value
		err_norm += std::abs(Tgas.back() - Tgas_exact.back());
		err_norm += std::abs(Trad.back() - Trad_exact.back());
		sol_norm += std::abs(Tgas_exact.back()) + std::abs(Trad_exact.back());
	}
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	matplotlibcpp::xscale("log");
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	std::map<std::string, std::string> Texact_args;
	std::map<std::string, std::string> Tradexact_args;
	Trad_args["label"] = "radiation (numerical)";
	Trad_args["linestyle"] = "--";
	Trad_args["color"] = "C1";
	Tradexact_args["label"] = "radiation (exact)";
	Tradexact_args["linestyle"] = "-";
	Tradexact_args["color"] = "k";
	Tgas_args["label"] = "gas (numerical)";
	Tgas_args["linestyle"] = "--";
	Tgas_args["color"] = "C2";
	Texact_args["label"] = "gas (exact)";
	Texact_args["linestyle"] = "-";
	Texact_args["color"] = "k";
	// plot exact solution; skip the first point because it has t = 0
	matplotlibcpp::plot(std::vector<double>(ts_exact.begin() + 1, ts_exact.end()), std::vector<double>(Tgas_exact.begin() + 1, Tgas_exact.end()),
			    Texact_args);
	matplotlibcpp::plot(std::vector<double>(ts_exact.begin() + 1, ts_exact.end()), std::vector<double>(Trad_exact.begin() + 1, Trad_exact.end()),
			    Tradexact_args);
	matplotlibcpp::plot(t, Tgas, Tgas_args);
	matplotlibcpp::plot(t, Trad, Trad_args);
	matplotlibcpp::xlabel("t (dimensionless)");
	matplotlibcpp::ylabel("T (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::tight_layout();
	if (!perfect_coupling) {
		matplotlibcpp::save("./rad_dust_MG_T.pdf");
	} else {
		matplotlibcpp::save("./rad_dust_MG_T_perfect_coupling.pdf");
	}
#endif

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error)) {
		status = 1;
	}
	return status;
}
