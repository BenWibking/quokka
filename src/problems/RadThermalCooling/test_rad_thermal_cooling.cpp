/// \file test_rad_thermal_cooling.cpp
/// \brief Defines a test problem for line cooling and cosmic-ray heating in a uniform medium.
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "fundamental_constants.H"
#include "physics_info.hpp"
#include "test_rad_thermal_cooling.hpp"
#include "util/fextract.hpp"

struct PulseProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

const bool only_check_last = 1;

// dimensionless

// const double sim_dt = 0.01;
const double sim_dt = 0.1;
const double max_time = 5.0;

constexpr double c = 1.0;
constexpr double chat = c;
constexpr double v0 = 0.0;
constexpr double kappa0 = 1.0;

constexpr double T0 = 1.0;   // temperature
constexpr double rho0 = 1.0; // matter density
constexpr double a_rad = 1.0;
constexpr double mu = 1.5; // mean molecular weight; so that C_V = 1.0
constexpr double C_V = 1.0;
constexpr double k_B = 1.0;

constexpr double erad_floor = a_rad * 1e-20;


// cgs units

// // const double sim_dt = 0.01;
// const double sim_dt = 0.1;
// const double max_time = 5.0;

// constexpr double c = C::c_light;
// constexpr double chat = c;
// constexpr double v0 = 0.0;

// constexpr double T0 = 1.0;   // temperature
// constexpr double a_rad = C::a_rad;
// constexpr double mu = C::m_u;
// constexpr double k_B = C::k_B;
// constexpr double C_V = 1.5 * k_B / mu;
// constexpr double rho0 = a_rad * T0 * T0 * T0 / C_V;
// constexpr double dx = 2.997924580e+10; // cm
// constexpr double kappa0 = 1.0 / dx / rho0;

// constexpr double erad_floor = a_rad * 1e-20;

template <> struct SimulationData<PulseProblem> {
	std::vector<double> t_vec_;
	std::vector<double> Tgas_vec_;
	std::vector<double> Trad_vec_;
};

template <> struct quokka::EOS_Traits<PulseProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
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
	static constexpr int nGroups = 1;
};

template <> struct RadSystem_Traits<PulseProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = chat;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
};

template <> struct ISM_Traits<PulseProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
};

// template <typename I, typename T>
// AMREX_GPU_HOST_DEVICE AMREX_INLINE
// void actual_eos_in_iteration (I input, T& state)
// {
// 	state.e = C_V * state.rho * state.T;
// }

// template <typename problem_t>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void actual_rhs(burn_t& state, Array1D<Real, 1, neqs>& ydot)
{
	Array1D<Real, 0, NumSpec-1> X;
	for (int i = 0; i < NumSpec; ++i) {
		X(i) = state.xn[i];
	}

	Real const rho = state.rho;
	Real const T = state.T;
	// Real const T = state.e / (rho * C_V);

	const Real fourPiBoverc = a_rad * T * T * T * T;

	ydot(1) = chat * rho * kappa0 * (fourPiBoverc - X(0));
	const Real edot = - c * rho * kappa0 * (fourPiBoverc - X(0));

	// Append the energy equation (this is erg/g/s)
	ydot(net_ienuc) = edot;
}


// template<class MatrixType>
// AMREX_GPU_HOST_DEVICE AMREX_INLINE
// void actual_jac(const burn_t& state, MatrixType& jac)
// {
// 	const double rho = state.rho;
// 	Real const T = state.T;
// 	// Real const T = state.e / (rho * C_V);
// 	const double dEg_dT = 4.0 * a_rad * T * T * T;
// 	const double tau = kappa0 * rho * chat;

// 	// jac(1,1) = 1.0;
// 	// jac(1,2) = c / chat;
// 	// jac(2,1) = 1.0/ C_V * dEg_dT;
// 	// jac(2,2) = -1.0 / tau - 1.0;

// 	// Jacobian: 
// 	// 11 = - c * rho * kappa0 * dEg_dt, 12 = c * rho * kappa0
// 	// 21 = c * rho * kappa0 * dEg_dt, 22 = - c * rho * kappa0
// 	jac(1,1) = - tau * dEg_dT;
// 	jac(1,2) = tau;
// 	jac(2,1) = tau * dEg_dT;
// 	jac(2,2) = - tau;
// }

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<PulseProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<PulseProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const double Egas = quokka::EOS<PulseProblem>::ComputeEintFromTgas(rho0, T0);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<PulseProblem>::radEnergy_index) = erad_floor;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3RadFlux_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasEnergy_index) = Egas + 0.5 * rho0 * v0 * v0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<PulseProblem>::x1GasMomentum_index) = v0 * rho0;
		state_cc(i, j, k, RadSystem<PulseProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<PulseProblem>::x3GasMomentum_index) = 0.;
	});
}

template <> void QuokkaSimulation<PulseProblem>::computeAfterTimestep()
{
	auto [_, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5); // NOLINT

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);

		const amrex::Real Etot_i = values.at(RadSystem<PulseProblem>::gasEnergy_index)[0];
		const amrex::Real x1GasMom = values.at(RadSystem<PulseProblem>::x1GasMomentum_index)[0];
		const amrex::Real x2GasMom = values.at(RadSystem<PulseProblem>::x2GasMomentum_index)[0];
		const amrex::Real x3GasMom = values.at(RadSystem<PulseProblem>::x3GasMomentum_index)[0];
		const amrex::Real rho = values.at(RadSystem<PulseProblem>::gasDensity_index)[0];
		const amrex::Real Egas_i = RadSystem<PulseProblem>::ComputeEintFromEgas(rho, x1GasMom, x2GasMom, x3GasMom, Etot_i);
		userData_.Tgas_vec_.push_back(quokka::EOS<PulseProblem>::ComputeTgasFromEint(rho, Egas_i));
		const double Erad_i = values.at(RadSystem<PulseProblem>::radEnergy_index)[0];
		const double Trad_i = std::pow(Erad_i / a_rad, 1. / 4.);
		userData_.Trad_vec_.push_back(Trad_i);
	}
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
	const double CFL_number_rad = 0.8;

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
	sim.initDt_ = sim_dt;
	sim.maxDt_ = sim_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// append initial conditions to userData_ vectors
	auto [_, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	sim.userData_.t_vec_.push_back(0.0);
	sim.userData_.Tgas_vec_.push_back(T0);
	sim.userData_.Trad_vec_.push_back(0.0);

	// evolve
	sim.evolve();

	std::vector<double> &t_sim = sim.userData_.t_vec_;
	std::vector<double> &Tgas = sim.userData_.Tgas_vec_;
	std::vector<double> &Trad = sim.userData_.Trad_vec_;

	// read in exact solution from ../auxiliary_files/rad_thermal_cooling.csv
	std::vector<double> t_exact{};
	std::vector<double> Tgas_exact_vec{};
	std::vector<double> Trad_exact_vec{};

	std::ifstream fstream("../src/problems/auxiliary_files/rad_thermal_cooling.csv", std::ios::in);
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
		auto Tgas_val = values.at(1);
		auto Trad_val = values.at(2);
		t_exact.push_back(t_val);
		Tgas_exact_vec.push_back(Tgas_val);
		Trad_exact_vec.push_back(Trad_val);
	}

	// interpolate exact solution onto simulation times
	std::vector<double> Tgas_exact(t_sim.size());
	std::vector<double> Trad_exact(t_sim.size());
	interpolate_arrays(t_sim.data(), Tgas_exact.data(), static_cast<int>(t_sim.size()), t_exact.data(), Tgas_exact_vec.data(), static_cast<int>(t_exact.size()));
	interpolate_arrays(t_sim.data(), Trad_exact.data(), static_cast<int>(t_sim.size()), t_exact.data(), Trad_exact_vec.data(), static_cast<int>(t_exact.size()));

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> args;
	args["label"] = "T_gas (numerical)";
	args["linestyle"] = "-";
	matplotlibcpp::plot(t_sim, Tgas, args);
	args["label"] = "T_gas (exact)";
	args["linestyle"] = "--";
	matplotlibcpp::plot(t_exact, Tgas_exact_vec, args);
	args["label"] = "T_rad (numerical)";
	args["linestyle"] = "-";
	matplotlibcpp::plot(t_sim, Trad, args);
	args["label"] = "T_rad (exact)";
	args["linestyle"] = "--";
	matplotlibcpp::plot(t_exact, Trad_exact_vec, args);
	matplotlibcpp::xlabel("t (dimensionless)");
	matplotlibcpp::ylabel("T or Erad (dimensionless)");
	matplotlibcpp::legend();
	matplotlibcpp::xlim(0.0, 3.0);
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./rad_thermal_cooling_single_group.pdf");
#endif

	// compute L1 error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	if (only_check_last) {
		err_norm += std::abs(Tgas.back() - Tgas_exact.back());
		err_norm += std::abs(Trad.back() - Trad_exact.back());
		sol_norm += std::abs(Tgas_exact.back()) + std::abs(Trad_exact.back());
	} else {
		for (size_t i = 0; i < t_sim.size(); ++i) {
			err_norm += std::abs(Tgas[i] - Tgas_exact[i]);
			err_norm += std::abs(Trad[i] - Trad_exact[i]);
			sol_norm += std::abs(Tgas_exact[i]) + std::abs(Trad_exact[i]);
		}
	}
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	std::ofstream file;
	file.open("rad_thermal_cooling_temp.csv");
	file << "t,Tgas,Tgas_exact\n";
	for (size_t i = 0; i < t_sim.size(); ++i) {
		file << std::scientific << std::setprecision(12) << t_sim[i] << "," << Tgas[i] << "," << Tgas_exact[i] << "\n";
	}
	file.close();

	file.open("rad_thermal_cooling_rad_energy_density.csv");
	file << "t,Trad,Erad_exact\n";
	for (size_t i = 0; i < t_sim.size(); ++i) {
		file << std::scientific << std::setprecision(12) << t_sim[i] << "," << Trad[i] << "," << Trad_exact[i] << "\n";
	}
	file.close();

	// exit
	int status = 0;
	const double error_tol = 0.05 * sim_dt;
	if (rel_error > error_tol) {
		status = 1;
	}
	return status;
}
