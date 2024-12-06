//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak_dust.cpp
/// \brief Defines a test Marshak wave problem with weak coupling between dust and gas.
///

#include "test_radiation_marshak_dust.hpp"
#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"

struct MarshakProblem {
};

AMREX_GPU_MANAGED double kappa1 = NAN; // dust opacity at IR
AMREX_GPU_MANAGED double kappa2 = NAN; // dust opacity at FUV

constexpr double c = 1.0; // speed of light
constexpr double rho0 = 1.0;
constexpr double CV = 1.0;
constexpr double mu = 1.5 / CV; // mean molecular weight
constexpr double initial_T = 1.0;
constexpr double a_rad = 1.0e10;
constexpr double erad_floor = 1.0e-10;
constexpr double initial_Trad = 1.0e-5;
constexpr double T_rad_L = 1.0e-2; // so EradL = 1e2
constexpr double EradL = a_rad * T_rad_L * T_rad_L * T_rad_L * T_rad_L;
// constexpr double T_end_exact = 0.0031597766719577; // dust off; solution of 1 == a_rad * T^4 + T
constexpr double T_end_exact = initial_T; // dust on

// constexpr int n_group_ = 1;
// static constexpr amrex::GpuArray<double, n_group_ + 1> radBoundaries_{1e-10, 1e4};
// static constexpr OpacityModel opacity_model_ = OpacityModel::single_group;
constexpr int n_group_ = 2;
static constexpr amrex::GpuArray<double, n_group_ + 1> radBoundaries_{1e-10, 100, 1e4};
static constexpr OpacityModel opacity_model_ = OpacityModel::piecewise_constant_opacity;

template <> struct quokka::EOS_Traits<MarshakProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<MarshakProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_group_; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = 1.0;
	static constexpr double radiation_constant = a_rad;
};

template <> struct RadSystem_Traits<MarshakProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = 1.0;
	static constexpr amrex::GpuArray<double, n_group_ + 1> radBoundaries = radBoundaries_;
	static constexpr OpacityModel opacity_model = opacity_model_;
};

template <> struct ISM_Traits<MarshakProblem> {
	static constexpr bool enable_dust_gas_thermal_coupling_model = true;
	static constexpr bool enable_photoelectric_heating = false;
	// 1.0e-5 is the minimum value allowed for this test; smaller values will result in negative T_d.
	static constexpr double gas_dust_coupling_threshold = 1.0e-5;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa1;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<MarshakProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa1;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<MarshakProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
								const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		if (i == 0) {
			exponents_and_values[1][i] = kappa1;
		} else {
			exponents_and_values[1][i] = kappa2;
		}
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<MarshakProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Egas0 = initial_T * CV;
	const auto Erads = RadSystem<MarshakProblem>::ComputeThermalRadiationMultiGroup(initial_Trad, radBoundaries_);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<MarshakProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<MarshakProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erads[g];
			state_cc(i, j, k, RadSystem<MarshakProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<MarshakProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<MarshakProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<MarshakProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<MarshakProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<MarshakProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<MarshakProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<MarshakProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<MarshakProblem>::x3GasMomentum_index) = 0.;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<MarshakProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							   amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
							   int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int j = 0;
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();

	// const auto Erads = RadSystem<MarshakProblem>::ComputeThermalRadiation(T_rad_L, radBoundaries_);
	quokka::valarray<double, 2> const Erads = {erad_floor, EradL};
	const double c_device = c;
	const auto Frads = Erads * c_device;

	if (i < lo[0]) {
		// streaming inflow boundary
		// multigroup radiation
		// x1 left side boundary (Marshak)
		for (int g = 0; g < Physics_Traits<MarshakProblem>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<MarshakProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erads[g];
			consVar(i, j, k, RadSystem<MarshakProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = Frads[g];
			consVar(i, j, k, RadSystem<MarshakProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<MarshakProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
	}

	// gas boundary conditions are the same everywhere
	const double Egas = initial_T * CV;
	consVar(i, j, k, RadSystem<MarshakProblem>::gasEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<MarshakProblem>::gasDensity_index) = rho0;
	consVar(i, j, k, RadSystem<MarshakProblem>::gasInternalEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<MarshakProblem>::x1GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<MarshakProblem>::x2GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<MarshakProblem>::x3GasMomentum_index) = 0.;
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double dt_max = 1;
	const int max_timesteps = 5000;

	// read user parameters
	amrex::ParmParse pp("problem");
	pp.query("kappa1", kappa1);
	pp.query("kappa2", kappa2);

	// Boundary conditions
	constexpr int nvars = RadSystem<MarshakProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // Dirichlet x1
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<MarshakProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	// sim.stopTime_ = tmax; // set with runtime parameters
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
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
	std::vector<double> xs(nx);
	std::vector<double> T(nx);
	std::vector<double> T_exact(nx);
	std::vector<double> erad(nx);
	std::vector<double> erad1(nx);
	std::vector<double> erad2(nx);
	std::vector<double> erad1_exact(nx);
	std::vector<double> erad2_exact(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		erad1.at(i) = values.at(RadSystem<MarshakProblem>::radEnergy_index + Physics_NumVars::numRadVars * 0)[i];
		erad.at(i) = erad1.at(i);
		if (n_group_ > 1) {
			erad2.at(i) = values.at(RadSystem<MarshakProblem>::radEnergy_index + Physics_NumVars::numRadVars * 1)[i];
			erad.at(i) += erad2.at(i);
		}
		const double e_gas = values.at(RadSystem<MarshakProblem>::gasInternalEnergy_index)[i];
		T.at(i) = quokka::EOS<MarshakProblem>::ComputeTgasFromEint(rho0, e_gas);
		T_exact.at(i) = T_end_exact;

		erad2_exact.at(i) = x < sim.tNew_[0] ? EradL * std::exp(-x * rho0 * kappa2) : erad_floor;
		erad1_exact.at(i) = x < sim.tNew_[0] ? EradL * std::exp(-x * rho0 * kappa2) * (sim.tNew_[0] - x) : erad_floor;
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 1; i < nx; ++i) { // skip the first cell
		err_norm += std::abs(T[i] - T_exact[i]);
		err_norm += std::abs(erad1[i] - erad1_exact[i]);
		err_norm += std::abs(erad2[i] - erad2_exact[i]);
		sol_norm += std::abs(T_exact[i]);
		sol_norm += std::abs(erad1_exact[i]);
		sol_norm += std::abs(erad2_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.01;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
	// Plot erad1
	matplotlibcpp::clf();
	std::map<std::string, std::string> plot_args;
	std::map<std::string, std::string> plot_args2;
	plot_args["label"] = "numerical solution";
	plot_args2["label"] = "exact solution";
	matplotlibcpp::plot(xs, erad1, plot_args);
	matplotlibcpp::plot(xs, erad1_exact, plot_args2);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("E_rad_group1");
	matplotlibcpp::legend();
	matplotlibcpp::title(std::format("Marshak_dust test at t = {:.1f}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radiation_marshak_dust_Erad1.pdf");

	// Plot erad2
	if (n_group_ > 1) {
		matplotlibcpp::clf();
		matplotlibcpp::plot(xs, erad2, plot_args);
		matplotlibcpp::plot(xs, erad2_exact, plot_args2);
		matplotlibcpp::xlabel("x");
		matplotlibcpp::ylabel("E_rad_group2");
		matplotlibcpp::legend();
		matplotlibcpp::title(std::format("Marshak_dust test at t = {:.1f}", sim.tNew_[0]));
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./radiation_marshak_dust_Erad2.pdf");
	}

	// plot temperature
	matplotlibcpp::clf();
	matplotlibcpp::ylim(0.0, 1.1);
	matplotlibcpp::plot(xs, T, plot_args);
	matplotlibcpp::plot(xs, T_exact, plot_args2);
	matplotlibcpp::xlabel("x");
	matplotlibcpp::ylabel("Temperature");
	matplotlibcpp::legend();
	matplotlibcpp::title(std::format("Marshak_dust test at t = {:.1f}", sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radiation_marshak_dust_temperature.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}
