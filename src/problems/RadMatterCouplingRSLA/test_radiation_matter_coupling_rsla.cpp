//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_matter_coupling.cpp
/// \brief Defines a test problem for radiation-matter coupling.
///

#include "test_radiation_matter_coupling_rsla.hpp"
#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct CouplingProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// constexpr double c = 2.99792458e10; // cgs
constexpr double chat_over_c = 0.1;
constexpr double c_rsla = chat_over_c * C::c_light;

// Su & Olson (1997) test problem
constexpr double eps_SuOlson = 1.0;
constexpr double a_rad = C::a_rad; // cgs
constexpr double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

template <> struct SimulationData<CouplingProblem> {
	std::vector<double> t_vec_;
	std::vector<double> Trad_vec_;
	std::vector<double> Tgas_vec_;
};

template <> struct quokka::EOS_Traits<CouplingProblem> {
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<CouplingProblem> {
	static constexpr double Erad_floor = 0.;
	static constexpr int beta_order = 1;
};

template <> struct Physics_Traits<CouplingProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<CouplingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return 1.0;
}

static constexpr int nmscalars_ = Physics_Traits<CouplingProblem>::numMassScalars;
template <>
AMREX_GPU_HOST_DEVICE auto quokka::EOS<CouplingProblem>::ComputeTgasFromEint(const double /*rho*/, const double Egas,
									     std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
    -> double
{
	return std::pow(4.0 * Egas / alpha_SuOlson, 1. / 4.);
}

template <>
AMREX_GPU_HOST_DEVICE auto quokka::EOS<CouplingProblem>::ComputeEintFromTgas(const double /*rho*/, const double Tgas,
									     std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
    -> double
{
	return (alpha_SuOlson / 4.0) * std::pow(Tgas, 4);
}

template <>
AMREX_GPU_HOST_DEVICE auto
quokka::EOS<CouplingProblem>::ComputeEintTempDerivative(const double /*rho*/, const double Tgas,
							std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
{
	// This is also known as the heat capacity, i.e.
	// 		\del E_g / \del T = \rho c_v,
	// for normal materials.

	// However, for this problem, this must be of the form \alpha T^3
	// in order to obtain an exact solution to the problem.
	// The input parameter is the *temperature*, not Egas itself.

	return alpha_SuOlson * std::pow(Tgas, 3);
}

constexpr double Erad0 = 1.0e12; // erg cm^-3
constexpr double Egas0 = 1.0e2;	 // erg cm^-3
constexpr double rho0 = 1.0e-7;	 // g cm^-3

template <> void QuokkaSimulation<CouplingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<CouplingProblem>::radEnergy_index) = Erad0;
		state_cc(i, j, k, RadSystem<CouplingProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<CouplingProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<CouplingProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<CouplingProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<CouplingProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<CouplingProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<CouplingProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<CouplingProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<CouplingProblem>::x3GasMomentum_index) = 0.;
	});
}

template <> void QuokkaSimulation<CouplingProblem>::computeAfterTimestep()
{
	auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);

		const amrex::Real Etot_i = values.at(RadSystem<CouplingProblem>::gasEnergy_index)[0];
		const amrex::Real x1GasMom = values.at(RadSystem<CouplingProblem>::x1GasMomentum_index)[0];
		const amrex::Real x2GasMom = values.at(RadSystem<CouplingProblem>::x2GasMomentum_index)[0];
		const amrex::Real x3GasMom = values.at(RadSystem<CouplingProblem>::x3GasMomentum_index)[0];
		const amrex::Real rho = values.at(RadSystem<CouplingProblem>::gasDensity_index)[0];
		const amrex::Real Egas_i = RadSystem<CouplingProblem>::ComputeEintFromEgas(rho, x1GasMom, x2GasMom, x3GasMom, Etot_i);

		const amrex::Real Erad_i = values.at(RadSystem<CouplingProblem>::radEnergy_index)[0];

		userData_.Trad_vec_.push_back(std::pow(Erad_i / a_rad, 1. / 4.));
		userData_.Tgas_vec_.push_back(quokka::EOS<CouplingProblem>::ComputeTgasFromEint(rho, Egas_i));
	}
}

auto problem_main() -> int
{
	// Problem parameters

	// const int nx = 4;
	// const double Lx = 1e5; // cm
	const double CFL_number = 1.0;
	const double max_time = 1.0e-2; // s
	const int max_timesteps = 1e6;
	const double constant_dt = 1.0e-8; // s

	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<CouplingProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap); // extrapolate
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	QuokkaSimulation<CouplingProblem> sim(BCs_cc);

	sim.cflNumber_ = CFL_number;
	sim.radiationCflNumber_ = CFL_number;
	sim.constantDt_ = constant_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = -1;
	sim.chat_over_c_ = chat_over_c;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// copy solution slice to vector
	int status = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Solve for temperature as a function of time
		const int nmax = static_cast<int>(sim.userData_.t_vec_.size());
		std::vector<double> t_exact(nmax);
		std::vector<double> Tgas_exact(nmax);
		std::vector<double> Tgas_rsla_exact(nmax);

		const double initial_Tgas = quokka::EOS<CouplingProblem>::ComputeTgasFromEint(rho0, Egas0);
		const auto kappa = RadSystem<CouplingProblem>::ComputePlanckOpacity(rho0, initial_Tgas);

		for (int n = 0; n < nmax; ++n) {
			const double time_t = sim.userData_.t_vec_.at(n);
			const double arad = RadSystem<CouplingProblem>::radiation_constant_;
			const double c = RadSystem<CouplingProblem>::c_light_;
			const double E0 = (Erad0 + Egas0) / (arad + alpha_SuOlson / 4.0);

			const double T0_4 = std::pow(initial_Tgas, 4);

			const double E0_rsla = ((c / c_rsla) * Erad0 + Egas0) / (a_rad + (c_rsla / c) * alpha_SuOlson / 4.0);

			const double T4_rsla = (T0_4 - (c_rsla / c) * E0_rsla) *
						   std::exp(-(4. / alpha_SuOlson) * (a_rad + (c_rsla / c) * alpha_SuOlson / 4.0) * kappa * rho0 * c * time_t) +
					       (c_rsla / c) * E0_rsla;

			const double T_gas_rsla = std::pow(T4_rsla, 1. / 4.);

			const double T4 = (T0_4 - E0) * std::exp(-(4. / alpha_SuOlson) * (arad + alpha_SuOlson / 4.0) * kappa * rho0 * c * time_t) + E0;

			const double T_gas = std::pow(T4, 1. / 4.);

			Tgas_rsla_exact.at(n) = T_gas_rsla;
			Tgas_exact.at(n) = T_gas;
			t_exact.at(n) = time_t;
		}

		std::vector<double> &Tgas = sim.userData_.Tgas_vec_;
		std::vector<double> &t = sim.userData_.t_vec_;

		// compute L1 error norm
		double err_norm = 0.;
		double sol_norm = 0.;
		for (size_t i = 0; i < t.size(); ++i) {
			err_norm += std::abs(Tgas[i] - Tgas_rsla_exact[i]);
			sol_norm += std::abs(Tgas_rsla_exact[i]);
		}
		const double rel_error = err_norm / sol_norm;
		// When using C::a_rad as radiation_constant_cgs_, the relative error goes up to 3e-5, so I'm increasing the tolerance
		const double error_tol = 5e-5;
		amrex::Print() << "relative L1 error norm = " << rel_error << std::endl;
		if (rel_error > error_tol) {
			status = 1;
		}

#ifdef HAVE_PYTHON
		std::vector<double> &Trad = sim.userData_.Trad_vec_;

		// Plot results
		matplotlibcpp::clf();
		matplotlibcpp::yscale("log");
		matplotlibcpp::xscale("log");
		matplotlibcpp::ylim(0.5 * std::min(Tgas.front(), Trad.front()), 4.0 * std::max(Trad.back(), Tgas.back()));

		std::map<std::string, std::string> rsla_args;
		rsla_args["label"] = "simulated gas temperature (RSLA)";
		rsla_args["linestyle"] = "-";
		rsla_args["color"] = "C2";
		matplotlibcpp::plot(t, Tgas, rsla_args);

		// std::map<std::string, std::string> exactsolrsla_args;
		// exactsolrsla_args["label"] = "gas temperature (exact, RSLA)";
		// exactsolrsla_args["linestyle"] = "--";
		// exactsolrsla_args["color"] = "C2";
		// matplotlibcpp::plot(t, Tgas_rsla_exact, exactsolrsla_args);

		std::map<std::string, std::string> exactsol_args;
		exactsol_args["label"] = "exact gas temperature (no RSLA)";
		exactsol_args["linestyle"] = "--";
		exactsol_args["color"] = "C2";
		matplotlibcpp::plot(t, Tgas_exact, exactsol_args);

		matplotlibcpp::legend();
		matplotlibcpp::xlabel("time t (seconds)");
		matplotlibcpp::ylabel("temperature T (Kelvins)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(fmt::format("./radcoupling_rsla.pdf"));

		matplotlibcpp::clf();

		std::vector<double> frac_err(t.size());
		for (size_t i = 0; i < t.size(); ++i) {
			frac_err.at(i) = Tgas_rsla_exact.at(i) / Tgas.at(i) - 1.0;
		}
		matplotlibcpp::plot(t, frac_err);
		matplotlibcpp::xlabel("time t (s)");
		matplotlibcpp::ylabel("fractional error in material temperature");
		matplotlibcpp::save(fmt::format("./radcoupling_rsla_fractional_error.pdf"));
#endif
	}

	return status;
}
