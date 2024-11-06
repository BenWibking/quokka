//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///

#include <cmath>

#include "AMReX_BLassert.H"
#include "AMReX_ParallelDescriptor.H"

#include "QuokkaSimulation.hpp"
#include "test_radiation_marshak.hpp"
#include "util/fextract.hpp"

struct SuOlsonProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// Su & Olson (1997) parameters
constexpr double eps_SuOlson = 1.0;
constexpr double kappa = 1.0;
constexpr double rho0 = 1.0;	   // g cm^-3 (matter density)
constexpr double T_hohlraum = 1.0; // dimensionless
constexpr double a_rad = 1.0;
constexpr double c = 1.0;
constexpr double alpha_SuOlson = 4.0 * a_rad / eps_SuOlson;

constexpr double T_initial = 1.0e-2;

template <> struct quokka::EOS_Traits<SuOlsonProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<SuOlsonProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double radiation_constant = a_rad;
	static constexpr double Erad_floor = 0.;
	static constexpr int beta_order = 0;
};

template <> struct Physics_Traits<SuOlsonProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = 1.0;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = a_rad;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<SuOlsonProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa;
}

static constexpr int nmscalars_ = Physics_Traits<SuOlsonProblem>::numMassScalars;
template <>
AMREX_GPU_HOST_DEVICE auto quokka::EOS<SuOlsonProblem>::ComputeTgasFromEint(const double /*rho*/, const double Egas,
									    std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
    -> double
{
	return std::pow(4.0 * Egas / alpha_SuOlson, 1. / 4.);
}

template <>
AMREX_GPU_HOST_DEVICE auto quokka::EOS<SuOlsonProblem>::ComputeEintFromTgas(const double /*rho*/, const double Tgas,
									    std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/)
    -> double
{
	return (alpha_SuOlson / 4.0) * std::pow(Tgas, 4);
}

template <>
AMREX_GPU_HOST_DEVICE auto
quokka::EOS<SuOlsonProblem>::ComputeEintTempDerivative(const double /*rho*/, const double Tgas,
						       std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const & /*massScalars*/) -> double
{
	// This is also known as the heat capacity, i.e.
	// 		\del E_g / \del T = \rho c_v,
	// for normal materials.

	// However, for this problem, this must be of the form \alpha T^3
	// in order to obtain an exact solution to the problem.
	// The input parameters are the density and *temperature*, not Egas
	// itself.

	return alpha_SuOlson * std::pow(Tgas, 3);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<SuOlsonProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							   amrex::GeometryData const & /*geom*/, const amrex::Real /*time*/, const amrex::BCRec *bcr,
							   int /*bcomp*/, int /*orig_comp*/)
{
	if (!((bcr->lo(0) == amrex::BCType::ext_dir) || (bcr->hi(0) == amrex::BCType::ext_dir))) {
		return;
	}

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

	if (i < 0) {
		// Marshak boundary condition
		const double T_H = T_hohlraum;
		const double E_inc = a_rad * std::pow(T_H, 4);
		const double E_0 = consVar(0, j, k, RadSystem<SuOlsonProblem>::radEnergy_index);
		const double F_0 = consVar(0, j, k, RadSystem<SuOlsonProblem>::x1RadFlux_index);

		// use value at interface to solve for F_rad in the ghost zones
		const double F_bdry = 0.5 * c * E_inc - 0.5 * (c * E_0 + 2.0 * F_0);

		AMREX_ASSERT(std::abs(F_bdry / (c * E_inc)) < 1.0);

		// x1 left side boundary (Marshak)
		consVar(i, j, k, RadSystem<SuOlsonProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<SuOlsonProblem>::x1RadFlux_index) = F_bdry;
		consVar(i, j, k, RadSystem<SuOlsonProblem>::x2RadFlux_index) = 0.;
		consVar(i, j, k, RadSystem<SuOlsonProblem>::x3RadFlux_index) = 0.;
	} else {
		// right-side boundary -- constant
		const double Erad = a_rad * std::pow(T_initial, 4);

		consVar(i, j, k, RadSystem<SuOlsonProblem>::radEnergy_index) = Erad;
		consVar(i, j, k, RadSystem<SuOlsonProblem>::x1RadFlux_index) = 0;
		consVar(i, j, k, RadSystem<SuOlsonProblem>::x2RadFlux_index) = 0;
		consVar(i, j, k, RadSystem<SuOlsonProblem>::x3RadFlux_index) = 0;
	}

	// gas boundary conditions are the same on both sides
	const double Egas = quokka::EOS<SuOlsonProblem>::ComputeEintFromTgas(rho0, T_initial);
	consVar(i, j, k, RadSystem<SuOlsonProblem>::gasEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<SuOlsonProblem>::gasDensity_index) = rho0;
	consVar(i, j, k, RadSystem<SuOlsonProblem>::gasInternalEnergy_index) = Egas;
	consVar(i, j, k, RadSystem<SuOlsonProblem>::x1GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<SuOlsonProblem>::x2GasMomentum_index) = 0.;
	consVar(i, j, k, RadSystem<SuOlsonProblem>::x3GasMomentum_index) = 0.;
}

template <> void QuokkaSimulation<SuOlsonProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double Egas = quokka::EOS<SuOlsonProblem>::ComputeEintFromTgas(rho0, T_initial);
		const double Erad = a_rad * std::pow(T_initial, 4);

		state_cc(i, j, k, RadSystem<SuOlsonProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SuOlsonProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters

	const int max_timesteps = 2e4;
	const double CFL_number = 0.4;

	const double initial_dtau = 1e-9; // dimensionless time
	const double max_dtau = 1e-3;	  // dimensionless time
	const double max_tau = 10.0;	  // dimensionless time
	// const double Lz = 20.0;	  // dimensionless length

	// Su & Olson (1997) parameters
	const double chi = rho0 * kappa; // cm^-1 (total matter opacity)
	// const double Lx = Lz / chi;	// cm
	const double max_time = max_tau / (eps_SuOlson * c * chi);	  // s
	const double max_dt = max_dtau / (eps_SuOlson * c * chi);	  // s
	const double initial_dt = initial_dtau / (eps_SuOlson * c * chi); // s

	constexpr int nvars = RadSystem<SuOlsonProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // custom (Marshak) x1
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<SuOlsonProblem> sim(BCs_cc);

	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.initDt_ = initial_dt;
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

	// Check result
	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> Trad(nx);
		std::vector<double> Tgas(nx);
		std::vector<double> Erad(nx);
		std::vector<double> Egas(nx);

		for (int i = 0; i < nx; ++i) {
			const double x = position[i];
			xs.at(i) = std::sqrt(3.0) * x;

			const double Erad_t = values.at(RadSystem<SuOlsonProblem>::radEnergy_index)[i];
			Erad.at(i) = Erad_t;
			Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);

			const double Etot_t = values.at(RadSystem<SuOlsonProblem>::gasEnergy_index)[i];
			const double rho = values.at(RadSystem<SuOlsonProblem>::gasDensity_index)[i];
			const double x1GasMom = values.at(RadSystem<SuOlsonProblem>::x1GasMomentum_index)[i];
			const double Ekin = (x1GasMom * x1GasMom) / (2.0 * rho);

			const double Egas_t = (Etot_t - Ekin);
			Egas.at(i) = Egas_t;
			Tgas.at(i) = quokka::EOS<SuOlsonProblem>::ComputeTgasFromEint(rho, Egas_t);
		}

		// read in exact solution

		std::vector<double> xs_exact;
		std::vector<double> Trad_exact;
		std::vector<double> Tmat_exact;

		std::string filename = "../extern/SuOlson/100pt_tau10p0.dat";
		std::ifstream fstream(filename, std::ios::in);
		AMREX_ALWAYS_ASSERT(fstream.is_open());

		std::string header;
		std::getline(fstream, header);

		for (std::string line; std::getline(fstream, line);) {
			std::istringstream iss(line);
			std::vector<double> values;

			for (double value = NAN; iss >> value;) {
				values.push_back(value);
			}
			auto x_val = std::sqrt(3.0) * values.at(1);
			auto Trad_val = values.at(4);
			auto Tmat_val = values.at(5);

			xs_exact.push_back(x_val);
			Trad_exact.push_back(Trad_val);
			Tmat_exact.push_back(Tmat_val);
		}

		// compute error norm

		std::vector<double> Trad_exact_interp(xs.size());
		interpolate_arrays(xs.data(), Trad_exact_interp.data(), static_cast<int>(xs.size()), xs_exact.data(), Trad_exact.data(),
				   static_cast<int>(xs_exact.size()));

		double err_norm = 0.;
		double sol_norm = 0.;
		const double t = sim.tNew_[0];
		const double xmax = c * t;
		amrex::Print() << "diffusion length = " << xmax << std::endl;
		for (size_t i = 0; i < xs.size(); ++i) {
			if (xs[i] < xmax) {
				err_norm += std::abs(Trad[i] - Trad_exact_interp[i]);
				sol_norm += std::abs(Trad_exact_interp[i]);
			}
		}

		const double error_tol = 0.02; // 2 per cent
		const double rel_error = err_norm / sol_norm;
		amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

		if ((rel_error > error_tol) || std::isnan(rel_error)) {
			status = 1;
		}

#ifdef HAVE_PYTHON

		// plot results

		// radiation temperature
		std::map<std::string, std::string> Trad_args;
		Trad_args["label"] = "radiation temperature";
		matplotlibcpp::plot(xs, Trad, Trad_args);

		std::map<std::string, std::string> Trad_exact_args;
		Trad_exact_args["label"] = "radiation temperature (exact)";
		matplotlibcpp::plot(xs_exact, Trad_exact, Trad_exact_args);

		matplotlibcpp::xlabel("length x (dimensionless)");
		matplotlibcpp::ylabel("temperature (dimensionless)");
		matplotlibcpp::xlim(0.4, 100.); // dimensionless
		matplotlibcpp::ylim(0.0, 1.0);	// dimensionless
		matplotlibcpp::xscale("log");
		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
		matplotlibcpp::save("./marshak_wave_temperature.pdf");

		// material temperature
		matplotlibcpp::clf();

		std::map<std::string, std::string> Tgas_args;
		Tgas_args["label"] = "gas temperature";
		matplotlibcpp::plot(xs, Tgas, Tgas_args);

		std::map<std::string, std::string> Tgas_exact_args;
		Tgas_exact_args["label"] = "gas temperature (exact)";
		matplotlibcpp::plot(xs_exact, Tmat_exact, Tgas_exact_args);

		matplotlibcpp::xlabel("length x (dimensionless)");
		matplotlibcpp::ylabel("temperature (dimensionless)");
		matplotlibcpp::xlim(0.4, 100.); // dimensionless
		matplotlibcpp::ylim(0.0, 1.0);	// dimensionless
		matplotlibcpp::xscale("log");
		matplotlibcpp::legend();
		matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
		matplotlibcpp::save("./marshak_wave_gastemperature.pdf");
#endif
	}

	return status;
}
