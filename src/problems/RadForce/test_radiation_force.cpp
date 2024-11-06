//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_force.cpp
/// \brief Defines a test problem for radiation force terms.
///

#include <cstdint>
#include <string>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "test_radiation_force.hpp"
#include "util/ArrayUtil.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct TubeProblem {
};

constexpr amrex::Real kappa0 = 5.0;  // cm^2 g^-1
constexpr double mu = 2.33 * C::m_u; // g
constexpr double gamma_gas = 1.0;    // isothermal gas EOS
constexpr double a0 = 0.2e5;	     // cm s^-1
constexpr double tau = 1.0e-6;	     // optical depth (dimensionless)

constexpr double rho0 = 1.0e5 * mu; // g cm^-3
constexpr double Mach0 = 1.1;	    // Mach number at wind base
constexpr double Mach1 = 2.128410288469465339;

constexpr double Frad0 = rho0 * a0 * c_light_cgs_ / tau; // erg cm^-2 s^-1
constexpr double g0 = kappa0 * Frad0 / c_light_cgs_;	 // cm s^{-2}
constexpr double Lx = (a0 * a0) / g0;			 // cm

template <> struct quokka::EOS_Traits<TubeProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = gamma_gas;
	static constexpr double cs_isothermal = a0; // only used when gamma = 1
};

template <> struct Physics_Traits<TubeProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	// number of radiation groups
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<TubeProblem> {
	static constexpr double c_hat_over_c = 10. * (Mach1 * a0) / C::c_light;
	static constexpr double Erad_floor = 0.;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr int beta_order = 1;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real { return 0.; }

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<TubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// calculate radEnergyFractions
	quokka::valarray<amrex::Real, Physics_Traits<TubeProblem>::nGroups> radEnergyFractions{};
	for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
		radEnergyFractions[g] = 1.0 / Physics_Traits<TubeProblem>::nGroups;
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const rho = rho0;

		for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<TubeProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) =
			    Frad0 * radEnergyFractions[g] / c_light_cgs_;
			state_cc(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = Frad0 * radEnergyFractions[g];
			state_cc(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}

		state_cc(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = 0;
		state_cc(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0;
		state_cc(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0;
		state_cc(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = 0;
		state_cc(i, j, k, RadSystem<TubeProblem>::gasInternalEnergy_index) = 0.;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<TubeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
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

	amrex::Real const Erad = Frad0 / c_light_cgs_;
	amrex::Real const Frad = Frad0;
	amrex::Real rho = NAN;
	amrex::Real vel = NAN;

	quokka::valarray<amrex::Real, Physics_Traits<TubeProblem>::nGroups> radEnergyFractions{};
	for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
		radEnergyFractions[g] = 1.0 / Physics_Traits<TubeProblem>::nGroups;
	}

	if (i < lo[0]) {
		// left side
		rho = rho0;
		vel = Mach0 * a0;
		// Dirichlet
		for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<TubeProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad * radEnergyFractions[g];
			consVar(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = Frad * radEnergyFractions[g];
			consVar(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}

		consVar(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho;
		consVar(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = 0.;
		consVar(i, j, k, RadSystem<TubeProblem>::gasInternalEnergy_index) = 0.;
		consVar(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = rho * vel;
		consVar(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;
	}
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 128;
	constexpr double CFL_number = 0.4;
	double max_dt = 1.0e10;
	constexpr double tmax = 10.0 * (Lx / a0);
	constexpr int max_timesteps = 1e6;

	// Boundary conditions
	constexpr int nvars = RadSystem<TubeProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		// for x-axis:
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		// for y-, z- axes:
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			// periodic
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Read max_dt from parameter file
	amrex::ParmParse const pp;
	pp.query("max_dt", max_dt);

	// Problem initialization
	QuokkaSimulation<TubeProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.reconstructionOrder_ = 3;	       // PPM
	sim.stopTime_ = tmax;
	sim.cflNumber_ = CFL_number;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;
	sim.maxDt_ = max_dt;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// compute error norm
	std::vector<double> xs(nx);
	std::vector<double> xs_norm(nx);
	std::vector<double> rho_arr(nx);
	std::vector<double> Mach_arr(nx);

	for (int i = 0; i < nx; ++i) {
		xs.at(i) = position[i];
		xs_norm.at(i) = position[i] / Lx;

		double const rho = values.at(RadSystem<TubeProblem>::gasDensity_index)[i];
		double const x1GasMom = values.at(RadSystem<TubeProblem>::x1GasMomentum_index)[i];
		double const vx = x1GasMom / rho;

		rho_arr.at(i) = rho;
		Mach_arr.at(i) = vx / a0;
	}

	// read in exact solution
	std::vector<double> x_exact;
	std::vector<double> x_exact_scaled;
	std::vector<double> rho_exact;
	std::vector<double> Mach_exact;

	std::string const filename = "../extern/pressure_tube/optically_thin_wind.txt";
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
		auto x = values.at(0);	  // position
		auto rho = values.at(1);  // density
		auto Mach = values.at(2); // Mach number

		x_exact.push_back(x);
		x_exact_scaled.push_back(x * Lx);
		rho_exact.push_back(rho * rho0);
		Mach_exact.push_back(Mach);
	}

	// interpolate exact solution to simulation grid
	std::vector<double> Mach_interp(nx);

	interpolate_arrays(xs_norm.data(), Mach_interp.data(), nx, x_exact.data(), Mach_exact.data(), static_cast<int>(x_exact.size()));

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(Mach_arr[i] - Mach_interp[i]);
		sol_norm += std::abs(Mach_interp[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.002;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
	// Plot density
	std::map<std::string, std::string> rhoexact_args;
	std::unordered_map<std::string, std::string> rho_args;
	rhoexact_args["label"] = "exact solution";
	rhoexact_args["color"] = "C0";
	rho_args["label"] = "simulation";
	rho_args["color"] = "C1";
	matplotlibcpp::plot(x_exact_scaled, rho_exact, rhoexact_args);
	matplotlibcpp::scatter(xs, rho_arr, 1.0, rho_args);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("x (cm)");
	matplotlibcpp::ylabel("density");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radiation_force_tube.pdf");

	// plot velocity
	int const s = nx / 64; // stride
	std::map<std::string, std::string> vx_exact_args;
	std::unordered_map<std::string, std::string> vx_args;
	vx_exact_args["label"] = "exact solution";
	vx_exact_args["color"] = "C0";
	vx_args["label"] = "simulation";
	vx_args["marker"] = "o";
	vx_args["color"] = "C1";
	matplotlibcpp::clf();
	matplotlibcpp::plot(x_exact_scaled, Mach_exact, vx_exact_args);
	matplotlibcpp::scatter(strided_vector_from(xs, s), strided_vector_from(Mach_arr, s), 10.0, vx_args);
	matplotlibcpp::legend();
	// matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("Mach number");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radiation_force_tube_vel.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}
