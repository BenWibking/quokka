//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_force.cpp
/// \brief Defines a test problem for radiation force terms.
///

#include <string>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_REAL.H"

#include "ArrayUtil.hpp"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "hydro_system.hpp"
#include "interpolate.hpp"
#include "physics_info.hpp"
#include "radiation_system.hpp"
#include "test_radiation_force.hpp"
#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

struct TubeProblem {
};

constexpr double kappa0 = 5.0;	     // cm^2 g^-1
constexpr double mu = 2.33 * C::m_u; // g
constexpr double gamma_gas = 1.0;    // isothermal gas EOS
constexpr double a0 = 0.2e5;	     // cm s^-1
constexpr double tau = 1.0e-6;	     // optical depth (dimensionless)

constexpr double rho0 = 1.0e5 * mu; // g cm^-3
constexpr double Mach0 = 1.1;	    // Mach number at wind base
constexpr double Mach1 = 2.128410288469465339;
constexpr double rho1 = (Mach0 / Mach1) * rho0;

constexpr double Frad0 = rho0 * a0 * c_light_cgs_ / tau; // erg cm^-2 s^-1
constexpr double g0 = kappa0 * Frad0 / c_light_cgs_;	 // cm s^{-2}
constexpr double Lx = (a0 * a0) / g0;			 // cm

template <> struct quokka::EOS_Traits<TubeProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = C::k_B;
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
};

template <> struct RadSystem_Traits<TubeProblem> {
	static constexpr double c_light = c_light_cgs_;
	static constexpr double c_hat = 10. * (Mach1 * a0);
	static constexpr double radiation_constant = radiation_constant_cgs_;
	static constexpr double Erad_floor = 0.;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr int beta_order = 1;
};

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/)
    -> quokka::valarray<double, Physics_Traits<TubeProblem>::nGroups>
{
	quokka::valarray<double, Physics_Traits<TubeProblem>::nGroups> kappaPVec{};
	for (int g = 0; g < nGroups_; ++g) {
		kappaPVec[g] = 0.; // no heating/cooling
	}
	return kappaPVec;
}

template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TubeProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/)
    -> quokka::valarray<double, Physics_Traits<TubeProblem>::nGroups>
{
	quokka::valarray<double, Physics_Traits<TubeProblem>::nGroups> kappaFVec{};
	for (int g = 0; g < nGroups_; ++g) {
		kappaFVec[g] = kappa0;
	}
	return kappaFVec;
}

// declare global variables
// initial conditions read from file
amrex::Gpu::HostVector<double> x_arr;
amrex::Gpu::HostVector<double> rho_arr;
amrex::Gpu::HostVector<double> Mach_arr;

amrex::Gpu::DeviceVector<double> x_arr_g;
amrex::Gpu::DeviceVector<double> rho_arr_g;
amrex::Gpu::DeviceVector<double> Mach_arr_g;

template <> void RadhydroSimulation<TubeProblem>::preCalculateInitialConditions()
{
	std::string filename = "../extern/pressure_tube/optically_thin_wind.txt";
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

		x_arr.push_back(x);
		rho_arr.push_back(rho);
		Mach_arr.push_back(Mach);
	}

	// copy to device
	x_arr_g.resize(x_arr.size());
	rho_arr_g.resize(rho_arr.size());
	Mach_arr_g.resize(Mach_arr.size());

	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, x_arr.begin(), x_arr.end(), x_arr_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, rho_arr.begin(), rho_arr.end(), rho_arr_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Mach_arr.begin(), Mach_arr.end(), Mach_arr_g.begin());
	amrex::Gpu::streamSynchronizeAll();
}

template <> void RadhydroSimulation<TubeProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
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
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate
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
	RadhydroSimulation<TubeProblem> sim(BCs_cc);

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
	auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx_tot = static_cast<int>(position0.size());

	// cut off at x = 1
	int nx = 0;
	for (int i = 0; i < nx_tot; ++i) {
		if (position[i] < 0.98e16) {
			nx++;
		}
	}

	// compute error norm
	std::vector<double> xs(nx);
	std::vector<double> rho_arr(nx);
	std::vector<double> rho_exact_arr(nx);
	std::vector<double> rho_err(nx);
	std::vector<double> vx_arr(nx);
	std::vector<double> vx_exact_arr(nx);
	std::vector<double> Frad_err(nx);

	auto const &x_ptr = x_arr_g.dataPtr();
	auto const &rho_ptr = rho_arr_g.dataPtr();
	auto const &Mach_ptr = Mach_arr_g.dataPtr();
	int const x_size = static_cast<int>(x_arr_g.size());

	for (int i = 0; i < nx; ++i) {
		amrex::Real const x_ = position[i] / Lx;
		amrex::Real const D_ = interpolate_value(x_, x_ptr, rho_ptr, x_size);
		amrex::Real const Mach_ = interpolate_value(x_, x_ptr, Mach_ptr, x_size);
		amrex::Real const _rho = D_ * rho0;
		amrex::Real const _vel = Mach_ * a0;

		xs.at(i) = position[i];
		double const rho_exact = _rho;
		// double x1GasMom_exact = values0.at(RadSystem<TubeProblem>::x1GasMomentum_index)[i];
		double const rho = values.at(RadSystem<TubeProblem>::gasDensity_index)[i];
		double const Frad = values.at(RadSystem<TubeProblem>::x1RadFlux_index)[i];
		double const x1GasMom = values.at(RadSystem<TubeProblem>::x1GasMomentum_index)[i];
		double const vx = x1GasMom / rho;
		double const vx_exact = _vel;

		vx_arr.at(i) = vx / a0;
		vx_exact_arr.at(i) = vx_exact / a0;
		Frad_err.at(i) = (Frad - Frad0) / Frad0;
		rho_err.at(i) = (rho - rho_exact) / rho_exact;
		rho_exact_arr.at(i) = rho_exact;
		rho_arr.at(i) = rho;
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(vx_arr[i] - vx_exact_arr[i]);
		sol_norm += std::abs(vx_exact_arr[i]);
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
	std::map<std::string, std::string> rho_args;
	std::unordered_map<std::string, std::string> rhoexact_args;
	rho_args["label"] = "simulation";
	rho_args["color"] = "C0";
	rhoexact_args["label"] = "exact solution";
	rhoexact_args["color"] = "C1";
	matplotlibcpp::plot(xs, rho_arr, rho_args);
	matplotlibcpp::scatter(xs, rho_exact_arr, 1.0, rhoexact_args);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("x (cm)");
	matplotlibcpp::ylabel("density");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radiation_force_tube.pdf");

	// plot velocity
	int const s = nx_tot / 64; // stride
	std::map<std::string, std::string> vx_args;
	std::unordered_map<std::string, std::string> vxexact_args;
	vxexact_args["label"] = "exact solution";
	vx_args["label"] = "simulation";
	vx_args["color"] = "C0";
	vxexact_args["marker"] = "o";
	vxexact_args["color"] = "C1";
	// vxexact_args["edgecolors"] = "k";
	matplotlibcpp::clf();
	matplotlibcpp::plot(xs, vx_arr, vx_args);
	matplotlibcpp::scatter(strided_vector_from(xs, s), strided_vector_from(vx_exact_arr, s), 10.0, vxexact_args);
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
