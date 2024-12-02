//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_streaming.cpp
/// \brief Defines a test problem for radiation in the free-streaming regime.
///

#include "test_radiation_streaming.hpp"
#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "QuokkaSimulation.hpp"
#include "particles/RadParticles.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"

struct StreamingProblem {
};

constexpr double initial_Erad = 1.0e-5;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = 1.0;	   // reduced speed of light
constexpr double kappa0 = 1.0e-10; // opacity
constexpr double rho = 1.0;

const double lum1 = c * 2.0 * 1.0 * 1.0; // L = c * 2 * r * E
// const double lum2 = c * 2.0 * PI * 1.0 * 1.0; // L = c * 2 * PI * r * E

template <> struct quokka::EOS_Traits<StreamingProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<StreamingProblem> {
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
	static constexpr double radiation_constant = 1.0;
};

template <> struct RadSystem_Traits<StreamingProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = initial_Erad;
	static constexpr int beta_order = 0;
	static constexpr bool do_rad_particles = true;
};

// template <> 
// void QuokkaSimulation<StreamingProblem>::createInitialRadParticles()
// {
// 	// read particles from ASCII file
// 	const int nreal_extra = 3; // mass birth_time death_time
// 	RadParticles->SetVerbose(0);
// 	RadParticles->InitFromAsciiFile("RadParticles.txt", nreal_extra, nullptr);

// 	// const double mass = 1.0;
// 	// // MyParticleContainer::ParticleInitData pdata = {{mass}, {},{},{}};
// 	// // MyParticleContainer::ParticleInitData pdata = {{mass}, {}, {}, {}};
// 	// // MyParticleContainer::ParticleInitData pdata = {{mass}, {0.0, 0.0, 0.0}, {}, {}};
// 	// // RadParticles->InitRandom(1, 333, pdata, true);

//   // // MyParticleContainer::ParticleInitData pdata = {{mass, AMREX_D_DECL(1.0, 2.0, 3.0), AMREX_D_DECL(0.0, 0.0, 0.0)}, {},{},{}};
//   // quokka::RadParticleContainer::ParticleInitData pdata = {{mass}, {},{},{}};
//   // // myPC.InitRandom(num_particles, iseed, pdata, serialize);
// 	// RadParticles->InitRandom(1, 333, pdata, true);

// 	// RadParticles->Redistribute();
// }

// template <> void QuokkaSimulation<StreamingProblem>::createInitialParticles()
// {
// 	// read particles from ASCII file
// 	const int nreal_extra = 3; // mass birth_time death_time
// 	CICParticles->SetVerbose(1);
// 	CICParticles->InitFromAsciiFile("RadParticles.txt", nreal_extra, nullptr);
// }

template <> void QuokkaSimulation<StreamingProblem>::createInitialParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 3; // mass vx vy vz
	CICParticles->SetVerbose(1);
	CICParticles->InitFromAsciiFile("RadParticles.txt", nreal_extra, nullptr);
}

template <>
void RadSystem<StreamingProblem>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange,
						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & dx,
						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/,
						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
{
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (i == 32) {
			// src = lum / c / (dx[0] * dx[1]);
			const double src = lum1 / c / (dx[0]);
			radEnergySource(i, j, k, 0) += src;
		}
	});
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<StreamingProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;
	const auto Egas0 = initial_Egas;

	// calculate radEnergyFractions
	quokka::valarray<amrex::Real, Physics_Traits<StreamingProblem>::nGroups> radEnergyFractions{};
	for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
		radEnergyFractions[g] = 1.0 / Physics_Traits<StreamingProblem>::nGroups;
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad0 * radEnergyFractions[g];
			state_cc(i, j, k, RadSystem<StreamingProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<StreamingProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<StreamingProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<StreamingProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<StreamingProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double dt_max = 1e-2;
	const double tmax = 0.8;
	const int max_timesteps = 5000;

	// Boundary conditions
	constexpr int nvars = RadSystem<StreamingProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::foextrap);
		BCs_cc[n].setHi(0, amrex::BCType::foextrap);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<StreamingProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = tmax;
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
	std::vector<double> erad(nx);
	std::vector<double> erad_exact(nx);
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		erad_exact.at(i) = (x <= chat * tmax) ? 1.0 : 0.0;
		double erad_sim = 0.0;
		for (int g = 0; g < Physics_Traits<StreamingProblem>::nGroups; ++g) {
			erad_sim += values.at(RadSystem<StreamingProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		erad.at(i) = erad_sim;
	}

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(erad[i] - erad_exact[i]);
		sol_norm += std::abs(erad_exact[i]);
	}

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.01;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	matplotlibcpp::ylim(-0.1, 2.6);

	std::map<std::string, std::string> erad_args;
	std::map<std::string, std::string> erad_exact_args;
	erad_args["label"] = "numerical solution";
	erad_exact_args["label"] = "exact solution";
	erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, erad, erad_args);
	// matplotlibcpp::plot(xs, erad_exact, erad_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./radiation_streaming.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
