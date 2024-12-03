/// \file test_radparticle_2D.cpp
/// \brief Defines a 2D test problem for radiating particles.
///

#include "test_radparticle_2D.hpp"
#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"

struct ParticleProblem {
};

constexpr double erad_floor = 1.0e-15;
constexpr double initial_Erad = 1.0e-5;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = 1.0;	   // reduced speed of light
constexpr double kappa0 = 1.0e-10; // opacity
constexpr double rho = 1.0;

template <> struct quokka::EOS_Traits<ParticleProblem> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<ParticleProblem> {
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

template <> struct RadSystem_Traits<ParticleProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = erad_floor;
	static constexpr int beta_order = 0;
};

template <> void QuokkaSimulation<ParticleProblem>::createInitialRadParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 3; // mass birth_time death_time
	RadParticles->SetVerbose(1);
	RadParticles->InitFromAsciiFile("RadParticles2D.txt", nreal_extra, nullptr);
}

// template <>
// void RadSystem<StreamingProblem>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange,
// 						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & dx,
// 						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_lo*/,
// 						   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & /*prob_hi*/, amrex::Real /*time*/)
// {
// 	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
// 		if (i == 32) {
// 			// src = lum / c / (dx[0] * dx[1]);
// 			const double src = lum1 / c / (dx[0]);
// 			radEnergySource(i, j, k, 0) += src;
// 		}
// 	});
// }

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ParticleProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> void QuokkaSimulation<ParticleProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;
	const auto Egas0 = initial_Egas;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, RadSystem<ParticleProblem>::radEnergy_index) = Erad0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::gasEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<ParticleProblem>::gasInternalEnergy_index) = Egas0;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<ParticleProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 1000;
	// const double Lx = 1.0;
	const double CFL_number = 0.8;
	const double dt_max = 1e-2;
	const int max_timesteps = 5000;

	// Boundary conditions
	constexpr int nvars = RadSystem<ParticleProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir); // periodic
		}
	}

	// Problem initialization
	QuokkaSimulation<ParticleProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.maxTimesteps_ = max_timesteps;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.25, true);
	const int nx = static_cast<int>(position.size());

	// compute error norm
	std::vector<double> erad(nx);
	// std::vector<double> erad_exact(nx);
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		// erad_exact.at(i) = (x <= chat * tmax) ? 1.0 : 0.0;
		erad.at(i) = values.at(RadSystem<ParticleProblem>::radEnergy_index)[i];
	}

	// double err_norm = 0.;
	// double sol_norm = 0.;
	// for (int i = 0; i < nx; ++i) {
	// 	err_norm += std::abs(erad[i] - erad_exact[i]);
	// 	sol_norm += std::abs(erad_exact[i]);
	// }

	// const double rel_err_norm = err_norm / sol_norm;
	// const double rel_err_tol = 0.05;
	// int status = 1;
	// if (rel_err_norm < rel_err_tol) {
	// 	status = 0;
	// }
	// amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	// matplotlibcpp::ylim(-0.1, 3.1);

	std::map<std::string, std::string> erad_args;
	std::map<std::string, std::string> erad_exact_args;
	erad_args["label"] = "numerical solution";
	erad_exact_args["label"] = "exact solution";
	erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, erad, erad_args);
	// matplotlibcpp::plot(xs, erad_exact, erad_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./radparticle_2D.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
