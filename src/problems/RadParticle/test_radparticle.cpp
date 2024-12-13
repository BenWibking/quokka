/// \file test_radparticle.cpp
/// \brief Defines a 1D test problem for radiating particles.
///

#include "test_radparticle.hpp"
#include "AMReX.H"
#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"

struct ParticleProblem {
};

constexpr int nGroups_ = 3;

constexpr double erad_floor = 1.0e-15;
constexpr double initial_Erad = 1.0e-5;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = 1.0;	   // reduced speed of light
constexpr double kappa0 = 1.0e-10; // opacity
constexpr double rho = 1.0;

const double lum1 = 1.0;
const double lum2 = 0.5;
const double lum3 = 0.0;

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
	static constexpr int nGroups = nGroups_; // number of radiation groups
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
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr double energy_unit = 1.0;
	static constexpr amrex::GpuArray<double, nGroups_ + 1> radBoundaries{0.001, 1.0, 3.0, 100.0};
};

template <> void QuokkaSimulation<ParticleProblem>::createInitialRadParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 3 + nGroups_; // mass birth_time death_time lum1 lum2 lum3
	RadParticles->InitFromAsciiFile("RadParticles.txt", nreal_extra, nullptr);
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

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ParticleProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double rho,
								 const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = kappa0 / rho;
	}
	return exponents_and_values;
}

template <> void QuokkaSimulation<ParticleProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto Erad0 = initial_Erad;
	const auto Egas0 = initial_Egas;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		for (int g = 0; g < nGroups_; ++g) {
			state_cc(i, j, k, RadSystem<ParticleProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad0;
			state_cc(i, j, k, RadSystem<ParticleProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<ParticleProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
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
	// sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.5, true);
	const int nx = static_cast<int>(position.size());
	const double dx = sim.Geom(0).CellSize(0);

	// compute error norm
	std::vector<double> Erad_group0(nx);
	std::vector<double> Erad_group1(nx);
	std::vector<double> Erad_group2(nx);
	// std::vector<double> erad_exact(nx);
	std::vector<double> xs(nx);
	double tot_lum_group0 = 0.0;
	double tot_lum_group1 = 0.0;
	double tot_lum_group2 = 0.0;
	for (int i = 0; i < nx; ++i) {
		amrex::Real const x = position[i];
		xs.at(i) = x;
		// erad_exact.at(i) = (x <= chat * tmax) ? 1.0 : 0.0;
		Erad_group0.at(i) = values.at(RadSystem<ParticleProblem>::radEnergy_index)[i];
		Erad_group1.at(i) = values.at(RadSystem<ParticleProblem>::radEnergy_index + Physics_NumVars::numRadVars)[i];
		Erad_group2.at(i) = values.at(RadSystem<ParticleProblem>::radEnergy_index + 2 * Physics_NumVars::numRadVars)[i];
		tot_lum_group0 += Erad_group0.at(i) * dx;
		tot_lum_group1 += Erad_group1.at(i) * dx;
		tot_lum_group2 += Erad_group2.at(i) * dx;
	}

	const double tmax = sim.tNew_[0];
	const double lum_exact_group0 = lum1 * tmax;
	const double lum_exact_group1 = lum2 * tmax;
	const double lum_exact_group2 = lum3 * tmax;

	const double err_norm = std::abs(tot_lum_group0 - lum_exact_group0) + std::abs(tot_lum_group1 - lum_exact_group1) + std::abs(tot_lum_group2 - lum_exact_group2);
	const double sol_norm = lum_exact_group0 + lum_exact_group1 + lum_exact_group2;

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.05;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
	// Plot results
	matplotlibcpp::clf();
	matplotlibcpp::ylim(-0.05, 1.4);

	std::map<std::string, std::string> erad_args;
	// std::map<std::string, std::string> erad_exact_args;
	erad_args["label"] = "Erad group 0";
	// erad_exact_args["label"] = "exact solution";
	// erad_exact_args["linestyle"] = "--";
	matplotlibcpp::plot(xs, Erad_group0, erad_args);
	erad_args["label"] = "Erad group 1";
	matplotlibcpp::plot(xs, Erad_group1, erad_args);
	erad_args["label"] = "Erad group 2";
	matplotlibcpp::plot(xs, Erad_group2, erad_args);
	// matplotlibcpp::plot(xs, erad_exact, erad_exact_args);

	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("t = {:.4f}", sim.tNew_[0]));
	matplotlibcpp::save("./radparticle.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}
