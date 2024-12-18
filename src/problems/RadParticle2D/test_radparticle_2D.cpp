/// \file test_radparticle_2D.cpp
/// \brief Defines a 2D test problem for radiating particles.
///

#include "test_radparticle_2D.hpp"
#include "QuokkaSimulation.hpp"
#include "util/fextract.hpp"

struct ParticleProblem {
};

constexpr int nGroups_ = 1;

constexpr double erad_floor = 1.0e-15;
constexpr double initial_Erad = 1.0e-10;
constexpr double initial_Egas = 1.0e-10;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = 1.0;	   // reduced speed of light
constexpr double kappa0 = 1.0e-10; // opacity
constexpr double rho = 1.0;

const double lum1 = 1.0;

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
};

template <> void QuokkaSimulation<ParticleProblem>::createInitialRadParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 2 + nGroups_; // birth_time death_time lum1
	RadParticles->SetVerbose(1);
	RadParticles->InitFromAsciiFile("RadParticles2D.txt", nreal_extra, nullptr);
}

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

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// compute total radiation energy
	const double total_Erad_over_vol = sim.state_new_cc_[0].sum(RadSystem<ParticleProblem>::radEnergy_index);
	const double dx = sim.Geom(0).CellSize(0);
	const double dy = sim.Geom(0).CellSize(1);
	const double dvol = dx * dy;
	const double total_Erad = total_Erad_over_vol * dvol;
	const double total_Erad_exact = 4.0 * lum1 * sim.tNew_[0];
	const double rel_err = std::abs(total_Erad - total_Erad_exact) / total_Erad_exact;
	amrex::Print() << "Total radiation energy = " << total_Erad << "\n";

	int status = 1;
	const double rel_err_tol = 1.0e-5;
	if (rel_err < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err << "\n";

	// Cleanup and exit
	amrex::Print() << "Finished." << "\n";
	return status;
}
