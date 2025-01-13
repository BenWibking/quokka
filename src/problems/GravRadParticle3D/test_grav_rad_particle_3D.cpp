/// \file test_grav_rad_particle_3D.cpp
/// \brief Defines a 3D test problem for radiating particles with gravity.
///

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "AMReX.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Box.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"
#include "QuokkaSimulation.hpp"
#include "hydro/EOS.hpp"
#include "particles/PhysicsParticles.hpp"
#include "radiation/radiation_system.hpp"
#include "test_grav_rad_particle_3D.hpp"

struct ParticleProblem {
};

constexpr int nGroups_ = 1;

constexpr double erad_floor = 1.0e-15;
constexpr double initial_Erad = erad_floor;
constexpr double initial_Egas = 1.0e-5;
constexpr double c = 1.0;	   // speed of light
constexpr double chat = 0.1;	   // reduced speed of light
constexpr double kappa0 = 1.0e-20; // opacity
constexpr double rho = 1.0e-6;

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

template <> void QuokkaSimulation<ParticleProblem>::createInitialCICRadParticles()
{
	// read particles from ASCII file
	const int nreal_extra = 6 + nGroups_; // mass vx vy vz birth_time death_time lum1
	CICRadParticles->SetVerbose(1);
	CICRadParticles->InitFromAsciiFile("GravRadParticles3D.txt", nreal_extra, nullptr);
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

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<ParticleProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ParticleProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ParticleProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<ParticleProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ParticleProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ParticleProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	// Boundary conditions
	constexpr int nvars = RadSystem<ParticleProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
				BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<ParticleProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.radiationCflNumber_ = CFL_number;
	sim.maxDt_ = dt_max;
	sim.doPoissonSolve_ = 1; // enable self-gravity

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// compute total radiation energy
	const double total_Erad_over_vol = sim.state_new_cc_[0].sum(RadSystem<ParticleProblem>::radEnergy_index);
	const double dx = sim.Geom(0).CellSize(0);
	const double dy = sim.Geom(0).CellSize(1);
	const double dz = sim.Geom(0).CellSize(2);
	const double dvol = dx * dy * dz;
	const double total_Erad = total_Erad_over_vol * dvol;
	const double t_alive = std::min(0.5, sim.tNew_[0]);		   // particles only live for 0.5 time units
	const double total_Erad_exact = 2.0 * lum1 * t_alive * (chat / c); // two particles with luminosity lum1
	const double rel_err = std::abs(total_Erad - total_Erad_exact) / total_Erad_exact;

	// compute exact location of the particles
	// the particles are originally at (-0.5, 0) and (0.5, 0) and they move with
	// velocity 1/sqrt(2) in the y/-y direction the problem is designed such that
	// the particles will move in a circle with radius 0.5
	const double velocity = 1.0 / std::sqrt(2.0);
	const double radius = 0.5;
	const double theta = velocity * sim.tNew_[0] / radius;
	const double x1_exact = radius * std::cos(theta);
	const double y1_exact = radius * std::sin(theta);
	const double x2_exact = -x1_exact;
	const double y2_exact = -y1_exact;

	double position_error = NAN;
	double position_norm = NAN;

	// get particle positions
	quokka::CICRadParticleContainer<ParticleProblem> analysisPC{};
	amrex::Box const box(amrex::IntVect{AMREX_D_DECL(0, 0, 0)}, amrex::IntVect{AMREX_D_DECL(1, 1, 1)});
	amrex::Geometry const geom(box);
	amrex::BoxArray const boxArray(box);
	amrex::Vector<int> const ranks({0}); // workaround nvcc bug
	amrex::DistributionMapping const dmap(ranks);
	analysisPC.Define(geom, dmap, boxArray);
	auto *container =
	    sim.particleRegister_.getParticleDescriptor("CICRad_particles")->getParticleContainer<quokka::CICRadParticleContainer<ParticleProblem>>();
	analysisPC.copyParticles(*container);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		quokka::CICRadParticleIterator<ParticleProblem> const pIter(analysisPC, 0);
		if (pIter.isValid()) {
			const amrex::Long np = pIter.numParticles();
			auto &particles = pIter.GetArrayOfStructs();
			// copy particles from device to host
			quokka::CICRadParticleContainer<ParticleProblem>::ParticleType *pData = particles().data();
			amrex::Vector<quokka::CICRadParticleContainer<ParticleProblem>::ParticleType> pData_h(np);
			amrex::Gpu::copy(amrex::Gpu::deviceToHost, pData, std::next(pData, np), pData_h.begin()); // NOLINT
			quokka::CICRadParticleContainer<ParticleProblem>::ParticleType &p1 = pData_h[0];
			quokka::CICRadParticleContainer<ParticleProblem>::ParticleType &p2 = pData_h[1];

			// // Uncomment to print exact particle positions for debugging
			// amrex::Print() << "Exact particle 1 position: (" << x1_exact << ", " << y1_exact << ", " << 0.0 << ")\n";
			// amrex::Print() << "Exact particle 2 position: (" << x2_exact << ", " << y2_exact << ", " << 0.0 << ")\n";
			// amrex::Print() << "Particle 1 position: (" << p1.pos(0) << ", " << p1.pos(1) << ", " << p1.pos(2) << ")\n";
			// amrex::Print() << "Particle 2 position: (" << p2.pos(0) << ", " << p2.pos(1) << ", " << p2.pos(2) << ")\n";

			// We don't know which particle is which, so I compute the error for both possible assignments
			const double position_error_1 =
			    std::abs(p1.pos(0) - x1_exact) + std::abs(p1.pos(1) - y1_exact) + std::abs(p2.pos(0) - x2_exact) + std::abs(p2.pos(1) - y2_exact);
			const double position_error_2 =
			    std::abs(p1.pos(0) - x2_exact) + std::abs(p1.pos(1) - y2_exact) + std::abs(p2.pos(0) - x1_exact) + std::abs(p2.pos(1) - y1_exact);
			position_error = std::min(position_error_1, position_error_2);
			position_norm = std::abs(x1_exact) + std::abs(y1_exact) + std::abs(x2_exact) + std::abs(y2_exact);
		}
	}

	const double rel_position_error = position_error / position_norm;

	int status = 1;
	const double rel_err_tol = 1.0e-7;
	const double rel_position_error_tol = 1.0e-3;
	if (rel_err < rel_err_tol && rel_position_error < rel_position_error_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm on radiation energy = " << rel_err << "\n";
	amrex::Print() << "Relative L1 norm on particle positions = " << rel_position_error << "\n";

	// Cleanup and exit
	amrex::Print() << "Finished."
		       << "\n";
	return status;
}
