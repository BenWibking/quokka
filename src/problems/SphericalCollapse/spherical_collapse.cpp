//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file spherical_collapse.cpp
/// \brief Defines a test problem for pressureless spherical collapse.
///
#include <limits>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"

#include "AMReX_SPACE.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "spherical_collapse.hpp"

struct CollapseProblem {
};

template <> struct quokka::EOS_Traits<CollapseProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<CollapseProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<CollapseProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gravity_constant = 1.0;
};

template <> void QuokkaSimulation<CollapseProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

		double rho_min = 1.0e-5;
		double rho_max = 10.0;
		double R_sphere = 0.5;
		double R_smooth = 0.025;
		double rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		double P = 1.0e-1;

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		state_cc(i, j, k, HydroSystem<CollapseProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<CollapseProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CollapseProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CollapseProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<CollapseProblem>::energy_index) = quokka::EOS<CollapseProblem>::ComputeEintFromPres(rho, P);
		state_cc(i, j, k, HydroSystem<CollapseProblem>::internalEnergy_index) = quokka::EOS<CollapseProblem>::ComputeEintFromPres(rho, P);
	});
}

template <> void QuokkaSimulation<CollapseProblem>::createInitialParticles()
{
	// add particles at random positions in the box
	const bool generate_on_root_rank = true;
	const int iseed = 42;
	const int num_particles = 1000;
	const double total_particle_mass = 0.5; // about 0.1 of the total fluid mass
	const double particle_mass = total_particle_mass / static_cast<double>(num_particles);

	const quokka::CICParticleContainer::ParticleInitData pdata = {{particle_mass, 0, 0, 0}, {}, {}, {}}; // {mass vx vy vz}, empty, empty, empty
	CICParticles->InitRandom(num_particles, iseed, pdata, generate_on_root_rank);
}

template <> void QuokkaSimulation<CollapseProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	const Real q_min = 5.0; // minimum density for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<CollapseProblem>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real const q = state(i, j, k, nidx);
			if (q > q_min) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <> void QuokkaSimulation<CollapseProblem>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "gpot") {
		const int ncomp = ncomp_cc_in;
		auto const &phi_arr = phi[lev].const_arrays();
		auto output = mf.arrays();
		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output[bx](i, j, k, ncomp) = phi_arr[bx](i, j, k); });
	}
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<CollapseProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<CollapseProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<CollapseProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<CollapseProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
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
	QuokkaSimulation<CollapseProblem> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int status = 0;
	return status;
}
