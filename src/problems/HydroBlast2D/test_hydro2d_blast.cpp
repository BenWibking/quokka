//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include "AMReX_Array.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_TagBox.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "test_hydro2d_blast.hpp"

struct BlastProblem {
};

template <> struct quokka::EOS_Traits<BlastProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<BlastProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> void QuokkaSimulation<BlastProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
		amrex::Real const R = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));

		double vx = 0.;
		double vy = 0.;
		double vz = 0.;
		double rho = 1.0;
		double P = NAN;

		if (R < 0.1) { // inside circle
			P = 10.;
		} else {
			P = 0.1;
		}

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(vy));
		AMREX_ASSERT(!std::isnan(vz));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto v_sq = vx * vx + vy * vy + vz * vz;
		const auto gamma = quokka::EOS_Traits<BlastProblem>::gamma;

		state_cc(i, j, k, HydroSystem<BlastProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<BlastProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<BlastProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<BlastProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<BlastProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * v_sq;

		// initialize radiation variables to zero
		state_cc(i, j, k, RadSystem<BlastProblem>::radEnergy_index) = 0;
		state_cc(i, j, k, RadSystem<BlastProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<BlastProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<BlastProblem>::x3RadFlux_index) = 0;
	});
}

template <> void QuokkaSimulation<BlastProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.1; // gradient refinement threshold
	const amrex::Real P_min = 1.0e-3;      // minimum pressure for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const P = HydroSystem<BlastProblem>::ComputePressure(state, i, j, k);
			amrex::Real const P_xplus = HydroSystem<BlastProblem>::ComputePressure(state, i + 1, j, k);
			amrex::Real const P_xminus = HydroSystem<BlastProblem>::ComputePressure(state, i - 1, j, k);
			amrex::Real const P_yplus = HydroSystem<BlastProblem>::ComputePressure(state, i, j + 1, k);
			amrex::Real const P_yminus = HydroSystem<BlastProblem>::ComputePressure(state, i, j - 1, k);

			amrex::Real const del_x = std::max(std::abs(P_xplus - P), std::abs(P - P_xminus));
			amrex::Real const del_y = std::max(std::abs(P_yplus - P), std::abs(P - P_yminus));

			amrex::Real const gradient_indicator = std::max(del_x, del_y) / std::max(P, P_min);

			if (gradient_indicator > eta_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// Problem parameters
	constexpr bool reflecting_boundary = true;

	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<BlastProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<BlastProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<BlastProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = QuokkaSimulation<BlastProblem>::nvarTotal_cc_;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if (reflecting_boundary) {
				if (isNormalComp(n, i)) {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
				} else {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
				}
			} else {
				// periodic
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<BlastProblem> sim(BCs_cc);

	sim.stopTime_ = 0.1; // 1.5;
	sim.cflNumber_ = 0.3;
	sim.maxTimesteps_ = 20000;
	sim.plotfileInterval_ = 2000;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
