//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro3d_blast.cpp
/// \brief Defines a test problem for a 3D explosion.
///

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"

struct SedovProblem {
};

// if false, use octant symmetry instead
constexpr bool simulate_full_box = false;

bool test_passes = false; // if one of the energy checks fails, set to false

template <> struct quokka::EOS_Traits<SedovProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<SedovProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<SedovProblem> {
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

// declare global variables
const double rho = 1.0;	   // g cm^-3
double E_blast = 0.851072; // ergs

template <> void QuokkaSimulation<SedovProblem>::preCalculateInitialConditions()
{
	if constexpr (!simulate_full_box) {
		E_blast /= 8.0; // only one octant, so 1/8 of the total energy
	}
}

template <> void QuokkaSimulation<SedovProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// initialize a Sedov test problem using parameters from
	// Richard Klein and J. Bolstad
	// [Reference: J.R. Kamm and F.X. Timmes, On Efficient Generation of
	//   Numerically Robust Sedov Solutions, LA-UR-07-2849.]

	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
	double rho_copy = rho;
	double E_blast_copy = E_blast;

	amrex::Real x0 = NAN;
	amrex::Real y0 = NAN;
	amrex::Real z0 = NAN;
	if constexpr (simulate_full_box) {
		x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
		y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
		z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);
	} else {
		x0 = 0.;
		y0 = 0.;
		z0 = 0.;
	}

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		static_assert(!simulate_full_box, "single-cell initialization is only "
						  "implemented for octant symmetry!");
		double rho_e = NAN;
		if ((i == 0) && (j == 0) && (k == 0)) {
			rho_e = E_blast_copy / cell_vol;
		} else {
			rho_e = 1.0e-10 * (E_blast_copy / cell_vol);
		}

		AMREX_ASSERT(!std::isnan(rho_copy));
		AMREX_ASSERT(!std::isnan(rho_e));

		for (int n = 0; n < state_cc.nComp(); ++n) {
			state_cc(i, j, k, n) = 0.; // zero fill all components
		}

		state_cc(i, j, k, HydroSystem<SedovProblem>::density_index) = rho_copy;
		state_cc(i, j, k, HydroSystem<SedovProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<SedovProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<SedovProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<SedovProblem>::energy_index) = rho_e;
	});
}

template <> void QuokkaSimulation<SedovProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.1; // gradient refinement threshold
	const amrex::Real P_min = 1.0e-3;      // minimum pressure for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const P = HydroSystem<SedovProblem>::ComputePressure(state, i, j, k);

			amrex::Real const P_xplus = HydroSystem<SedovProblem>::ComputePressure(state, i + 1, j, k);
			amrex::Real const P_xminus = HydroSystem<SedovProblem>::ComputePressure(state, i - 1, j, k);
			amrex::Real const P_yplus = HydroSystem<SedovProblem>::ComputePressure(state, i, j + 1, k);
			amrex::Real const P_yminus = HydroSystem<SedovProblem>::ComputePressure(state, i, j - 1, k);
			amrex::Real const P_zplus = HydroSystem<SedovProblem>::ComputePressure(state, i, j, k + 1);
			amrex::Real const P_zminus = HydroSystem<SedovProblem>::ComputePressure(state, i, j, k - 1);

			amrex::Real const del_x = std::max(std::abs(P_xplus - P), std::abs(P - P_xminus));
			amrex::Real const del_y = std::max(std::abs(P_yplus - P), std::abs(P - P_yminus));
			amrex::Real const del_z = std::max(std::abs(P_zplus - P), std::abs(P - P_zminus));

			amrex::Real const gradient_indicator = std::max({del_x, del_y, del_z}) / P;

			if ((gradient_indicator > eta_threshold) && (P > P_min)) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <> void QuokkaSimulation<SedovProblem>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	// check conservation of total energy
	amrex::Real const Egas0 = initSumCons[RadSystem<SedovProblem>::gasEnergy_index];
	amrex::Real const Egas = state_new_cc_[0].sum(RadSystem<SedovProblem>::gasEnergy_index) * vol;

	// compute kinetic energy
	amrex::MultiFab Ekin_mf(boxArray(0), DistributionMap(0), 1, 0);
	for (amrex::MFIter iter(state_new_cc_[0]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = state_new_cc_[0].const_array(iter);
		auto const &ekin = Ekin_mf.array(iter);
		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			// compute kinetic energy
			Real rho = state(i, j, k, HydroSystem<SedovProblem>::density_index);
			Real px = state(i, j, k, HydroSystem<SedovProblem>::x1Momentum_index);
			Real py = state(i, j, k, HydroSystem<SedovProblem>::x2Momentum_index);
			Real pz = state(i, j, k, HydroSystem<SedovProblem>::x3Momentum_index);
			Real psq = px * px + py * py + pz * pz;
			ekin(i, j, k) = psq / (2.0 * rho) * vol;
		});
	}
	amrex::Real const Ekin = Ekin_mf.sum(0);

	amrex::Real const frac_Ekin = Ekin / Egas;
	amrex::Real const frac_Ekin_exact = 0.218729;

	amrex::Real const abs_err = (Egas - Egas0);
	amrex::Real const rel_err = abs_err / Egas0;

	amrex::Real const rel_err_Ekin = frac_Ekin - frac_Ekin_exact;

	amrex::Print() << "\nInitial energy = " << Egas0 << '\n';
	amrex::Print() << "Final energy = " << Egas << '\n';
	amrex::Print() << "\tabsolute conservation error = " << abs_err << '\n';
	amrex::Print() << "\trelative conservation error = " << rel_err << '\n';
	amrex::Print() << "\tkinetic energy = " << Ekin << '\n';
	amrex::Print() << "\trelative K.E. error = " << rel_err_Ekin << '\n';
	amrex::Print() << '\n';

	bool E_test_passes = false;  // does total energy test pass?
	bool KE_test_passes = false; // does kinetic energy test pass?

	if ((std::abs(rel_err) > 2.0e-15) || std::isnan(rel_err)) {
		// note that this tolerance is appropriate for a 256^3 grid
		// it may need to be modified for coarser resolutions
		amrex::Print() << "Energy not conserved to machine precision!\n";
		E_test_passes = false;
	} else {
		amrex::Print() << "Energy conservation is OK.\n";
		E_test_passes = true;
	}

	if ((std::abs(rel_err_Ekin) > 0.01) || std::isnan(rel_err_Ekin)) {
		amrex::Print() << "Kinetic energy production is incorrect by more than 1 percent!\n";
		KE_test_passes = false;
	} else {
		amrex::Print() << "Kinetic energy production is OK.\n";
		KE_test_passes = true;
	}

	// if both tests pass, then overall pass
	test_passes = E_test_passes && KE_test_passes;
	amrex::Print() << "\n";
}

auto problem_main() -> int
{
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<SedovProblem>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<SedovProblem>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<SedovProblem>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<SedovProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if constexpr (simulate_full_box) { // periodic boundaries
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			} else { // octant symmetry
				if (isNormalComp(n, i)) {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_odd);
				} else {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(i, amrex::BCType::reflect_even);
				}
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<SedovProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.stopTime_ = 1.0;	      // seconds
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	int status = 1;
	if (test_passes) {
		status = 0;
	} else {
		status = 1;
	}

	return status;
}
