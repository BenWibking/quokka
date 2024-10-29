//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file blast.cpp
/// \brief Implements the random blast problem with radiative cooling.
///
#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "AMReX_iMultiFab.H"

#include "QuokkaSimulation.hpp"
#include "blast.hpp"
#include "cooling/GrackleLikeCooling.hpp"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/quadrature.hpp"
#include "physics_info.hpp"

using amrex::Real;

struct RandomBlast {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double seconds_in_year = 3.1536e7; // s == 1 yr
constexpr double parsec_in_cm = 3.086e18;    // cm == 1 pc
constexpr double solarmass_in_g = 1.99e33;   // g == 1 Msun
constexpr double keV_in_ergs = 1.60218e-9;   // ergs == 1 keV
constexpr double m_H = C::m_p + C::m_e;	     // mass of hydrogen atom

template <> struct Physics_Traits<RandomBlast> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct quokka::EOS_Traits<RandomBlast> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

constexpr Real Tgas0 = 1.0e4;								// K
constexpr Real nH0 = 0.1;								// cm^-3
constexpr Real rho0 = nH0 * (m_H / quokka::GrackleLikeCooling::cloudy_H_mass_fraction); // g cm^-3

template <> struct SimulationData<RandomBlast> {
	std::unique_ptr<amrex::TableData<Real, 1>> blast_x;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z;

	int nblast = 0;
	int SN_counter_cumulative = 0;
	Real SN_rate_per_vol = NAN; // rate per unit time per unit volume
	Real E_blast = 1.0e51;	    // ergs
	Real M_ejecta = 0;	    // 10.0 * Msun; // g

	Real refine_threshold = 1.0; // gradient refinement threshold
	int use_periodic_bc = 1;     // default is periodic
};

template <> void QuokkaSimulation<RandomBlast>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = rho0;
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = quokka::EOS<RandomBlast>::ComputeEintFromTgas(rho, Tgas0);
		Real const Egas = Eint;
		Real const scalar_density = 0;

		state_cc(i, j, k, HydroSystem<RandomBlast>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<RandomBlast>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<RandomBlast>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<RandomBlast>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<RandomBlast>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<RandomBlast>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<RandomBlast>::scalar0_index) = scalar_density;
	});
}

void injectEnergy(amrex::MultiFab &mf, amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<Real, AMREX_SPACEDIM> const &prob_hi,
		  amrex::GpuArray<Real, AMREX_SPACEDIM> const &dx, SimulationData<RandomBlast> const &userData)
{
	// inject energy into cells with stochastic sampling
	const BL_PROFILE("QuokkaSimulation::injectEnergy()");

	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]); // cm^3
	const Real rho_eint_blast = userData.E_blast / cell_vol;   // ergs cm^-3
	const Real rho_ejecta = userData.M_ejecta / cell_vol;	   // g cm^-3

	const Real Lx = prob_hi[0] - prob_lo[0];
	const Real Ly = prob_hi[1] - prob_lo[1];
	const Real Lz = prob_hi[2] - prob_lo[2];

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &px = userData.blast_x->table();
		auto const &py = userData.blast_y->table();
		auto const &pz = userData.blast_z->table();
		const int np = userData.nblast;
		const int use_periodic_bc = userData.use_periodic_bc;

		const Real r_scale = 8.0 * dx[0]; // TODO(ben): cannot be based on local dx when using AMR!
		const Real normfac = 1.0 / std::pow(r_scale, 3);

		auto kern = [=] AMREX_GPU_DEVICE(const Real x, const Real y, const Real z) {
			const Real r = std::sqrt(x * x + y * y + z * z);
			return kernel_wendland_c2(r / r_scale);
		};

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real xc = prob_lo[0] + static_cast<Real>(i) * dx[0];
			const Real yc = prob_lo[1] + static_cast<Real>(j) * dx[1];
			const Real zc = prob_lo[2] + static_cast<Real>(k) * dx[2];

			for (int n = 0; n < np; ++n) {
				Real x0 = NAN;
				Real y0 = NAN;
				Real z0 = NAN;
				if (use_periodic_bc == 1) {
					// compute distance to nearest periodic image
					x0 = std::remainder(xc - px(n), Lx);
					y0 = std::remainder(yc - py(n), Ly);
					z0 = std::remainder(zc - pz(n), Lz);
				} else {
					x0 = (xc - px(n));
					y0 = (yc - py(n));
					z0 = (zc - pz(n));
				}

				// integrate each particle kernel over the cell
				const Real weight = normfac * quad_3d(kern, x0, x0 + dx[0], y0, y0 + dx[1], z0, z0 + dx[2]);

				state(i, j, k, HydroSystem<RandomBlast>::density_index) += weight * rho_ejecta;
				state(i, j, k, HydroSystem<RandomBlast>::scalar0_index) += weight * rho_ejecta;
				state(i, j, k, HydroSystem<RandomBlast>::energy_index) += weight * rho_eint_blast;
				state(i, j, k, HydroSystem<RandomBlast>::internalEnergy_index) += weight * rho_eint_blast;
			}
		});
	}
}

template <> void QuokkaSimulation<RandomBlast>::computeBeforeTimestep()
{
	// compute how many (and where) SNe will go off on the this coarse timestep
	// sample from Poisson distribution
	const Real dt_coarse = dt_[0];
	const Real domain_vol = geom[0].ProbSize();
	const Real expectation_value = userData_.SN_rate_per_vol * domain_vol * dt_coarse;

	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));
	if (count > 0) {
		amrex::Print() << "\t" << count << " SNe to be exploded.\n";
	}

	// resize particle arrays
	amrex::Array<int, 1> const lo{0};
	amrex::Array<int, 1> const hi{count};
	userData_.blast_x = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.blast_y = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.blast_z = std::make_unique<amrex::TableData<Real, 1>>(lo, hi, amrex::The_Pinned_Arena());
	userData_.nblast = count;
	userData_.SN_counter_cumulative += count;

	// for each, sample location at random
	auto const &px = userData_.blast_x->table();
	auto const &py = userData_.blast_y->table();
	auto const &pz = userData_.blast_z->table();
	for (int i = 0; i < count; ++i) {
		px(i) = geom[0].ProbLength(0) * amrex::Random();
		py(i) = geom[0].ProbLength(1) * amrex::Random();
		pz(i) = geom[0].ProbLength(2) * amrex::Random();
	}

	// TODO(ben): need to force refinement to highest level for cells near particles
}

template <> void QuokkaSimulation<RandomBlast>::computeAfterLevelAdvance(int lev, Real /*time*/, Real /*dt_lev*/, int /*ncycle*/)
{
	// compute operator split physics
	injectEnergy(state_new_cc_[lev], geom[lev].ProbLoArray(), geom[lev].ProbHiArray(), geom[lev].CellSizeArray(), userData_);
}

template <> void QuokkaSimulation<RandomBlast>::computeAfterTimestep()
{
	// check conservation of mass
	static auto const &dx = geom[0].CellSizeArray();
	static Real const cvol = AMREX_D_TERM(dx[0], +dx[1], +dx[2]);
	static Real const initial_mass = cvol * state_new_cc_[0].sum(HydroSystem<RandomBlast>::density_index);

	const Real mass = cvol * state_new_cc_[0].sum(HydroSystem<RandomBlast>::density_index);
	const Real cons_err = (mass - initial_mass) / initial_mass;

	amrex::Print() << "Initial mass = " << initial_mass << "\n"
		       << "Final mass = " << mass << "\n"
		       << "Relative error = " << cons_err << "\n";

	if (std::abs(cons_err) > 1.0e-10) {
		// write out FABs with ghost zones
		// amrex::writeFabs(state_new_cc_[0], "state_new_" + std::to_string(istep[0]));
		// abort
		amrex::Abort("mass nonconservation detected!");
	}
}

template <> void QuokkaSimulation<RandomBlast>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		auto tables = grackleTables_.const_tables();

		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real const rho = state(i, j, k, HydroSystem<RandomBlast>::density_index);
				Real const x1Mom = state(i, j, k, HydroSystem<RandomBlast>::x1Momentum_index);
				Real const x2Mom = state(i, j, k, HydroSystem<RandomBlast>::x2Momentum_index);
				Real const x3Mom = state(i, j, k, HydroSystem<RandomBlast>::x3Momentum_index);
				Real const Egas = state(i, j, k, HydroSystem<RandomBlast>::energy_index);
				Real const Eint = RadSystem<RandomBlast>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
				Real const Tgas = ComputeTgasFromEgas(rho, Eint, HydroSystem<RandomBlast>::gamma_, tables);

				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

template <> void QuokkaSimulation<RandomBlast>::ErrorEst(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	const Real q_min = 1e-5 * rho0; // minimum density for refinement
	const Real eta_threshold = userData_.refine_threshold;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<RandomBlast>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real const q = state(i, j, k, nidx);
			Real const q_xplus = state(i + 1, j, k, nidx);
			Real const q_xminus = state(i - 1, j, k, nidx);
			Real const q_yplus = state(i, j + 1, k, nidx);
			Real const q_yminus = state(i, j - 1, k, nidx);
			Real const q_zplus = state(i, j, k + 1, nidx);
			Real const q_zminus = state(i, j, k - 1, nidx);

			Real const del_x = 0.5 * (q_xplus - q_xminus);
			Real const del_y = 0.5 * (q_yplus - q_yminus);
			Real const del_z = 0.5 * (q_zplus - q_zminus);
			Real const gradient_indicator = std::sqrt(del_x * del_x + del_y * del_y + del_z * del_z) / q;

			if ((gradient_indicator > eta_threshold) && (q > q_min)) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// This problem is only implemented in CGS units because the Grackle cooling tables are in CGS units.
	static_assert(Physics_Traits<RandomBlast>::unit_system == UnitSystem::CGS);

	// read parameters
	amrex::ParmParse const pp;

	// read in SN rate
	Real SN_rate_per_vol = NAN;
	pp.query("SN_rate_per_volume", SN_rate_per_vol); // yr^-1 kpc^-3
	SN_rate_per_vol /= seconds_in_year;
	SN_rate_per_vol /= std::pow(1.0e3 * parsec_in_cm, 3);
	AMREX_ALWAYS_ASSERT(!std::isnan(SN_rate_per_vol));

	// read in refinement threshold (relative gradient in density)
	Real refine_threshold = 0.1;
	pp.query("refine_threshold", refine_threshold); // dimensionless

	// use periodic boundary conditions or not
	int use_periodic_bc = 0;
	pp.query("use_periodic_bc", use_periodic_bc);

	// Problem initialization
	auto isNormalComp = [=](int n, int dim) {
		if ((n == HydroSystem<RandomBlast>::x1Momentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == HydroSystem<RandomBlast>::x2Momentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == HydroSystem<RandomBlast>::x3Momentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int nvars = HydroSystem<RandomBlast>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (use_periodic_bc == 1) { // periodic boundaries
				BCs_cc[n].setLo(idim, amrex::BCType::int_dir);
				BCs_cc[n].setHi(idim, amrex::BCType::int_dir);
			} else { // reflecting boundaries
				if (isNormalComp(n, idim)) {
					BCs_cc[n].setLo(idim, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(idim, amrex::BCType::reflect_odd);
				} else {
					BCs_cc[n].setLo(idim, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(idim, amrex::BCType::reflect_even);
				}
			}
		}
	}

	QuokkaSimulation<RandomBlast> sim(BCs_cc);
	sim.densityFloor_ = 1.0e-5 * rho0; // density floor (to prevent vacuum)
	sim.userData_.SN_rate_per_vol = SN_rate_per_vol;
	sim.userData_.refine_threshold = refine_threshold;
	sim.userData_.use_periodic_bc = use_periodic_bc;

	// Set initial conditions
	sim.setInitialConditions();

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// run simulation
	sim.evolve();

	// print injected energy, injected mass
	const Real E_in_cumulative = static_cast<Real>(sim.userData_.SN_counter_cumulative) * sim.userData_.E_blast;
	const Real M_in_cumulative = static_cast<Real>(sim.userData_.SN_counter_cumulative) * sim.userData_.M_ejecta;
	amrex::Print() << "Cumulative injected energy = " << E_in_cumulative << "\n";
	amrex::Print() << "Cumulative injected mass = " << M_in_cumulative << "\n";

	// Cleanup and exit
	const int status = 0;
	return status;
}
