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
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"
#include "AMReX_iMultiFab.H"

#include "CloudyCooling.hpp"
#include "ODEIntegrate.hpp"
#include "RadhydroSimulation.hpp"
#include "blast.hpp"
#include "hydro_system.hpp"
#include "quadrature.hpp"

using amrex::Real;

struct RandomBlast {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double m_H = hydrogen_mass_cgs_;
constexpr double seconds_in_year = 3.154e7; // s
constexpr double parsec_in_cm = 3.086e18;   // cm
constexpr double Msun = 1.99e33;	    // g

template <> struct HydroSystem_Traits<RandomBlast> {
	static constexpr double gamma = 5. / 3.; // default value
	// if true, reconstruct e_int instead of pressure
	static constexpr bool reconstruct_eint = true;
};

template <> struct Physics_Traits<RandomBlast> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numPassiveScalars = 1; // number of passive scalars
};

constexpr Real Tgas0 = 1.0e4; // K
constexpr Real nH0 = 0.1;     // cm^-3
constexpr Real T_floor = 100.0;
constexpr Real rho0 = nH0 * (m_H / cloudy_H_mass_fraction); // g cm^-3

static Real SN_rate_per_vol = NAN;	   // rate per unit time per unit volume
static amrex::Real E_blast = 1.0e51;	   // ergs
static amrex::Real M_ejecta = 10.0 * Msun; // g

static amrex::TableData<Real, 1> blast_x({0}, {0}, amrex::The_Pinned_Arena());
static amrex::TableData<Real, 1> blast_y({0}, {0}, amrex::The_Pinned_Arena());
static amrex::TableData<Real, 1> blast_z({0}, {0}, amrex::The_Pinned_Arena());
static int nblast = 0;
static int SN_counter_cumulative = 0;

static Real refine_threshold = 1.0; // gradient refinement threshold

template <> void RadhydroSimulation<RandomBlast>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	auto tables = cloudyTables.const_tables();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real rho = rho0;
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = ComputeEgasFromTgas(rho0, Tgas0, HydroSystem<RandomBlast>::gamma_, tables);
		Real const Egas = RadSystem<RandomBlast>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);
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

struct ODEUserData {
	Real rho{};
	cloudyGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int
{
	// unpack user_data
	auto *udata = static_cast<ODEUserData *>(user_data);
	const Real rho = udata->rho;
	const Real gamma = HydroSystem<RandomBlast>::gamma_;
	cloudyGpuConstTables &tables = udata->tables;

	// check whether temperature is out-of-bounds
	const Real Tmin = 10.;
	const Real Tmax = 1.0e9;
	const Real Eint_min = ComputeEgasFromTgas(rho, Tmin, gamma, tables);
	const Real Eint_max = ComputeEgasFromTgas(rho, Tmax, gamma, tables);

	// compute temperature and cooling rate
	const Real Eint = y_data[0];

	if (Eint <= Eint_min) {
		// set cooling to value at Tmin
		y_rhs[0] = cloudy_cooling_function(rho, Tmin, tables);
	} else if (Eint >= Eint_max) {
		// set cooling to value at Tmax
		y_rhs[0] = cloudy_cooling_function(rho, Tmax, tables);
	} else {
		// ok, within tabulated cooling limits
		const Real T = ComputeTgasFromEgas(rho, Eint, gamma, tables);
		if (!std::isnan(T)) { // temp iteration succeeded
			y_rhs[0] = cloudy_cooling_function(rho, T, tables);
		} else { // temp iteration failed
			y_rhs[0] = NAN;
			return 1; // failed
		}
	}

	return 0; // success
}

void computeCooling(amrex::MultiFab &mf, const Real dt, cloudy_tables &cloudyTables)
{
	BL_PROFILE("RadhydroSimulation::computeCooling()")

	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4; // not recommended to change this

	auto tables = cloudyTables.const_tables();
	auto state = mf.arrays();

	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const Real rho = state[bx](i, j, k, HydroSystem<RandomBlast>::density_index);
		const Real x1Mom = state[bx](i, j, k, HydroSystem<RandomBlast>::x1Momentum_index);
		const Real x2Mom = state[bx](i, j, k, HydroSystem<RandomBlast>::x2Momentum_index);
		const Real x3Mom = state[bx](i, j, k, HydroSystem<RandomBlast>::x3Momentum_index);
		const Real Egas = state[bx](i, j, k, HydroSystem<RandomBlast>::energy_index);
		const Real Eint = RadSystem<RandomBlast>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);

		ODEUserData user_data{rho, tables};
		quokka::valarray<Real, 1> y = {Eint};
		quokka::valarray<Real, 1> abstol = {reltol_floor * ComputeEgasFromTgas(rho, T_floor, HydroSystem<RandomBlast>::gamma_, tables)};

		// do integration with RK2 (Heun's method)
		int nsteps = 0;
		rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol, nsteps);

		// check if integration failed
		if (nsteps >= maxStepsODEIntegrate) {
			Real T = ComputeTgasFromEgas(rho, Eint, HydroSystem<RandomBlast>::gamma_, tables);
			Real Edot = cloudy_cooling_function(rho, T, tables);
			Real t_cool = Eint / Edot;
			printf("max substeps exceeded! rho = %.17e, Eint = %.17e, T = %.17e, cooling "
			       "time = %.17e, dt = %.17e\n",
			       rho, Eint, T, t_cool, dt);
			amrex::Abort();
		}
		const Real Eint_new = y[0];
		const Real dEint = Eint_new - Eint;

		state[bx](i, j, k, HydroSystem<RandomBlast>::energy_index) += dEint;
		state[bx](i, j, k, HydroSystem<RandomBlast>::internalEnergy_index) += dEint;
	});
	amrex::Gpu::streamSynchronizeAll();
}

void injectEnergy(amrex::MultiFab &mf, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const Real dt)
{
	// inject energy into cells with stochastic sampling
	BL_PROFILE("RadhydroSimulation::injectEnergy()")

	const amrex::Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]); // cm^3
	const amrex::Real rho_eint_blast = E_blast / cell_vol;		  // ergs cm^-3
	const amrex::Real rho_ejecta = M_ejecta / cell_vol;		  // g cm^-3

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &px = blast_x.table();
		auto const &py = blast_y.table();
		auto const &pz = blast_z.table();
		const int np = ::nblast;

		const amrex::Real r_scale = 4.0 * dx[0];
    const amrex::Real normfac = 1.0 / std::pow(r_scale, 3);

		auto kern = [=] AMREX_GPU_DEVICE(const amrex::Real x, const amrex::Real y, const amrex::Real z) {
			const amrex::Real r = std::sqrt(x * x + y * y + z * z);
			return kernel_wendland_c2(r / r_scale);
		};

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const amrex::Real xc = prob_lo[0] + amrex::Real(i) * dx[0];
			const amrex::Real yc = prob_lo[1] + amrex::Real(j) * dx[1];
			const amrex::Real zc = prob_lo[2] + amrex::Real(k) * dx[2];

			for (int n = 0; n < np; ++n) {
				// integrate each particle kernel over the cell
				amrex::Real x0 = xc - px(n);
				amrex::Real y0 = yc - py(n);
				amrex::Real z0 = zc - pz(n);
				const amrex::Real weight = normfac * quad_3d(kern, x0, x0 + dx[0], y0, y0 + dx[1], z0, z0 + dx[2]);

				state(i, j, k, HydroSystem<RandomBlast>::density_index) += weight * rho_ejecta;
				state(i, j, k, HydroSystem<RandomBlast>::scalar0_index) += weight * rho_ejecta;
				state(i, j, k, HydroSystem<RandomBlast>::energy_index) += weight * rho_eint_blast;
				state(i, j, k, HydroSystem<RandomBlast>::internalEnergy_index) += weight * rho_eint_blast;
			}
		});
	}
}

template <> void RadhydroSimulation<RandomBlast>::computeBeforeTimestep()
{
	// compute how many (and where) SNe will go off on the this coarse timestep
	// sample from Poisson distribution
	amrex::Real dt_coarse = dt_[0];
	amrex::Real domain_vol = geom[0].ProbSize();
	const amrex::Real expectation_value = SN_rate_per_vol * domain_vol * dt_coarse;

	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));
	if (count > 0) {
		amrex::Print() << "\t" << count << " SNe to be exploded.\n";
	}

	// resize particle arrays
	blast_x.resize({0}, {count}, amrex::The_Pinned_Arena());
	blast_y.resize({0}, {count}, amrex::The_Pinned_Arena());
	blast_z.resize({0}, {count}, amrex::The_Pinned_Arena());
	::nblast = count;
	::SN_counter_cumulative += count;

	// for each, sample location at random
	auto const &px = blast_x.table();
	auto const &py = blast_y.table();
	auto const &pz = blast_z.table();
	for (int i = 0; i < count; ++i) {
		px(i) = geom[0].ProbLength(0) * amrex::Random();
		py(i) = geom[0].ProbLength(1) * amrex::Random();
		pz(i) = geom[0].ProbLength(2) * amrex::Random();
	}
}

template <> void RadhydroSimulation<RandomBlast>::computeAfterLevelAdvance(int lev, Real /*time*/, Real dt_lev, int /*ncycle*/)
{
	// compute operator split physics
	computeCooling(state_new_cc_[lev], dt_lev, cloudyTables);
	injectEnergy(state_new_cc_[lev], geom[lev].ProbLoArray(), geom[lev].CellSizeArray(), dt_lev);
}

template <> void RadhydroSimulation<RandomBlast>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		auto tables = cloudyTables.const_tables();

		for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &output = mf.array(iter);
			auto const &state = state_new_cc_[lev].const_array(iter);

			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				Real rho = state(i, j, k, HydroSystem<RandomBlast>::density_index);
				Real x1Mom = state(i, j, k, HydroSystem<RandomBlast>::x1Momentum_index);
				Real x2Mom = state(i, j, k, HydroSystem<RandomBlast>::x2Momentum_index);
				Real x3Mom = state(i, j, k, HydroSystem<RandomBlast>::x3Momentum_index);
				Real Egas = state(i, j, k, HydroSystem<RandomBlast>::energy_index);
				Real Eint = RadSystem<RandomBlast>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
				Real Tgas = ComputeTgasFromEgas(rho, Eint, HydroSystem<RandomBlast>::gamma_, tables);

				output(i, j, k, ncomp) = Tgas;
			});
		}
	}
}

template <> void RadhydroSimulation<RandomBlast>::ErrorEst(int lev, amrex::TagBoxArray &tags, Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement
	const Real q_min = 1e-5 * rho0; // minimum density for refinement
	const Real eta_threshold = ::refine_threshold;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<RandomBlast>::scalar0_index;

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
	// read parameters
	amrex::ParmParse pp;

	// read in SN rate
	pp.query("SN_rate_per_volume", ::SN_rate_per_vol); // yr^-1 kpc^-3
	::SN_rate_per_vol /= seconds_in_year;
	::SN_rate_per_vol /= std::pow(1.0e3 * parsec_in_cm, 3);
	AMREX_ALWAYS_ASSERT(!std::isnan(::SN_rate_per_vol));

	// read in refinement threshold (relative gradient in density)
	pp.query("refine_threshold", ::refine_threshold); // dimensionless

	// Problem initialization
	constexpr int nvars = RadhydroSimulation<RandomBlast>::nvarTotal_cc_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_cc[n].setLo(0, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(0, amrex::BCType::int_dir);
		}
	}

	RadhydroSimulation<RandomBlast> sim(BCs_cc);
	// sim.reconstructionOrder_ = 2; // PLM
	sim.densityFloor_ = 1.0e-5 * rho0; // density floor (to prevent vacuum)

	// Read Cloudy tables
	readCloudyData(sim.cloudyTables);

	// Set initial conditions
	sim.setInitialConditions();

	// set random state
	int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// run simulation
	sim.evolve();

	// print injected energy, injected mass
	const amrex::Real E_in_cumulative = static_cast<Real>(SN_counter_cumulative) * E_blast;
	const amrex::Real M_in_cumulative = static_cast<Real>(SN_counter_cumulative) * M_ejecta;
	amrex::Print() << "Cumulative injected energy = " << E_in_cumulative << "\n";
	amrex::Print() << "Cumulative injected mass = " << M_in_cumulative << "\n";

	// Cleanup and exit
	int status = 0;
	return status;
}
