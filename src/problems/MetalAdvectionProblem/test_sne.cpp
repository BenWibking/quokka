
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_sne.cpp
/// \brief Defines a problem for disk galaxy ISM.
///

#include <cmath>
#include <iostream>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_Random.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "test_sne.hpp"

// global variables needed for Dirichlet boundary condition and initial conditions

AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 100> logphi_data{
    5.23749982, 5.83925514, 6.19098487, 6.44028658, 6.63341552, 6.79097415, 6.92395454, 7.03892608, 7.1401333,	7.23047697, 7.31202697, 7.38631194, 7.45449324,
    7.5174744,	7.57597231, 7.63056519, 7.68172614, 7.72984716, 7.77525663, 7.81823237, 7.85901159, 7.89779849, 7.93477008, 7.9700808,	8.00386622, 8.03624594,
    8.06732603, 8.09720099, 8.12595536, 8.1536651,  8.18039866, 8.206218,   8.23117933, 8.25533386, 8.2787285,	8.30140621, 8.32340645, 8.34476546, 8.36551667,
    8.38569095, 8.40531688, 8.42442095, 8.44302778, 8.46116029, 8.47883984, 8.49608638, 8.51291857, 8.52935389, 8.54540873, 8.5610985,	8.57643767, 8.59143987,
    8.60611797, 8.62048409, 8.6345497,	8.64832568, 8.66182248, 8.6750501,  8.68801804, 8.70073526, 8.71321029, 8.72545119, 8.73746565, 8.74926096, 8.76084405,
    8.77222155, 8.78339974, 8.79438465, 8.80518201, 8.81579732, 8.8262358,  8.83650248, 8.84660217, 8.85653946, 8.86631876, 8.87594432, 8.88542019, 8.89475028,
    8.90393833, 8.91298796, 8.92190263, 8.93068568, 8.93934034, 8.94786969, 8.95627673, 8.96456434, 8.97273531, 8.98079244, 8.98873851, 8.99657621, 9.00430812,
    9.01193675, 9.01946449, 9.02689367, 9.03422652, 9.0414652,	9.04861178, 9.0556683,	9.06263668, 9.06951882};
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 100> logg_data{
    -9.85457856, -9.39107618, -9.19504351, -9.08451673, -9.01789453, -8.9773358,  -8.95292764, -8.93860797, -8.93051793, -8.92611324, -8.92379746, -8.92261559,
    -8.92202532, -8.92173815, -8.92160118, -8.92153673, -8.92150669, -8.9214927,  -8.92148604, -8.92148266, -8.92148075, -8.92147948, -8.92147848, -8.92147761,
    -8.9214768,	 -8.92147602, -8.92147525, -8.9214745,	-8.92147377, -8.92147304, -8.92147233, -8.92147163, -8.92147094, -8.92147026, -8.92146959, -8.92146894,
    -8.92146829, -8.92146765, -8.92146703, -8.92146641, -8.92146581, -8.92146521, -8.92146462, -8.92146405, -8.92146348, -8.92146293, -8.92146238, -8.92146185,
    -8.92146132, -8.9214608,  -8.92146029, -8.92145979, -8.9214593,  -8.92145882, -8.92145835, -8.92145789, -8.92145744, -8.92145699, -8.92145655, -8.92145612,
    -8.92145569, -8.92145528, -8.92145487, -8.92145447, -8.92145407, -8.92145369, -8.92145331, -8.92145294, -8.92145257, -8.92145221, -8.92145186, -8.92145152,
    -8.92145118, -8.92145085, -8.92145052, -8.9214502,	-8.92144989, -8.92144959, -8.92144929, -8.921449,   -8.92144871, -8.92144844, -8.92144816, -8.9214479,
    -8.92144764, -8.92144738, -8.92144713, -8.92144689, -8.92144665, -8.92144642, -8.92144619, -8.92144596, -8.92144574, -8.92144553, -8.92144531, -8.92144511,
    -8.9214449,	 -8.9214447,  -8.92144451, -8.92144431};
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, 100> z_data{
    6.08467742e+19, 1.82540323e+20, 3.04233871e+20, 4.25927419e+20, 5.47620968e+20, 6.69314516e+20, 7.91008065e+20, 9.12701613e+20, 1.03439516e+21,
    1.15608871e+21, 1.27778226e+21, 1.39947581e+21, 1.52116935e+21, 1.64286290e+21, 1.76455645e+21, 1.88625000e+21, 2.00794355e+21, 2.12963710e+21,
    2.25133065e+21, 2.37302419e+21, 2.49471774e+21, 2.61641129e+21, 2.73810484e+21, 2.85979839e+21, 2.98149194e+21, 3.10318548e+21, 3.22487903e+21,
    3.34657258e+21, 3.46826613e+21, 3.58995968e+21, 3.71165323e+21, 3.83334677e+21, 3.95504032e+21, 4.07673387e+21, 4.19842742e+21, 4.32012097e+21,
    4.44181452e+21, 4.56350806e+21, 4.68520161e+21, 4.80689516e+21, 4.92858871e+21, 5.05028226e+21, 5.17197581e+21, 5.29366935e+21, 5.41536290e+21,
    5.53705645e+21, 5.65875000e+21, 5.78044355e+21, 5.90213710e+21, 6.02383065e+21, 6.14552419e+21, 6.26721774e+21, 6.38891129e+21, 6.51060484e+21,
    6.63229839e+21, 6.75399194e+21, 6.87568548e+21, 6.99737903e+21, 7.11907258e+21, 7.24076613e+21, 7.36245968e+21, 7.48415323e+21, 7.60584677e+21,
    7.72754032e+21, 7.84923387e+21, 7.97092742e+21, 8.09262097e+21, 8.21431452e+21, 8.33600806e+21, 8.45770161e+21, 8.57939516e+21, 8.70108871e+21,
    8.82278226e+21, 8.94447581e+21, 9.06616935e+21, 9.18786290e+21, 9.30955645e+21, 9.43125000e+21, 9.55294355e+21, 9.67463710e+21, 9.79633065e+21,
    9.91802419e+21, 1.00397177e+22, 1.01614113e+22, 1.02831048e+22, 1.04047984e+22, 1.05264919e+22, 1.06481855e+22, 1.07698790e+22, 1.08915726e+22,
    1.10132661e+22, 1.11349597e+22, 1.12566532e+22, 1.13783468e+22, 1.15000403e+22, 1.16217339e+22, 1.17434274e+22, 1.18651210e+22, 1.19868145e+22,
    1.21085081e+22};

AMREX_GPU_MANAGED Real z_star = 245.0 * pc;
AMREX_GPU_MANAGED Real Sigma_star = 42.0 * Msun / pc / pc;
AMREX_GPU_MANAGED Real rho_dm = 0.0064 * Msun / pc / pc / pc;
AMREX_GPU_MANAGED Real R0 = 8.e3 * pc;
AMREX_GPU_MANAGED Real ks_sigma_sfr = 2.088579882548443e-55;
AMREX_GPU_MANAGED Real hscale = 150. * pc;
AMREX_GPU_MANAGED Real sigma1 = 700000.0;
AMREX_GPU_MANAGED Real sigma2 = 7000000.0;
AMREX_GPU_MANAGED Real rho01 = 2.78556e-24;
AMREX_GPU_MANAGED Real rho02 = 2.7855600000000006e-29;

struct NewProblem {
};

template <> struct HydroSystem_Traits<NewProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr bool reconstruct_eint = true; // Set to true - temperature
};

template <> struct quokka::EOS_Traits<NewProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<NewProblem> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;    // number of mass scalars
	static constexpr int numPassiveScalars = 1; // number of passive scalars
	static constexpr int nGroups = 1;	    // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct SimulationData<NewProblem> {

	std::unique_ptr<amrex::TableData<Real, 1>> blast_x;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_y;
	std::unique_ptr<amrex::TableData<Real, 1>> blast_z;

	int nblast = 0;
	int SN_counter_cumulative = 0;
	Real SN_rate_per_vol = NAN;  // rate per unit time per unit volume
	Real E_blast = 1.0e51;	     // ergs
	Real M_ejecta = 5.0 * Msun;  // 5.0 * Msun; // g
	Real refine_threshold = 1.0; // gradient refinement threshold
};

template <> void QuokkaSimulation<NewProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	double vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];

		// Calculate DM Potential
		double prefac;
		prefac = 2. * M_PI * Const_G * rho_dm * std::pow(R0, 2);
		double Phidm = (prefac * std::log(1. + std::pow(z / R0, 2)));

		// Calculate Stellar Disk Potential
		double prefac2;
		prefac2 = 2. * M_PI * Const_G * Sigma_star * z_star;
		double Phist = prefac2 * (std::pow(1. + z * z / z_star / z_star, 0.5) - 1.);

		// Calculate Gas Disk Potential

		double Phigas;
		// Interpolate to find the accurate g-value from array-- because linterp doesn't work on Setonix
		// TODO - AV to find out why linterp doesn't work
		size_t ii = 0;
		double x_interp = std::abs(z);
		while (ii < z_data.size() - 1 && x_interp > z_data[ii + 1]) {
			ii++;
		}

		// Perform linear interpolation
		const Real x1 = z_data[ii];
		const Real x2 = z_data[ii + 1];
		const Real y1 = logphi_data[ii];
		const Real y2 = logphi_data[ii + 1];
		amrex::Real phi_interp = (y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1));
		Phigas = std::pow(10., phi_interp);

		double Phitot = Phist + Phidm + Phigas;

		double rho, rho_disk, rho_halo;
		rho_disk = rho01 * std::exp(-Phitot / std::pow(sigma1, 2.0));
		rho_halo = rho02 * std::exp(-Phitot / std::pow(sigma2, 2.0)); // in g/cc
		rho = (rho_disk + rho_halo);

		double P = rho_disk * std::pow(sigma1, 2.0) + rho_halo * std::pow(sigma2, 2.0);

		AMREX_ASSERT(!std::isnan(rho));

		const auto gamma = HydroSystem<NewProblem>::gamma_;

		state_cc(i, j, k, HydroSystem<NewProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = P / (gamma - 1.);
		state_cc(i, j, k, HydroSystem<NewProblem>::energy_index) = P / (gamma - 1.);
		state_cc(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) = 1.e-5 / vol; // Injected tracer
	});
}

void AddSupernova(amrex::MultiFab &mf, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_lo, amrex::GpuArray<Real, AMREX_SPACEDIM> prob_hi,
		  amrex::GpuArray<Real, AMREX_SPACEDIM> dx, SimulationData<NewProblem> const &userData, int level)
{
	// TODO for AV - ave (and restore) the RNG state in the metadata.yaml file
	//  inject energy into cells with stochastic sampling
	BL_PROFILE("QuokkaSimulation::Addsupernova()")

	const Real cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]); // cm^3
	const Real rho_eint_blast = userData.E_blast / cell_vol;   // ergs cm^-3
	const Real rho_blast = userData.M_ejecta / cell_vol;	   // g cm^-3
	const Real scalar_blast = 1.e3 / cell_vol;		   // g cm^-3
	const int cum_sn = userData.SN_counter_cumulative;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &px = userData.blast_x->table();
		auto const &py = userData.blast_y->table();
		auto const &pz = userData.blast_z->table();
		const int np = userData.nblast;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real xc = prob_lo[0] + static_cast<Real>(i) * dx[0] + 0.5 * dx[0];
			const Real yc = prob_lo[1] + static_cast<Real>(j) * dx[1] + 0.5 * dx[1];
			const Real zc = prob_lo[2] + static_cast<Real>(k) * dx[2] + 0.5 * dx[2];

			for (int n = 0; n < np; ++n) {
				Real x0 = std::abs(xc - px(n));
				Real y0 = std::abs(yc - py(n));
				Real z0 = std::abs(zc - pz(n));

				if (x0 < 0.5 * dx[0] && y0 < 0.5 * dx[1] && z0 < 0.5 * dx[2]) {
					state(i, j, k, HydroSystem<NewProblem>::density_index) += rho_blast;
					state(i, j, k, HydroSystem<NewProblem>::energy_index) += rho_eint_blast;
					state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) += rho_eint_blast;
					state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex) += scalar_blast;

					printf("The total number of SN gone off=%d\n", cum_sn);
					Real Rpds = 14. * std::pow(state(i, j, k, HydroSystem<NewProblem>::density_index) / Const_mH, -3. / 7.);
					printf("Rpds = %.2e pc\n", Rpds);
				}
			}
		});
	}
}

template <> void QuokkaSimulation<NewProblem>::computeBeforeTimestep()
{
	// compute how many (and where) SNe will go off on the this coarse timestep
	// sample from Poisson distribution

	const Real dt_coarse = dt_[0];
	const Real domain_area = geom[0].ProbLength(0) * geom[0].ProbLength(1);
	const Real mean = 0.0;
	const Real stddev = hscale / geom[0].ProbLength(2) / 2.;

	const Real expectation_value = ks_sigma_sfr * domain_area * dt_coarse;

	const int count = static_cast<int>(amrex::RandomPoisson(expectation_value));

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
		pz(i) = geom[0].ProbLength(2) * amrex::RandomNormal(mean, stddev);
	}
}

template <> void QuokkaSimulation<NewProblem>::computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom[lev].ProbHiArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();

	AddSupernova(state_new_cc_[lev], prob_lo, prob_hi, dx, userData_, lev);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto HydroSystem<NewProblem>::GetGradFixedPotential(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec)
    -> amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
{

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> grad_potential;
	grad_potential[0] = 0.0;
	grad_potential[1] = 0.0;

	double z = posvec[2];

	// Interpolate to find the accurate g-value from array-- because linterp doesn't work on Setonix
	size_t i = 0;
	double x_interp = std::abs(z);
	while (i < z_data.size() - 1 && x_interp > z_data[i + 1]) {
		i++;
	}

	// Perform linear interpolation
	const Real x1 = z_data[i];
	const Real x2 = z_data[i + 1];
	const Real y1 = logg_data[i];
	const Real y2 = logg_data[i + 1];

	amrex::Real ginterp = (y1 + (y2 - y1) * (x_interp - x1) / (x2 - x1));

	grad_potential[2] = 2. * M_PI * Const_G * rho_dm * std::pow(R0, 2) * (2. * z / std::pow(R0, 2)) / (1. + std::pow(z, 2) / std::pow(R0, 2));
	grad_potential[2] += 2. * M_PI * Const_G * Sigma_star * (z / z_star) * (std::pow(1. + z * z / (z_star * z_star), -0.5));
	grad_potential[2] += (z / std::abs(z)) * std::pow(10., ginterp);

	return grad_potential;
}

// Add Strang Split Source Term for External Fixed Potential Here
template <> void QuokkaSimulation<NewProblem>::addStrangSplitSources(amrex::MultiFab &mf, int lev, amrex::Real time, amrex::Real dt_lev)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom[lev].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx = geom[lev].CellSizeArray();
	const Real dt = dt_lev;

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posvec, GradPhi;
			double x1mom_new, x2mom_new, x3mom_new;

			const Real rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
			const Real x1mom = state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index);
			const Real x2mom = state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index);
			const Real x3mom = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<NewProblem>::energy_index);

			Real Eint = RadSystem<NewProblem>::ComputeEintFromEgas(rho, x1mom, x2mom, x3mom, Egas);

			posvec[0] = prob_lo[0] + (i + 0.5) * dx[0];

#if (AMREX_SPACEDIM >= 2)
			posvec[1] = prob_lo[1] + (j + 0.5) * dx[1];
#endif

#if (AMREX_SPACEDIM >= 3)
			posvec[2] = prob_lo[2] + (k + 0.5) * dx[2];
#endif

			GradPhi = HydroSystem<NewProblem>::GetGradFixedPotential(posvec);

			x1mom_new = x1mom + dt * (-rho * GradPhi[0]);
			x2mom_new = x2mom + dt * (-rho * GradPhi[1]);
			x3mom_new = x3mom + dt * (-rho * GradPhi[2]);

			// State momentum values need to be updated this way.
			state(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1mom_new;
			state(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2mom_new;
			state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3mom_new;

			state(i, j, k, HydroSystem<NewProblem>::energy_index) =
			    RadSystem<NewProblem>::ComputeEgasFromEint(rho, x1mom_new, x2mom_new, x3mom_new, Eint);
		});
	}
}

// Code for producing in-situ Projection plots
template <> auto QuokkaSimulation<NewProblem>::ComputeProjections(const int dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	// compute density projection
	std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;

	proj["mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    // int nmscalars = Physics_Traits<NewProblem>::numMassScalars;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rho * vx3);
	    },
	    dir);

	proj["hot_mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    flux = rho * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["warm_mass_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    flux = rho * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vz = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    return (rhoZ * vz);
	    },
	    dir);

	proj["warm_scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp < 2.e4) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["hot_scalar_outflow"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    double flux;
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    Real const vx3 = state(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) / rho;
		    Real const Eint = state(i, j, k, HydroSystem<NewProblem>::internalEnergy_index);
		    amrex::GpuArray<Real, 0> massScalars = RadSystem<NewProblem>::ComputeMassScalars(state, i, j, k);
		    Real const primTemp = quokka::EOS<NewProblem>::ComputeTgasFromEint(rho, Eint, massScalars);
		    if (primTemp > 1.e6) {
			    flux = rhoZ * vx3;
		    } else {
			    flux = 0.0;
		    }
		    return flux;
	    },
	    dir);

	proj["rho"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rho = state(i, j, k, HydroSystem<NewProblem>::density_index);
		    return (rho);
	    },
	    dir);

	proj["scalar"] = computePlaneProjection<amrex::ReduceOpSum>(
	    [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
		    Real const rhoZ = state(i, j, k, Physics_Indices<NewProblem>::pscalarFirstIndex);
		    return (rhoZ);
	    },
	    dir);
	return proj;
}

// Implement User-defined diode BC
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<NewProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
												int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
												const Real /*time*/, const amrex::BCRec * /*bcr*/,
												int /*bcomp*/, int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int klo = domain_lo[2];
	const int khi = domain_hi[2];
	int kedge = 0;
	int normal = 0;

	if (k < klo) {
		kedge = klo;
		normal = -1;
	} else if (k > khi) {
		kedge = khi;
		normal = 1.0;
	}

	const double rho_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::density_index);
	const double x1Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x1Momentum_index);
	const double x2Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x2Momentum_index);
	double x3Mom_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
	const double etot_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::energy_index);
	const double eint_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::internalEnergy_index);
	const double pscalar_edge = consVar(i, j, kedge, HydroSystem<NewProblem>::scalar0_index);

	if ((x3Mom_edge * normal) < 0) { // gas is inflowing
		x3Mom_edge = -1. * consVar(i, j, kedge, HydroSystem<NewProblem>::x3Momentum_index);
	}

	consVar(i, j, k, HydroSystem<NewProblem>::density_index) = rho_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x1Momentum_index) = x1Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x2Momentum_index) = x2Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::x3Momentum_index) = x3Mom_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::energy_index) = etot_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::internalEnergy_index) = eint_edge;
	consVar(i, j, k, HydroSystem<NewProblem>::scalar0_index) = pscalar_edge;
}

auto problem_main() -> int
{

	const int ncomp_cc = Physics_Indices<NewProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);

	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// diode boundary conditions
			if (i == 2) {
				BCs_cc[n].setLo(i, amrex::BCType::ext_dir);
				BCs_cc[n].setHi(i, amrex::BCType::ext_dir);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
				BCs_cc[n].setHi(i, amrex::BCType::int_dir); // periodic
			}
		}
	}

	// set random state
	const int seed = 42;
	amrex::InitRandom(seed, 1); // all ranks should produce the same values

	// Problem initialization
	QuokkaSimulation<NewProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.cflNumber_ = 0.3;	      // *must* be less than 1/3 in 3D!

	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
