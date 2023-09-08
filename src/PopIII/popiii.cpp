//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file popiii.cpp
/// \brief Defines a test problem for Pop III star formation.
/// Author: Piyush Sharda (Leiden University, 2023)
///
#include <limits>
#include <memory>
#include <random>

#include "AMReX.H"
#include "AMReX_Arena.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "RadhydroSimulation.hpp"
#include "SimulationData.hpp"
#include "TurbDataReader.hpp"
#include "hydro_system.hpp"
#include "popiii.hpp"

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

using amrex::Real;

struct PopIII {
};

template <> struct HydroSystem_Traits<PopIII> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<PopIII> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_chemistry_enabled = false;	     // in the future, this could point to microphysics, and set to true
	static constexpr int numMassScalars = NumSpec;		     // number of chemical species
	static constexpr int numPassiveScalars = numMassScalars + 0; // we only have mass scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
};


template <> struct SimulationData<PopIII> {
	// real-space perturbation fields
	amrex::TableData<Real, 3> dvx;
	amrex::TableData<Real, 3> dvy;
	amrex::TableData<Real, 3> dvz;
	amrex::Real dv_rms_generated{};
	amrex::Real dv_rms_target{};
	amrex::Real rescale_factor{};

	// cloud parameters
	amrex::Real R_sphere{};
	amrex::Real numdens_init{};
	amrex::Real omega_sphere{};

	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{};
	amrex::Real primary_species_1{};
	amrex::Real primary_species_2{};
	amrex::Real primary_species_3{};
	amrex::Real primary_species_4{};
	amrex::Real primary_species_5{};
	amrex::Real primary_species_6{};
	amrex::Real primary_species_7{};
	amrex::Real primary_species_8{};
	amrex::Real primary_species_9{};
	amrex::Real primary_species_10{};
	amrex::Real primary_species_11{};
	amrex::Real primary_species_12{};
	amrex::Real primary_species_13{};
	amrex::Real primary_species_14{};


};

template <> void RadhydroSimulation<PopIII>::preCalculateInitialConditions()
{

	// initialize microphysics routines
	init_extern_parameters();

	// parmparse species and temperature
	amrex::ParmParse pp("primordial_chem");
	userData_.small_temp = 1e1;
	pp.query("small_temp", userData_.small_temp);

	userData_.small_dens = 1e-60;
	pp.query("small_dens", userData_.small_dens);

	userData_.temperature = 1e1;
	pp.query("temperature", userData_.temperature);

	userData_.primary_species_1 = 1.0e0_rt;
	userData_.primary_species_2 = 0.0e0_rt;
	userData_.primary_species_3 = 0.0e0_rt;
	userData_.primary_species_4 = 0.0e0_rt;
	userData_.primary_species_5 = 0.0e0_rt;
	userData_.primary_species_6 = 0.0e0_rt;
	userData_.primary_species_7 = 0.0e0_rt;
	userData_.primary_species_8 = 0.0e0_rt;
	userData_.primary_species_9 = 0.0e0_rt;
	userData_.primary_species_10 = 0.0e0_rt;
	userData_.primary_species_11 = 0.0e0_rt;
	userData_.primary_species_12 = 0.0e0_rt;
	userData_.primary_species_13 = 0.0e0_rt;
	userData_.primary_species_14 = 0.0e0_rt;

	pp.query("primary_species_1", userData_.primary_species_1);
	pp.query("primary_species_2", userData_.primary_species_2);
	pp.query("primary_species_3", userData_.primary_species_3);
	pp.query("primary_species_4", userData_.primary_species_4);
	pp.query("primary_species_5", userData_.primary_species_5);
	pp.query("primary_species_6", userData_.primary_species_6);
	pp.query("primary_species_7", userData_.primary_species_7);
	pp.query("primary_species_8", userData_.primary_species_8);
	pp.query("primary_species_9", userData_.primary_species_9);
	pp.query("primary_species_10", userData_.primary_species_10);
	pp.query("primary_species_11", userData_.primary_species_11);
	pp.query("primary_species_12", userData_.primary_species_12);
	pp.query("primary_species_13", userData_.primary_species_13);
	pp.query("primary_species_14", userData_.primary_species_14);

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();


	static bool isSamplingDone = false;
	if (!isSamplingDone) {
		// read perturbations from file
		turb_data turbData;
		amrex::ParmParse pp("perturb");
		std::string turbdata_filename;
		pp.query("filename", turbdata_filename);
		initialize_turbdata(turbData, turbdata_filename);

		// copy to pinned memory
		auto pinned_dvx = get_tabledata(turbData.dvx);
		auto pinned_dvy = get_tabledata(turbData.dvy);
		auto pinned_dvz = get_tabledata(turbData.dvz);

		// compute normalisation
		userData_.dv_rms_generated = computeRms(pinned_dvx, pinned_dvy, pinned_dvz);
		amrex::Print() << "rms dv = " << userData_.dv_rms_generated << "\n";

		const Real rms_dv_target = 1.8050e5;
		const Real rms_dv_actual = userData_.dv_rms_generated;
		userData_.rescale_factor = rms_dv_target / rms_dv_actual;

		// copy to GPU
		userData_.dvx.resize(pinned_dvx.lo(), pinned_dvx.hi());
		userData_.dvx.copy(pinned_dvx);

		userData_.dvy.resize(pinned_dvy.lo(), pinned_dvy.hi());
		userData_.dvy.copy(pinned_dvy);

		userData_.dvz.resize(pinned_dvz.lo(), pinned_dvz.hi());
		userData_.dvz.copy(pinned_dvz);

		isSamplingDone = true;
	}
}

template <> void RadhydroSimulation<PopIII>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;


	std::array<Real, NumSpec> numdens = {-1.0};

	for (int n = 1; n <= NumSpec; ++n) {
		switch (n) {

			case 1:
				numdens[n - 1] = userData_.primary_species_1;
				break;
			case 2:
				numdens[n - 1] = userData_.primary_species_2;
				break;
			case 3:
				numdens[n - 1] = userData_.primary_species_3;
				break;
			case 4:
				numdens[n - 1] = userData_.primary_species_4;
				break;
			case 5:
				numdens[n - 1] = userData_.primary_species_5;
				break;
			case 6:
				numdens[n - 1] = userData_.primary_species_6;
				break;
			case 7:
				numdens[n - 1] = userData_.primary_species_7;
				break;
			case 8:
				numdens[n - 1] = userData_.primary_species_8;
				break;
			case 9:
				numdens[n - 1] = userData_.primary_species_9;
				break;
			case 10:
				numdens[n - 1] = userData_.primary_species_10;
				break;
			case 11:
				numdens[n - 1] = userData_.primary_species_11;
				break;
			case 12:
				numdens[n - 1] = userData_.primary_species_12;
				break;
			case 13:
				numdens[n - 1] = userData_.primary_species_13;
				break;
			case 14:
				numdens[n - 1] = userData_.primary_species_14;
				break;

			default:
				amrex::Abort("Cannot initialize number density for chem specie");
				break;
		}
	}


	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// cloud parameters
	const double R_sphere = userData_.R_sphere;
	const double omega_sphere = userData_.omega_sphere;
	const double renorm_amp = userData_.rescale_factor;
	const double numdens_init = userData_.numdens_init;
	const double core_temp = userData_.temperature;

	auto const &dvx = userData_.dvx.const_table();
	auto const &dvy = userData_.dvy.const_table();
	auto const &dvz = userData_.dvz.const_table();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
		amrex::Real const distxy = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));

		burn_t state;
		amrex::Real rhotot = 0.0;

		for (int n = 0; n < NumSpec; ++n) {
			state.xn[n] = numdens[n] * numdens_init;
			rhotot += state.xn[n] * spmasses[n]; // spmasses contains the masses of all species, defined in EOS
		}

		// normalize -- just in case
		std::array<Real, NumSpec> mfracs = {-1.0};
		Real msum = 0.0;
		for (int n = 0; n < NumSpec; ++n) {
			mfracs[n] = state.xn[n] * spmasses[n] / rhotot;
			msum += mfracs[n];
		}

		for (int n = 0; n < NumSpec; ++n) {
			mfracs[n] /= msum;
			// use the normalized mass fractions to obtain the corresponding number densities
			state.xn[n] = mfracs[n] * rhotot / spmasses[n];
		}

		//amrex::Print() << "cell " << i << j << k << " " << rhotot << " " << numdens_init << " " << numdens[0] << std::endl;

		amrex::Real phi = atan2((y-y0), (x-x0));

		double vx = 0;
		double vy = 0;
		double vz = 0;

		if (r <= R_sphere) {
			state.rho = rhotot;
			state.T = core_temp;
			vx = renorm_amp * dvx(i, j, k);
			vy = renorm_amp * dvy(i, j, k);
			vz = renorm_amp * dvz(i, j, k);

			// add rotation to vx and vy
			vx += (-1.0) * distxy * omega_sphere * std::sin(phi);
			vy += distxy * omega_sphere * std::cos(phi);

		} else {
			state.rho = 0.01 * rhotot;
			state.T = 1e2 * core_temp; // can use EOS.hpp function to solve for temp
		}

		// call the EOS to set initial internal energy e
		eos(eos_input_rt, state);
		amrex::Real e = state.rho * state.e;

		//amrex::Print() << "cell " << i << j << k << " " << state.rho << " " << state.T << " " << e << std::endl;

		state_cc(i, j, k, HydroSystem<PopIII>::density_index) = state.rho;
		state_cc(i, j, k, HydroSystem<PopIII>::x1Momentum_index) = state.rho * vx;
		state_cc(i, j, k, HydroSystem<PopIII>::x2Momentum_index) = state.rho * vy;
		state_cc(i, j, k, HydroSystem<PopIII>::x3Momentum_index) = state.rho * vz;
		state_cc(i, j, k, HydroSystem<PopIII>::internalEnergy_index) = e;

		Real const Egas = RadSystem<PopIII>::ComputeEgasFromEint(state.rho, state.rho*vx, state.rho*vy, state.rho*vz, e);
		state_cc(i, j, k, HydroSystem<PopIII>::energy_index) = Egas;

		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<PopIII>::scalar0_index + nn) =
			    mfracs[nn] * state.rho; // we use partial densities and not mass fractions
		}

	});
}

template <> void RadhydroSimulation<PopIII>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// refine on Jeans length
	const int N_cells = 32; // inverse of the 'Jeans number' [Truelove et al. (1997)]
	const amrex::Real dx = geom[lev].CellSizeArray()[0];
	const amrex::Real G = Gconst_;

	auto const &state = state_new_cc_[lev].const_arrays();
	auto tag = tags.arrays();
	amrex::Print() << "start error est " << std::endl;
	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {

		Real const rho = state[bx](i, j, k, HydroSystem<PopIII>::density_index);
		Real pressure = HydroSystem<PopIII>::ComputePressure(state[bx], i, j, k);
		amrex::GpuArray<Real, Physics_Traits<PopIII>::numMassScalars> massScalars = RadSystem<PopIII>::ComputeMassScalars(state[bx], i, j, k);

		amrex::Real cs = quokka::EOS<PopIII>::ComputeSoundSpeed(rho, pressure, massScalars);

		const amrex::Real l_Jeans = cs * std::sqrt(M_PI / (G * rho));

		if (l_Jeans < (N_cells * dx)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
			printf("tagging cell for refinement \n");
		}
	});
	amrex::Print() << "end error est " << std::endl;
}

template <> void RadhydroSimulation<PopIII>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "temperature") {
		const int ncomp = ncomp_cc_in;
		auto const &state = state_new_cc_[lev].const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			
			Real const rho = state[bx](i, j, k, HydroSystem<PopIII>::density_index);
			amrex::Real Eint = state[bx](i, j, k, HydroSystem<PopIII>::internalEnergy_index);

			amrex::GpuArray<Real, Physics_Traits<PopIII>::numMassScalars> massScalars = RadSystem<PopIII>::ComputeMassScalars(state[bx], i, j, k);

			output[bx](i, j, k, ncomp) = quokka::EOS<PopIII>::ComputeTgasFromEint(rho, Eint, massScalars);
		});
	}
}

auto problem_main() -> int
{
	// read problem parameters
	amrex::ParmParse const pp("perturb");

	// cloud radius
	Real R_sphere{};
	pp.query("cloud_radius", R_sphere);

	// cloud density
	Real numdens_init{};
	pp.query("cloud_numdens", numdens_init);

	// cloud angular velocity
	Real omega_sphere{};
	pp.query("cloud_omega", omega_sphere);


	// boundary conditions
	const int ncomp_cc = Physics_Indices<PopIII>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	RadhydroSimulation<PopIII> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	sim.densityFloor_ = 1e-60;

	sim.userData_.R_sphere = R_sphere;
	sim.userData_.numdens_init = numdens_init;
	sim.userData_.omega_sphere = omega_sphere;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int const status = 0;
	return status;
}
