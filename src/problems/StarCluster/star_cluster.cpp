//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file star_cluster.cpp
/// \brief Defines a test problem for pressureless spherical collapse of a star cluster.
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

#include "QuokkaSimulation.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "star_cluster.hpp"
#include "turbulence/TurbDataReader.hpp"

using amrex::Real;

struct StarCluster {
};

template <> struct quokka::EOS_Traits<StarCluster> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double boltzmann_constant = C::k_B;
};

template <> struct HydroSystem_Traits<StarCluster> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<StarCluster> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr amrex::Real gravitational_constant = 1.0;
};

template <> struct SimulationData<StarCluster> {
	// real-space perturbation fields
	amrex::TableData<Real, 3> dvx;
	amrex::TableData<Real, 3> dvy;
	amrex::TableData<Real, 3> dvz;
	Real dv_rms_generated{};
	Real dv_rms_target{};
	Real rescale_factor{};

	// cloud parameters
	Real R_sphere{};
	Real rho_sphere{};
	Real alpha_vir{};
};

template <> void QuokkaSimulation<StarCluster>::preCalculateInitialConditions()
{
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

		const Real R_sphere = userData_.R_sphere;
		const Real rho_sph = userData_.rho_sphere;
		const Real alpha_vir = userData_.alpha_vir;
		const Real M_sphere = (4. / 3.) * M_PI * std::pow(R_sphere, 3) * rho_sph;
		const Real rms_dv_target = std::sqrt(alpha_vir * (3. / 5.) * Gconst_ * M_sphere / R_sphere);
		const Real rms_Mach_target = rms_dv_target / quokka::EOS_Traits<StarCluster>::cs_isothermal;
		const Real rms_dv_actual = userData_.dv_rms_generated;
		userData_.rescale_factor = rms_dv_target / rms_dv_actual;
		amrex::Print() << "rms Mach target = " << rms_Mach_target << "\n";

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

template <> void QuokkaSimulation<StarCluster>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// cloud parameters
	const double rho_min = 0.01 * userData_.rho_sphere;
	const double rho_max = userData_.rho_sphere;
	const double R_sphere = userData_.R_sphere;
	const double R_smooth = 0.05 * R_sphere;
	const double renorm_amp = userData_.rescale_factor;

	auto const &dvx = userData_.dvx.const_table();
	auto const &dvy = userData_.dvy.const_table();
	auto const &dvz = userData_.dvz.const_table();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

		double const rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		AMREX_ASSERT(!std::isnan(rho));

		double vx = dvx(i, j, k);
		double vy = dvy(i, j, k);
		double vz = dvz(i, j, k);

		vx *= renorm_amp * (rho / rho_max);
		vy *= renorm_amp * (rho / rho_max);
		vz *= renorm_amp * (rho / rho_max);

		state_cc(i, j, k, HydroSystem<StarCluster>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<StarCluster>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<StarCluster>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<StarCluster>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<StarCluster>::energy_index) = 0;
		state_cc(i, j, k, HydroSystem<StarCluster>::internalEnergy_index) = 0;
	});
}

template <> void QuokkaSimulation<StarCluster>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// refine on Jeans length
	const int N_cells = 4; // inverse of the 'Jeans number' [Truelove et al. (1997)]
	const amrex::Real cs = quokka::EOS_Traits<StarCluster>::cs_isothermal;
	const amrex::Real dx = geom[lev].CellSizeArray()[0];
	const amrex::Real G = Gconst_;

	auto const &state = state_new_cc_[lev].const_arrays();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		Real const rho = state[bx](i, j, k, HydroSystem<StarCluster>::density_index);
		const amrex::Real l_Jeans = cs * std::sqrt(M_PI / (G * rho));

		if (l_Jeans < (N_cells * dx)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

template <> void QuokkaSimulation<StarCluster>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in) const
{
	// compute derived variables and save in 'mf'
	if (dname == "log_density") {
		const int ncomp = ncomp_cc_in;
		auto const &state = state_new_cc_[lev].const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<StarCluster>::density_index);
			output[bx](i, j, k, ncomp) = std::log10(rho);
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
	Real rho_sphere{};
	pp.query("cloud_density", rho_sphere);

	// cloud virial parameter
	Real alpha_vir{};
	pp.query("virial_parameter", alpha_vir);

	// boundary conditions
	const int ncomp_cc = Physics_Indices<StarCluster>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	QuokkaSimulation<StarCluster> sim(BCs_cc);
	sim.doPoissonSolve_ = 1; // enable self-gravity
	sim.densityFloor_ = 0.01;

	sim.userData_.R_sphere = R_sphere;
	sim.userData_.rho_sphere = rho_sphere;
	sim.userData_.alpha_vir = alpha_vir;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int const status = 0;
	return status;
}
