//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radhydro_shell.cpp
/// \brief Defines a test problem for a 3D radiation pressure-driven shell.
///

#include <limits>

#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_Loop.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "radiation/radiation_system.hpp"
#include "test_radhydro_shell.hpp"

struct ShellProblem {
};
// if false, use octant symmetry
constexpr bool simulate_full_box = true;

constexpr double a_rad = C::a_rad;
constexpr double c = C::c_light;
constexpr double a0 = 2.0e5;	   // ('reference' sound speed) [cm s^-1]
constexpr double chat = 860. * a0; // cm s^-1
constexpr double k_B = C::k_B;	   // erg K^-1
constexpr double m_H = C::m_u;	   // atomic mass unit
constexpr double gamma_gas = 5. / 3.;

template <> struct quokka::EOS_Traits<ShellProblem> {
	static constexpr double mean_molecular_weight = 2.2 * m_H;
	static constexpr double gamma = gamma_gas;
};

template <> struct RadSystem_Traits<ShellProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = 0.;
	static constexpr int beta_order = 1;
};

template <> struct HydroSystem_Traits<ShellProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<ShellProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr amrex::Real Msun = 2.0e33;	       // g
constexpr amrex::Real parsec_in_cm = 3.086e18; // cm

constexpr amrex::Real specific_luminosity = 2000.;			   // erg s^-1 g^-1
constexpr amrex::Real GMC_mass = 1.0e6 * Msun;				   // g
constexpr amrex::Real epsilon = 0.5;					   // dimensionless
constexpr amrex::Real M_shell = (1 - epsilon) * GMC_mass;		   // g
constexpr amrex::Real L_star = (epsilon * GMC_mass) * specific_luminosity; // erg s^-1

constexpr amrex::Real r_0 = 5.0 * parsec_in_cm; // cm
constexpr amrex::Real sigma_star = 0.3 * r_0;	// cm
constexpr amrex::Real H_shell = 0.3 * r_0;	// cm
constexpr amrex::Real kappa0 = 20.0;		// specific opacity [cm^2 g^-1]

constexpr amrex::Real rho_0 = M_shell / ((4. / 3.) * M_PI * r_0 * r_0 * r_0); // g cm^-3

constexpr amrex::Real P_0 = gamma_gas * rho_0 * (a0 * a0); // erg cm^-3
constexpr double c_v = k_B / ((2.2 * m_H) * (gamma_gas - 1.0));

template <>
void RadSystem<ShellProblem>::SetRadEnergySource(array_t &radEnergy, const amrex::Box &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
						 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real /*time*/)
{
	// point-like radiation source

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

	const amrex::Real source_norm = (1.0 / c) * L_star / std::pow(2.0 * M_PI * sigma_star * sigma_star, 1.5);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

		radEnergy(i, j, k) = source_norm * std::exp(-(r * r) / (2.0 * sigma_star * sigma_star));
	});
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShellProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
	;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShellProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return ComputePlanckOpacity(0.0, 0.0);
}

// declare global variables
// initial conditions read from file
amrex::Gpu::HostVector<double> r_arr;
amrex::Gpu::HostVector<double> Erad_arr;
amrex::Gpu::HostVector<double> Frad_arr;

amrex::Gpu::DeviceVector<double> r_arr_g;
amrex::Gpu::DeviceVector<double> Erad_arr_g;
amrex::Gpu::DeviceVector<double> Frad_arr_g;

template <> void QuokkaSimulation<ShellProblem>::preCalculateInitialConditions()
{
	std::string filename = "./initial_conditions.txt";
	std::ifstream fstream(filename, std::ios::in);
	AMREX_ALWAYS_ASSERT(fstream.is_open());
	std::string header;
	std::getline(fstream, header);

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;
		for (double value = NAN; iss >> value;) {
			values.push_back(value);
		}
		auto r = values.at(0) * r_0; // cm
		auto Erad = values.at(2);    // cgs
		auto Frad = values.at(3);    // cgs

		r_arr.push_back(r);
		Erad_arr.push_back(Erad);
		Frad_arr.push_back(Frad);
	}

	// copy to device
	r_arr_g.resize(r_arr.size());
	Erad_arr_g.resize(Erad_arr.size());
	Frad_arr_g.resize(Frad_arr.size());

	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r_arr.begin(), r_arr.end(), r_arr_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Erad_arr.begin(), Erad_arr.end(), Erad_arr_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Frad_arr.begin(), Frad_arr.end(), Frad_arr_g.begin());
	amrex::Gpu::streamSynchronizeAll();
}

template <> void QuokkaSimulation<ShellProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

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

	auto const &r_ptr = r_arr_g.dataPtr();
	auto const &Erad_ptr = Erad_arr_g.dataPtr();
	auto const &Frad_ptr = Frad_arr_g.dataPtr();
	int r_size = static_cast<int>(r_arr_g.size());

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
		amrex::Real const rhat_x = (x - x0) / r;
		amrex::Real const rhat_y = (y - y0) / r;
		amrex::Real const rhat_z = (z - z0) / r;

		double sigma_sh = H_shell / (2.0 * std::sqrt(2.0 * std::log(2.0)));
		double rho_norm = M_shell / (4.0 * M_PI * r * r * std::sqrt(2.0 * M_PI * sigma_sh * sigma_sh));
		double rho_shell = rho_norm * std::exp(-std::pow(r - r_0, 2) / (2.0 * sigma_sh * sigma_sh));
		double rho = std::max(rho_shell, 1.0e-8 * rho_0);

		// interpolate Frad from table
		const double Frad_r = interpolate_value(r, r_ptr, Frad_ptr, r_size);

		// interpolate Erad from table
		const double Erad = interpolate_value(r, r_ptr, Erad_ptr, r_size);

		const double Trad = std::pow(Erad / a_rad, 1. / 4.);
		const double Tgas = Trad;
		const double Eint = rho * c_v * Tgas;

		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(Erad));
		AMREX_ASSERT(!std::isnan(Frad));

		state_cc(i, j, k, HydroSystem<ShellProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ShellProblem>::x1Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ShellProblem>::x2Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ShellProblem>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<ShellProblem>::energy_index) = Eint;

		const double Frad_x = Frad_r * rhat_x;
		const double Frad_y = Frad_r * rhat_y;
		const double Frad_z = Frad_r * rhat_z;

		state_cc(i, j, k, RadSystem<ShellProblem>::gasInternalEnergy_index) = Eint;
		state_cc(i, j, k, RadSystem<ShellProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<ShellProblem>::x1RadFlux_index) = Frad_x;
		state_cc(i, j, k, RadSystem<ShellProblem>::x2RadFlux_index) = Frad_y;
		state_cc(i, j, k, RadSystem<ShellProblem>::x3RadFlux_index) = Frad_z;
	});
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto vec_dot_r(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec, int i, int j, int k) -> amrex::Real
{
	// compute dot product of vec into rhat
	amrex::Real xhat = (i + amrex::Real(0.5));
	amrex::Real yhat = (j + amrex::Real(0.5));
	amrex::Real zhat = (k + amrex::Real(0.5));
	amrex::Real const norminv = 1.0 / std::sqrt(xhat * xhat + yhat * yhat + zhat * zhat);

	xhat *= norminv;
	yhat *= norminv;
	zhat *= norminv;

	amrex::Real const dotproduct = vec[0] * xhat + vec[1] * yhat + vec[2] * zhat;
	return dotproduct;
}

#if 0
template <> void QuokkaSimulation<ShellProblem>::computeAfterTimestep() {
  // compute radial momentum for gas, radiation on level 0
  // (assuming octant symmetry)

  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 =
      geom[0].CellSizeArray();
  amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
  auto const &state = state_new_cc_[0];

  double radialMom =
      vol *
      amrex::ReduceSum(
          state, 0,
          [=] AMREX_GPU_DEVICE(amrex::Box const &bx,
                               amrex::Array4<amrex::Real const> const &arr) {
            amrex::Real result = 0.;
            amrex::Loop(bx, [&](int i, int j, int k) {
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec{
                  arr(i, j, k, RadSystem<ShellProblem>::x1GasMomentum_index),
                  arr(i, j, k, RadSystem<ShellProblem>::x2GasMomentum_index),
                  arr(i, j, k, RadSystem<ShellProblem>::x3GasMomentum_index)};
              result += vec_dot_r(vec, i, j, k);
            });
            return result;
          });

  amrex::ParallelAllReduce::Sum(radialMom,
                                amrex::ParallelContext::CommunicatorSub());

  double radialRadMom =
      (vol / c) *
      amrex::ReduceSum(
          state, 0,
          [=] AMREX_GPU_DEVICE(amrex::Box const &bx,
                               amrex::Array4<amrex::Real const> const &arr) {
            amrex::Real result = 0.;
            amrex::Loop(bx, [&](int i, int j, int k) {
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vec{
                  arr(i, j, k, RadSystem<ShellProblem>::x1RadFlux_index),
                  arr(i, j, k, RadSystem<ShellProblem>::x2RadFlux_index),
                  arr(i, j, k, RadSystem<ShellProblem>::x3RadFlux_index)};
              result += vec_dot_r(vec, i, j, k);
            });
            return result;
          });

  amrex::ParallelAllReduce::Sum(radialRadMom,
                                amrex::ParallelContext::CommunicatorSub());

  amrex::Print() << "radial gas momentum = " << radialMom << std::endl;
  amrex::Print() << "radial radiation momentum = " << radialRadMom << std::endl;
}
#endif

template <> void QuokkaSimulation<ShellProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.1;	    // gradient refinement threshold
	const amrex::Real rho_min = 1.0e-2 * rho_0; // minimum density for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const int n = HydroSystem<ShellProblem>::density_index;
			amrex::Real const rho = state(i, j, k, n);

			amrex::Real const rho_xplus = state(i + 1, j, k, n);
			amrex::Real const rho_xminus = state(i - 1, j, k, n);

			amrex::Real const rho_yplus = state(i, j + 1, k, n);
			amrex::Real const rho_yminus = state(i, j - 1, k, n);

			amrex::Real const rho_zplus = state(i, j, k + 1, n);
			amrex::Real const rho_zminus = state(i, j, k - 1, n);

			amrex::Real const del_x = std::max(std::abs(rho_xplus - rho), std::abs(rho - rho_xminus));
			amrex::Real const del_y = std::max(std::abs(rho_yplus - rho), std::abs(rho - rho_yminus));
			amrex::Real const del_z = std::max(std::abs(rho_zplus - rho), std::abs(rho - rho_zminus));

			amrex::Real const gradient_indicator = std::max({del_x, del_y, del_z}) / rho;

			if ((gradient_indicator > eta_threshold) && (rho >= rho_min)) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// This problem can only be run in 3D
	static_assert(AMREX_SPACEDIM == 3);

	auto isNormalComp = [=](int n, int dim) {
		// it is critical to reflect both the radiation and gas momenta!
		if ((n == RadSystem<ShellProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ShellProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	const int ncomp_cc = Physics_Indices<ShellProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			if constexpr (simulate_full_box) {
				// periodic boundaries
				BCs_cc[n].setLo(i, amrex::BCType::int_dir);
				BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			} else {
				// reflecting boundaries, outflow boundaries
				if (isNormalComp(n, i)) {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
					BCs_cc[n].setHi(i, amrex::BCType::foextrap); // outflow
				} else {
					BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
					BCs_cc[n].setHi(i, amrex::BCType::foextrap); // outflow
				}
			}
		}
	}

	// Problem initialization
	QuokkaSimulation<ShellProblem> sim(BCs_cc);

	sim.cflNumber_ = 0.3;
	sim.densityFloor_ = 1.0e-8 * rho_0;
	sim.pressureFloor_ = 1.0e-8 * P_0;
	// reconstructionOrder: 1 == donor cell, 2 == PLM, 3 == PPM (not recommended
	// for this problem)
	sim.reconstructionOrder_ = 2;
	sim.radiationReconstructionOrder_ = 2;
	sim.integratorOrder_ = 2; // RK2

	constexpr amrex::Real t0_hydro = r_0 / a0; // seconds
	sim.stopTime_ = 0.125 * t0_hydro;	   // 0.124 * t0_hydro;

	// for production
	// sim.checkpointInterval_ = 1000;
	// sim.plotfileInterval_ = 100;
	// sim.maxTimesteps_ = 5000;

	// for scaling tests
	sim.checkpointInterval_ = -1;
	sim.plotfileInterval_ = -1;
	sim.maxTimesteps_ = 50;

	// initialize
	sim.setInitialConditions();
	sim.computeAfterTimestep();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return 0;
}
