//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_shadow.cpp
/// \brief Defines a 2D test problem for radiation in the transport regime.
///

#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "simulation.hpp"

struct ShadowProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double sigma0 = 0.1;	     // cm^-1 (opacity)
constexpr double rho_bg = 1.0e-3;    // g cm^-3 (matter density)
constexpr double rho_clump = 1.0;    // g cm^-3 (matter density)
constexpr double T_hohlraum = 1740.; // K
constexpr double T_initial = 290.;   // K

constexpr double a_rad = C::a_rad; // erg cm^-3 K^-4
constexpr double c = C::c_light;   // cm s^-1

template <> struct quokka::EOS_Traits<ShadowProblem> {
	static constexpr double mean_molecular_weight = 10. * C::m_u;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<ShadowProblem> {
	static constexpr double Erad_floor = 0.;
	static constexpr int beta_order = 1;
};

template <> struct Physics_Traits<ShadowProblem> {
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShadowProblem>::ComputePlanckOpacity(const double rho, const double /*Tgas*/) -> amrex::Real
{
	const amrex::Real sigma = sigma0 * std::pow(rho / rho_bg, 2);
	const amrex::Real kappa = sigma / rho; // specific opacity [cm^2 g^-1]
	return kappa;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ShadowProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShadowProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							  amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
							  int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	const amrex::Box &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();

	if (i < lo[0]) {
		// free-streaming boundary condition
		const double Egas = consVar(lo[0], j, k, RadSystem<ShadowProblem>::gasEnergy_index);
		const double rho = consVar(lo[0], j, k, RadSystem<ShadowProblem>::gasDensity_index);
		const double px = consVar(lo[0], j, k, RadSystem<ShadowProblem>::x1GasMomentum_index);
		const double py = consVar(lo[0], j, k, RadSystem<ShadowProblem>::x2GasMomentum_index);
		const double pz = consVar(lo[0], j, k, RadSystem<ShadowProblem>::x3GasMomentum_index);

		const double E_inc = a_rad * std::pow(T_hohlraum, 4);
		const double Fx_bdry = c * E_inc; // free-streaming (F/cE == 1)
		const double Fy_bdry = 0.;
		const double Fz_bdry = 0.;

		// x1 left side boundary (streaming)
		consVar(i, j, k, RadSystem<ShadowProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<ShadowProblem>::x1RadFlux_index) = Fx_bdry;
		consVar(i, j, k, RadSystem<ShadowProblem>::x2RadFlux_index) = Fy_bdry;
		consVar(i, j, k, RadSystem<ShadowProblem>::x3RadFlux_index) = Fz_bdry;

		// extrapolated/outflow boundary for gas variables
		consVar(i, j, k, RadSystem<ShadowProblem>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<ShadowProblem>::gasDensity_index) = rho;
		consVar(i, j, k, RadSystem<ShadowProblem>::gasInternalEnergy_index) = Egas - (px * px + py * py + pz * pz) / (2 * rho);
		consVar(i, j, k, RadSystem<ShadowProblem>::x1GasMomentum_index) = px;
		consVar(i, j, k, RadSystem<ShadowProblem>::x2GasMomentum_index) = py;
		consVar(i, j, k, RadSystem<ShadowProblem>::x3GasMomentum_index) = pz;
	}
}

template <> void QuokkaSimulation<ShadowProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
		amrex::Real const xc = 0.5;
		amrex::Real const yc = 0.0;
		amrex::Real const x0 = 0.1;
		amrex::Real const y0 = 0.06;

		amrex::Real const Delta = 10. * (std::pow((x - xc) / x0, 2) + std::pow((y - yc) / y0, 2) - 1.0);
		amrex::Real const rho = rho_bg + (rho_clump - rho_bg) / (1.0 + std::exp(Delta));

		amrex::Real const Erad = a_rad * std::pow(T_initial, 4);
		amrex::Real const Egas = quokka::EOS<ShadowProblem>::ComputeEintFromTgas(rho, T_initial);

		state_cc(i, j, k, RadSystem<ShadowProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<ShadowProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ShadowProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ShadowProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<ShadowProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<ShadowProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<ShadowProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<ShadowProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<ShadowProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<ShadowProblem>::x3GasMomentum_index) = 0.;
	});
}

template <> void QuokkaSimulation<ShadowProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.1; // gradient refinement threshold
	const amrex::Real erad_min = 1.0e-3;   // minimum erad for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const P = state(i, j, k, RadSystem<ShadowProblem>::radEnergy_index);
			amrex::Real const P_xplus = state(i + 1, j, k, RadSystem<ShadowProblem>::radEnergy_index);
			amrex::Real const P_xminus = state(i - 1, j, k, RadSystem<ShadowProblem>::radEnergy_index);
			amrex::Real const P_yplus = state(i, j + 1, k, RadSystem<ShadowProblem>::radEnergy_index);
			amrex::Real const P_yminus = state(i, j - 1, k, RadSystem<ShadowProblem>::radEnergy_index);

			amrex::Real const del_x = std::max(std::abs(P_xplus - P), std::abs(P - P_xminus));
			amrex::Real const del_y = std::max(std::abs(P_yplus - P), std::abs(P - P_yminus));

			amrex::Real const gradient_indicator = std::max(del_x, del_y) / std::max(P, erad_min);

			if (gradient_indicator > eta_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

auto problem_main() -> int
{
	// Problem parameters
	constexpr int max_timesteps = 20000;
	constexpr double CFL_number = 0.4;
	constexpr int nx = 560;		     // 280;
	constexpr int ny = 160;		     // 80;
	constexpr double Lx = 1.0;	     // cm
	constexpr double Ly = 0.12;	     // cm
	constexpr double max_time = 5.0e-11; // 10.0 * (Lx / c); // s

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<ShadowProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<ShadowProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	amrex::Vector<amrex::BCRec> BCs_cc(Physics_Indices<ShadowProblem>::nvarTotal_cc);
	for (int n = 0; n < Physics_Indices<ShadowProblem>::nvarTotal_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // left x1 -- streaming
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // right x1 -- extrapolate
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			if (isNormalComp(n, i)) { // reflect lower
				BCs_cc[n].setLo(i, amrex::BCType::reflect_odd);
			} else {
				BCs_cc[n].setLo(i, amrex::BCType::reflect_even);
			}
			// extrapolate upper
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Print radiation epsilon ("stiffness parameter" from Su & Olson).
	// (if epsilon is smaller than tolerance, there can be unphysical radiation shocks.)
	const auto dt_CFL = CFL_number * std::min(Lx / nx, Ly / ny) / c;
	const auto c_v = C::k_B / (quokka::EOS_Traits<ShadowProblem>::mean_molecular_weight * (quokka::EOS_Traits<ShadowProblem>::gamma - 1.0));
	const auto epsilon = 4.0 * a_rad * (T_initial * T_initial * T_initial) * sigma0 * (c * dt_CFL) / c_v;
	amrex::Print() << "radiation epsilon (stiffness parameter) = " << epsilon << "\n";

	// radiation-matter implicit solver must have a tolerance near machine precision!
	// const double resid_tol = 1.0e-15;

	// Problem initialization
	QuokkaSimulation<ShadowProblem> sim(BCs_cc);
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = 50; // for debugging

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
