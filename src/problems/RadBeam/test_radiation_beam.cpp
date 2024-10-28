//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_beam.cpp
/// \brief Defines a test problem for radiation in the streaming regime.
///

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_IntVect.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"

struct BeamProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double kappa0 = 0.0;	     // cm^2 g^-1 (specific opacity)
constexpr double rho0 = 1.0;	     // g cm^-3 (matter density)
constexpr double T_hohlraum = 1000.; // K
constexpr double T_initial = 300.;   // K

constexpr double a_rad = 7.5646e-15; // erg cm^-3 K^-4
constexpr double c = 2.99792458e10;  // cm s^-1

template <> struct quokka::EOS_Traits<BeamProblem> {
	static constexpr double mean_molecular_weight = C::m_u;
	static constexpr double gamma = 5. / 3.;
};

template <> struct RadSystem_Traits<BeamProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 0.;
	static constexpr int beta_order = 1;
};

template <> struct Physics_Traits<BeamProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<BeamProblem>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<BeamProblem>::ComputeFluxMeanOpacity(const double /*rho*/, const double /*Tgas*/) -> amrex::Real
{
	return kappa0;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<BeamProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
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

	amrex::Real const *dx = geom.CellSize();
	amrex::Real const *prob_lo = geom.ProbLo();
	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();

	amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

	if ((i < lo[0]) && !(j < lo[1])) {
		// streaming boundary condition

		const double E_0 = consVar(lo[0], j, k, RadSystem<BeamProblem>::radEnergy_index);
		const double Fx_0 = consVar(lo[0], j, k, RadSystem<BeamProblem>::x1RadFlux_index);
		const double Fy_0 = consVar(lo[0], j, k, RadSystem<BeamProblem>::x2RadFlux_index);
		const double Fz_0 = consVar(lo[0], j, k, RadSystem<BeamProblem>::x3RadFlux_index);

		const double Egas = consVar(lo[0], j, k, RadSystem<BeamProblem>::gasEnergy_index);
		const double rho = consVar(lo[0], j, k, RadSystem<BeamProblem>::gasDensity_index);
		const double px = consVar(lo[0], j, k, RadSystem<BeamProblem>::x1GasMomentum_index);
		const double py = consVar(lo[0], j, k, RadSystem<BeamProblem>::x2GasMomentum_index);
		const double pz = consVar(lo[0], j, k, RadSystem<BeamProblem>::x3GasMomentum_index);

		double E_inc = NAN;
		double Fx_bdry = NAN;
		double Fy_bdry = NAN;
		double Fz_bdry = NAN;

		const double y_max = 0.0625;

		if (y <= y_max) {
			E_inc = a_rad * std::pow(T_hohlraum, 4);
			Fx_bdry = (1.0 / std::sqrt(2.0)) * c * E_inc;
			Fy_bdry = (1.0 / std::sqrt(2.0)) * c * E_inc;
			Fz_bdry = 0.;
		} else {
			// reflecting/absorbing boundary
			E_inc = E_0;
			Fx_bdry = -Fx_0;
			Fy_bdry = Fy_0;
			Fz_bdry = Fz_0;
		}

		// x1, x2 left side boundary
		consVar(i, j, k, RadSystem<BeamProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<BeamProblem>::x1RadFlux_index) = Fx_bdry;
		consVar(i, j, k, RadSystem<BeamProblem>::x2RadFlux_index) = Fy_bdry;
		consVar(i, j, k, RadSystem<BeamProblem>::x3RadFlux_index) = Fz_bdry;

		// extrapolated/outflow boundary for gas variables
		consVar(i, j, k, RadSystem<BeamProblem>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<BeamProblem>::gasDensity_index) = rho;
		consVar(i, j, k, RadSystem<BeamProblem>::gasInternalEnergy_index) = Egas - (px * px + py * py + pz * pz) / (2 * rho);
		consVar(i, j, k, RadSystem<BeamProblem>::x1GasMomentum_index) = px;
		consVar(i, j, k, RadSystem<BeamProblem>::x2GasMomentum_index) = py;
		consVar(i, j, k, RadSystem<BeamProblem>::x3GasMomentum_index) = pz;
	} else if ((j < lo[1]) && !(i < lo[0])) {
		// streaming boundary condition
		double E_inc = NAN;

		const double E_0 = consVar(i, lo[1], k, RadSystem<BeamProblem>::radEnergy_index);
		const double Fx_0 = consVar(i, lo[1], k, RadSystem<BeamProblem>::x1RadFlux_index);
		const double Fy_0 = consVar(i, lo[1], k, RadSystem<BeamProblem>::x2RadFlux_index);
		const double Fz_0 = consVar(i, lo[1], k, RadSystem<BeamProblem>::x3RadFlux_index);

		const double Egas = consVar(i, lo[1], k, RadSystem<BeamProblem>::gasEnergy_index);
		const double rho = consVar(i, lo[1], k, RadSystem<BeamProblem>::gasDensity_index);
		const double px = consVar(i, lo[1], k, RadSystem<BeamProblem>::x1GasMomentum_index);
		const double py = consVar(i, lo[1], k, RadSystem<BeamProblem>::x2GasMomentum_index);
		const double pz = consVar(i, lo[1], k, RadSystem<BeamProblem>::x3GasMomentum_index);

		double Fx_bdry = NAN;
		double Fy_bdry = NAN;
		double Fz_bdry = NAN;

		const double x_max = 0.0625;

		if (x <= x_max) {
			E_inc = a_rad * std::pow(T_hohlraum, 4);
			Fx_bdry = (1.0 / std::sqrt(2.0)) * c * E_inc;
			Fy_bdry = (1.0 / std::sqrt(2.0)) * c * E_inc;
			Fz_bdry = 0.;
		} else {
			// reflecting/absorbing boundary
			E_inc = E_0;
			Fx_bdry = Fx_0;
			Fy_bdry = -Fy_0;
			Fz_bdry = Fz_0;
		}

		// x1, x2 left side boundary
		consVar(i, j, k, RadSystem<BeamProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<BeamProblem>::x1RadFlux_index) = Fx_bdry;
		consVar(i, j, k, RadSystem<BeamProblem>::x2RadFlux_index) = Fy_bdry;
		consVar(i, j, k, RadSystem<BeamProblem>::x3RadFlux_index) = Fz_bdry;

		// extrapolated/outflow boundary for gas variables
		consVar(i, j, k, RadSystem<BeamProblem>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<BeamProblem>::gasDensity_index) = rho;
		consVar(i, j, k, RadSystem<BeamProblem>::gasInternalEnergy_index) = Egas - (px * px + py * py + pz * pz) / (2 * rho);
		consVar(i, j, k, RadSystem<BeamProblem>::x1GasMomentum_index) = px;
		consVar(i, j, k, RadSystem<BeamProblem>::x2GasMomentum_index) = py;
		consVar(i, j, k, RadSystem<BeamProblem>::x3GasMomentum_index) = pz;
	} else if ((i < lo[0]) && (j < lo[1])) {
		// streaming boundary condition

		const double Egas = consVar(lo[0], lo[1], k, RadSystem<BeamProblem>::gasEnergy_index);
		const double rho = consVar(lo[0], lo[1], k, RadSystem<BeamProblem>::gasDensity_index);
		const double px = consVar(lo[0], lo[1], k, RadSystem<BeamProblem>::x1GasMomentum_index);
		const double py = consVar(lo[0], lo[1], k, RadSystem<BeamProblem>::x2GasMomentum_index);
		const double pz = consVar(lo[0], lo[1], k, RadSystem<BeamProblem>::x3GasMomentum_index);

		double E_inc = a_rad * std::pow(T_hohlraum, 4);
		double Fx_bdry = (1.0 / std::sqrt(2.0)) * c * E_inc;
		double Fy_bdry = (1.0 / std::sqrt(2.0)) * c * E_inc;
		double Fz_bdry = 0.;

		// x1, x2 left side boundary
		consVar(i, j, k, RadSystem<BeamProblem>::radEnergy_index) = E_inc;
		consVar(i, j, k, RadSystem<BeamProblem>::x1RadFlux_index) = Fx_bdry;
		consVar(i, j, k, RadSystem<BeamProblem>::x2RadFlux_index) = Fy_bdry;
		consVar(i, j, k, RadSystem<BeamProblem>::x3RadFlux_index) = Fz_bdry;

		// extrapolated/outflow boundary for gas variables
		consVar(i, j, k, RadSystem<BeamProblem>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<BeamProblem>::gasDensity_index) = rho;
		consVar(i, j, k, RadSystem<BeamProblem>::gasInternalEnergy_index) = Egas - (px * px + py * py + pz * pz) / (2 * rho);
		consVar(i, j, k, RadSystem<BeamProblem>::x1GasMomentum_index) = px;
		consVar(i, j, k, RadSystem<BeamProblem>::x2GasMomentum_index) = py;
		consVar(i, j, k, RadSystem<BeamProblem>::x3GasMomentum_index) = pz;
	}
}

template <> void QuokkaSimulation<BeamProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double Erad = a_rad * std::pow(T_initial, 4);
		const double rho = rho0;
		const double Egas = quokka::EOS<BeamProblem>::ComputeEintFromTgas(rho, T_initial);

		state_cc(i, j, k, RadSystem<BeamProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<BeamProblem>::x1RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<BeamProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<BeamProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<BeamProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<BeamProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<BeamProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<BeamProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<BeamProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<BeamProblem>::x3GasMomentum_index) = 0.;
	});
}

template <> void QuokkaSimulation<BeamProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.1; // gradient refinement threshold
	const amrex::Real erad_min = 1.0e-3;   // minimum erad for refinement

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const P = state(i, j, k, RadSystem<BeamProblem>::radEnergy_index);
			amrex::Real const P_xplus = state(i + 1, j, k, RadSystem<BeamProblem>::radEnergy_index);
			amrex::Real const P_xminus = state(i - 1, j, k, RadSystem<BeamProblem>::radEnergy_index);
			amrex::Real const P_yplus = state(i, j + 1, k, RadSystem<BeamProblem>::radEnergy_index);
			amrex::Real const P_yminus = state(i, j - 1, k, RadSystem<BeamProblem>::radEnergy_index);

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
	const int max_timesteps = 10000;
	const double CFL_number = 0.4;
	// const int nx = 128;
	const double Lx = 2.0;			// cm
	const double max_time = 2.0 * (Lx / c); // s

	constexpr int nvars = RadSystem<BeamProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // left x1 -- inflow
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // right x1 -- extrapolate
		BCs_cc[n].setLo(1, amrex::BCType::ext_dir);  // left x2 -- inflow
		BCs_cc[n].setHi(1, amrex::BCType::foextrap); // right x2 -- extrapolate
		if (AMREX_SPACEDIM == 3) {
			BCs_cc[n].setLo(2, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(2, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<BeamProblem> sim(BCs_cc);

	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.radiationReconstructionOrder_ = 2; // PLM
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = 20; // for debugging

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
