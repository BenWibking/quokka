//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_quirk.cpp
/// \brief Defines a test problem for the odd-even decoupling instability.
///

#include <algorithm>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FabArrayBase.H"
#include "AMReX_GpuAsyncArray.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"

using Real = amrex::Real;

struct QuirkProblem {
};

template <> struct quokka::EOS_Traits<QuirkProblem> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<QuirkProblem> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<QuirkProblem> {
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

constexpr Real dl = 3.692;
constexpr Real ul = -0.625;
constexpr Real pl = 26.85;
constexpr Real dr = 1.0;
constexpr Real ur = -5.0;
constexpr Real pr = 0.6;
int ishock_g = 0;

template <> void QuokkaSimulation<QuirkProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	Real xshock = 0.4;
	int ishock = 0;
	for (ishock = 0; (prob_lo[0] + dx[0] * (ishock + static_cast<Real>(0.5))) < xshock; ++ishock) {
	}
	ishock--;
	amrex::Print() << "ishock = " << ishock << "\n";

	Real dd = dl - 0.135;
	Real ud = ul + 0.219;
	Real pd = pl - 1.31;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		double vx = NAN;
		double vy = 0.;
		double vz = 0.;
		double rho = NAN;
		double P = NAN;

		if (i <= ishock) {
			rho = dl;
			vx = ul;
			P = pl;
		} else {
			rho = dr;
			vx = ur;
			P = pr;
		}

		if ((i == ishock) && (j % 2 == 0)) {
			rho = dd;
			vx = ud;
			P = pd;
		}

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(vy));
		AMREX_ASSERT(!std::isnan(vz));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		const auto v_sq = vx * vx + vy * vy + vz * vz;

		state_cc(i, j, k, HydroSystem<QuirkProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<QuirkProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<QuirkProblem>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<QuirkProblem>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<QuirkProblem>::energy_index) = quokka::EOS<QuirkProblem>::ComputeEintFromPres(rho, P) + 0.5 * rho * v_sq;
		state_cc(i, j, k, HydroSystem<QuirkProblem>::internalEnergy_index) = quokka::EOS<QuirkProblem>::ComputeEintFromPres(rho, P);
	});
}

auto getDeltaEntropyVector() -> std::vector<Real> &
{
	static std::vector<Real> delta_s_vec;
	return delta_s_vec;
}

template <> void QuokkaSimulation<QuirkProblem>::computeAfterTimestep()
{
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// it should be sufficient examine a single box on level 0
		// (no AMR should be used for this problem, and the odd-even decoupling will
		// manifest in every row along the shock, if it happens)

		amrex::MultiFab &mf_state = state_new_cc_[0];
		int box_no = -1;
		int ilo = ishock_g;
		int jlo = 0;
		int klo = 0;
		for (amrex::MFIter mfi(mf_state); mfi.isValid(); ++mfi) {
			const amrex::Box &bx = mfi.validbox();
			amrex::GpuArray<int, 3> box_lo = bx.loVect3d();
			jlo = box_lo[1];
			klo = box_lo[2];
			amrex::IntVect cell{AMREX_D_DECL(ilo, jlo, klo)};
			if (bx.contains(cell)) {
				box_no = mfi.index();
				break;
			}
		}

		AMREX_ALWAYS_ASSERT(box_no != -1);
		auto const &state = mf_state.const_array(box_no);
		amrex::Box bx = amrex::makeSingleCellBox(ilo, jlo, klo);
		Real host_s = NAN;
		amrex::AsyncArray async_s(&host_s, 1);
		Real *s = async_s.data();

		amrex::launch(bx, [=] AMREX_GPU_DEVICE(amrex::Box const &tbx) {
			amrex::GpuArray<int, 3> const idx = tbx.loVect3d();
			int i = idx[0];
			int j = idx[1];
			int k = idx[2];
			Real dodd = state(i, j + 1, k, HydroSystem<QuirkProblem>::density_index);
			Real podd = HydroSystem<QuirkProblem>::ComputePressure(state, i, j + 1, k);
			Real deven = state(i, j, k, HydroSystem<QuirkProblem>::density_index);
			Real peven = HydroSystem<QuirkProblem>::ComputePressure(state, i, j, k);

			// the 'entropy function' s == P / rho^gamma
			const Real gamma = quokka::EOS_Traits<QuirkProblem>::gamma;
			Real sodd = podd / std::pow(dodd, gamma);
			Real seven = peven / std::pow(deven, gamma);
			s[0] = std::abs(sodd - seven);
		});

		async_s.copyToHost(&host_s, 1);
		getDeltaEntropyVector().push_back(host_s);
	}
}

template <> void QuokkaSimulation<QuirkProblem>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	if (amrex::ParallelDescriptor::IOProcessor()) {
		auto const &deltas_vec = getDeltaEntropyVector();
		const Real deltas = *std::max_element(deltas_vec.begin(), deltas_vec.end());

		if (deltas > 0.06) {
			amrex::Print() << "The scheme suffers from the Carbuncle phenomenon : max delta s = " << deltas << "\n\n";
			amrex::Abort("Carbuncle detected!");
		} else {
			amrex::Print() << "The scheme looks stable against the Carbuncle phenomenon : "
					  "max delta s = "
				       << deltas << "\n\n";
		}
	}
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<QuirkProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							 amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
							 int /*bcomp*/, int /*orig_comp*/)
{
#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int j = 0;
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 2)
	auto [i, j] = iv.toArray();
	int k = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	auto [i, j, k] = iv.toArray();
#endif

	amrex::Box const &box = geom.Domain();
	amrex::GpuArray<int, 3> lo = box.loVect3d();
	amrex::GpuArray<int, 3> hi = box.hiVect3d();
	const auto gamma = quokka::EOS_Traits<QuirkProblem>::gamma;

	if (i < lo[0]) {
		// x1 left side boundary
		consVar(i, j, k, RadSystem<QuirkProblem>::gasEnergy_index) = pl / (gamma - 1.) + 0.5 * dl * ul * ul;
		consVar(i, j, k, RadSystem<QuirkProblem>::gasInternalEnergy_index) = pl / (gamma - 1.);
		consVar(i, j, k, RadSystem<QuirkProblem>::gasDensity_index) = dl;
		consVar(i, j, k, RadSystem<QuirkProblem>::x1GasMomentum_index) = dl * ul;
		consVar(i, j, k, RadSystem<QuirkProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<QuirkProblem>::x3GasMomentum_index) = 0.;
	} else if (i >= hi[0]) {
		// x1 right-side boundary
		consVar(i, j, k, RadSystem<QuirkProblem>::gasEnergy_index) = pr / (gamma - 1.) + 0.5 * dr * ur * ur;
		consVar(i, j, k, RadSystem<QuirkProblem>::gasInternalEnergy_index) = pr / (gamma - 1.);
		consVar(i, j, k, RadSystem<QuirkProblem>::gasDensity_index) = dr;
		consVar(i, j, k, RadSystem<QuirkProblem>::x1GasMomentum_index) = dr * ur;
		consVar(i, j, k, RadSystem<QuirkProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<QuirkProblem>::x3GasMomentum_index) = 0.;
	}
}

auto problem_main() -> int
{
	// Boundary conditions
	const int ncomp_cc = Physics_Indices<QuirkProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		// outflow
		BCs_cc[0].setLo(0, amrex::BCType::ext_dir);
		BCs_cc[0].setHi(0, amrex::BCType::ext_dir);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			// periodic
			BCs_cc[n].setLo(i, amrex::BCType::int_dir);
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<QuirkProblem> sim(BCs_cc);

	sim.reconstructionOrder_ = 2; // PLM
	sim.stopTime_ = 0.4;
	sim.cflNumber_ = 0.4;
	sim.maxTimesteps_ = 2000;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// Cleanup and exit
	amrex::Print() << "Finished." << '\n';
	return 0;
}
