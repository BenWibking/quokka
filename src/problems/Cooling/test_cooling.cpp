//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_cooling.cpp
/// \brief Defines a test problem for SUNDIALS cooling.
///
#include <random>

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_TableData.H"

#include "QuokkaSimulation.hpp"
#include "radiation/radiation_system.hpp"

using amrex::Real;

struct CoolingTest {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr double m_H = C::m_u;
constexpr double seconds_in_year = 3.154e7;

template <> struct quokka::EOS_Traits<CoolingTest> {
	static constexpr double gamma = 5. / 3.; // default value
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<CoolingTest> {
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

template <> struct SimulationData<CoolingTest> {
	std::unique_ptr<amrex::TableData<Real, 3>> table_data;
};

constexpr double Tgas0 = 6000.;	   // K
constexpr double rho0 = 0.6 * m_H; // g cm^-3

// perturbation parameters
const int kmin = 0;
const int kmax = 16;
Real const A = 0.05 / kmax;

template <> void QuokkaSimulation<CoolingTest>::preCalculateInitialConditions()
{
	// generate random phases
	amrex::Array<int, 3> tlo{kmin, kmin, kmin}; // lower bounds
	amrex::Array<int, 3> thi{kmax, kmax, kmax}; // upper bounds
	userData_.table_data = std::make_unique<amrex::TableData<Real, 3>>(tlo, thi);

	amrex::TableData<Real, 3> h_table_data(tlo, thi, amrex::The_Pinned_Arena());
	auto const &h_table = h_table_data.table();

	// 64-bit Mersenne Twister (do not use 32-bit version for sampling doubles!)
	std::mt19937_64 rng(1); // NOLINT
	std::uniform_real_distribution<double> sample_phase(0., 2.0 * M_PI);

	// Initialize data on the host
	for (int j = tlo[0]; j <= thi[0]; ++j) {
		for (int i = tlo[1]; i <= thi[1]; ++i) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				h_table(i, j, k) = sample_phase(rng);
			}
		}
	}

	// Copy data to GPU memory
	userData_.table_data->copy(h_table_data);
	amrex::Gpu::streamSynchronize();
}

template <> void QuokkaSimulation<CoolingTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const auto &phase_table = userData_.table_data->const_table();

	Real const Lx = (prob_hi[0] - prob_lo[0]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const x = prob_lo[0] + (i + static_cast<Real>(0.5)) * dx[0];
		Real const y = prob_lo[1] + (j + static_cast<Real>(0.5)) * dx[1];
		Real const z = prob_lo[2] + (k + static_cast<Real>(0.5)) * dx[2];

		// compute perturbations
		Real delta_rho = 0;
		for (int ki = kmin; ki < kmax; ++ki) {
			for (int kj = kmin; kj < kmax; ++kj) {
				for (int kk = kmin; kk < kmax; ++kk) {
					if ((ki == 0) && (kj == 0) && (kk == 0)) {
						continue;
					}
					Real const kx = 2.0 * M_PI * static_cast<Real>(ki) / Lx;
					Real const ky = 2.0 * M_PI * static_cast<Real>(kj) / Lx;
					Real const kz = 2.0 * M_PI * static_cast<Real>(kk) / Lx;
					delta_rho += A * std::sin(x * kx + y * ky + z * kz + phase_table(ki, kj, kk));
				}
			}
		}
		AMREX_ALWAYS_ASSERT(delta_rho > -1.0);

		Real rho = 0.12 * m_H * (1.0 + delta_rho); // g cm^-3
		Real xmom = 0;
		Real ymom = 0;
		Real zmom = 0;
		Real const P = 4.0e4 * C::k_B; // erg cm^-3
		Real Eint = (quokka::EOS_Traits<CoolingTest>::gamma - 1.) * P;

		Real const Egas = RadSystem<CoolingTest>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		state_cc(i, j, k, RadSystem<CoolingTest>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<CoolingTest>::gasInternalEnergy_index) = Eint;
		state_cc(i, j, k, RadSystem<CoolingTest>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<CoolingTest>::x1GasMomentum_index) = xmom;
		state_cc(i, j, k, RadSystem<CoolingTest>::x2GasMomentum_index) = ymom;
		state_cc(i, j, k, RadSystem<CoolingTest>::x3GasMomentum_index) = zmom;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<CoolingTest>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
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
	amrex::GpuArray<int, 3> hi = box.hiVect3d();

	if (j >= hi[1]) {
		// x2 upper boundary -- constant
		Real rho = rho0;
		Real xmom = 0;
		Real ymom = rho * (-26.0e5); // [-26 km/s]
		Real zmom = 0;
		Real Eint = quokka::EOS<CoolingTest>::ComputeEintFromTgas(rho, Tgas0);
		Real const Egas = RadSystem<CoolingTest>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);

		consVar(i, j, k, RadSystem<CoolingTest>::gasDensity_index) = rho;
		consVar(i, j, k, RadSystem<CoolingTest>::x1GasMomentum_index) = xmom;
		consVar(i, j, k, RadSystem<CoolingTest>::x2GasMomentum_index) = ymom;
		consVar(i, j, k, RadSystem<CoolingTest>::x3GasMomentum_index) = zmom;
		consVar(i, j, k, RadSystem<CoolingTest>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<CoolingTest>::gasInternalEnergy_index) = Eint;
	}
}

auto problem_main() -> int
{
	// Problem parameters
	const double CFL_number = 0.25;
	const double max_time = 7.5e4 * seconds_in_year; // 75 kyr
	const int max_timesteps = 2e4;

	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<CoolingTest>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::int_dir); // periodic
		BCs_cc[n].setHi(0, amrex::BCType::int_dir);
		BCs_cc[n].setLo(1, amrex::BCType::foextrap); // extrapolate
		BCs_cc[n].setHi(1, amrex::BCType::ext_dir);  // Dirichlet
#if AMREX_SPACEDIM == 3
		BCs_cc[n].setLo(2, amrex::BCType::int_dir); // periodic
		BCs_cc[n].setHi(2, amrex::BCType::int_dir);
#endif
	}

	QuokkaSimulation<CoolingTest> sim(BCs_cc);

	// Standard PPM gives unphysically enormous temperatures when used for
	// this problem (e.g., ~1e14 K or higher), but can be fixed by
	// reconstructing the temperature instead of the pressure
	sim.reconstructionOrder_ = 3; // PLM

	sim.cflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = 100;
	sim.checkpointInterval_ = -1;

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Cleanup and exit
	int status = 0;
	return status;
}
