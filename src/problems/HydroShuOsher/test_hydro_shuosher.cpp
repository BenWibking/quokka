//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_shocktube.cpp
/// \brief Defines a test problem for a shock tube.
///

#include <cmath>
#include <vector>

#include "AMReX_BC_TYPES.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "test_hydro_shuosher.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct ShocktubeProblem {
};

template <> struct quokka::EOS_Traits<ShocktubeProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ShocktubeProblem> {
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

template <> void QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const int ncomp_cc = Physics_Indices<ShocktubeProblem>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
		double vx = NAN;
		double rho = NAN;
		double P = NAN;

		if (x < 1.0) {
			rho = 3.857143;
			vx = 2.629369;
			P = 10.33333;
		} else {
			rho = 1.0 + 0.2 * std::sin(5.0 * x);
			vx = 0.0;
			P = 1.0;
		}

		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}

		const auto gamma = quokka::EOS_Traits<ShocktubeProblem>::gamma;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx);
		state_cc(i, j, k, HydroSystem<ShocktubeProblem>::internalEnergy_index) = P / (gamma - 1.);
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
							     int /*numcomp*/, amrex::GeometryData const &geom, const amrex::Real /*time*/,
							     const amrex::BCRec * /*bcr*/, int /*bcomp*/, int /*orig_comp*/)
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
	const auto gamma = quokka::EOS_Traits<ShocktubeProblem>::gamma;

	double vx = NAN;
	double rho = NAN;
	double P = NAN;

	if (i < lo[0]) {
		rho = 3.857143;
		vx = 2.629369;
		P = 10.33333;
	} else if (i >= hi[0]) {
		rho = 1.0;
		vx = 0.0;
		P = 1.0;
	}

	consVar(i, j, k, RadSystem<ShocktubeProblem>::gasDensity_index) = rho;
	consVar(i, j, k, RadSystem<ShocktubeProblem>::x1GasMomentum_index) = rho * vx;
	consVar(i, j, k, RadSystem<ShocktubeProblem>::x2GasMomentum_index) = 0;
	consVar(i, j, k, RadSystem<ShocktubeProblem>::x3GasMomentum_index) = 0;
	consVar(i, j, k, RadSystem<ShocktubeProblem>::gasEnergy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx);
	consVar(i, j, k, RadSystem<ShocktubeProblem>::gasInternalEnergy_index) = P / (gamma - 1.);
}

template <>
void QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{

	// read in exact solution
	std::vector<double> xs_exact;
	std::vector<double> density_exact;
	std::vector<double> pressure_exact;
	std::vector<double> velocity_exact;

	std::string filename = "../extern/ShuOsher_athena_3c_hllc_vl.txt";
	std::ifstream fstream(filename, std::ios::in);
	AMREX_ALWAYS_ASSERT(fstream.is_open());

	std::string header;
	std::string blank_line;
	std::getline(fstream, header);
	std::getline(fstream, blank_line);

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;

		for (double value = NAN; iss >> value;) {
			values.push_back(value);
		}
		auto x = values.at(1);
		auto density = values.at(2);
		auto pressure = values.at(3);
		auto velocity = values.at(4);

		xs_exact.push_back(x);
		density_exact.push_back(density);
		pressure_exact.push_back(pressure);
		velocity_exact.push_back(velocity);
	}

	// interpolate exact solution onto coarse grid
	auto const box = geom[0].Domain();
	int nx = (box.hiVect3d()[0] - box.loVect3d()[0]) + 1;
	std::vector<double> xs(nx);
	for (int i = 0; i < nx; ++i) {
		xs.at(i) = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
	}

	std::vector<double> density_exact_interp(xs.size());
	std::vector<double> velocity_exact_interp(xs.size());
	std::vector<double> pressure_exact_interp(xs.size());

	interpolate_arrays(xs.data(), density_exact_interp.data(), static_cast<int>(xs.size()), xs_exact.data(), density_exact.data(),
			   static_cast<int>(xs_exact.size()));
	interpolate_arrays(xs.data(), velocity_exact_interp.data(), static_cast<int>(xs.size()), xs_exact.data(), velocity_exact.data(),
			   static_cast<int>(xs_exact.size()));
	interpolate_arrays(xs.data(), pressure_exact_interp.data(), static_cast<int>(xs.size()), xs_exact.data(), pressure_exact.data(),
			   static_cast<int>(xs_exact.size()));

	amrex::Gpu::DeviceVector<double> rho_g(density_exact_interp.size());
	amrex::Gpu::DeviceVector<double> vx_g(velocity_exact_interp.size());
	amrex::Gpu::DeviceVector<double> P_g(pressure_exact_interp.size());

	// copy exact solution to device
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, density_exact_interp.begin(), density_exact_interp.end(), rho_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, velocity_exact_interp.begin(), velocity_exact_interp.end(), vx_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, pressure_exact_interp.begin(), pressure_exact_interp.end(), P_g.begin());
	amrex::Gpu::streamSynchronizeAll();

	// fill reference solution multifab
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();
		auto const &rho_arr = rho_g.data();
		auto const &vx_arr = vx_g.data();
		auto const &P_arr = P_g.data();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}
			amrex::Real rho = rho_arr[i];
			amrex::Real vx = vx_arr[i];
			amrex::Real P = P_arr[i];

			const auto gamma = quokka::EOS_Traits<ShocktubeProblem>::gamma;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x1Momentum_index) = rho * vx;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::x3Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::energy_index) = P / (gamma - 1.) + 0.5 * rho * (vx * vx);
			stateExact(i, j, k, HydroSystem<ShocktubeProblem>::internalEnergy_index) = P / (gamma - 1.);
		});
	}

#ifdef HAVE_PYTHON
	// Plot results
	auto [position, values] = fextract(state_new_cc_[0], geom[0], 0, 0.5);
	auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.5);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// extract values
		std::vector<double> d(nx);
		std::vector<double> vx(nx);
		std::vector<double> P(nx);

		for (int i = 0; i < nx; ++i) {
			amrex::Real rho = values.at(HydroSystem<ShocktubeProblem>::density_index)[i];
			amrex::Real xmom = values.at(HydroSystem<ShocktubeProblem>::x1Momentum_index)[i];
			amrex::Real Egas = values.at(HydroSystem<ShocktubeProblem>::energy_index)[i];

			amrex::Real xvel = xmom / rho;
			amrex::Real Eint = Egas - xmom * xmom / (2.0 * rho);
			amrex::Real pressure = (quokka::EOS_Traits<ShocktubeProblem>::gamma - 1.) * Eint;

			d.at(i) = rho;
			vx.at(i) = xvel;
			P.at(i) = pressure;
		}

		matplotlibcpp::clf();
		std::unordered_map<std::string, std::string> d_args;
		std::map<std::string, std::string> dexact_args;
		d_args["label"] = "simulation";
		d_args["marker"] = "o";
		// d_args["edgecolors"] = "k";
		d_args["color"] = "C0";
		dexact_args["label"] = "reference solution";
		dexact_args["color"] = "C0";

		matplotlibcpp::scatter(xs, d, 5.0, d_args);
		matplotlibcpp::plot(xs_exact, density_exact, dexact_args);

		matplotlibcpp::legend();
		// matplotlibcpp::title(fmt::format("t = {:.4f}", tNew_[0]));
		matplotlibcpp::ylabel("density");
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(fmt::format("./hydro_shuosher.pdf", tNew_[0]));
	}
#endif
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 400;
	// const double Lx = 10.0;
	const double CFL_number = 0.2;
	const double max_time = 1.8;
	const double fixed_dt = 1e-4;
	const int max_timesteps = 20000;

	// Problem initialization
	const int ncomp_cc = Physics_Indices<ShocktubeProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[0].setLo(0, amrex::BCType::foextrap); // Dirichlet
		BCs_cc[0].setHi(0, amrex::BCType::ext_dir);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<ShocktubeProblem> sim(BCs_cc);

	sim.cflNumber_ = CFL_number;
	sim.constantDt_ = fixed_dt;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.computeReferenceSolution_ = true;
	sim.plotfileInterval_ = -1;

	// Main time loop
	sim.setInitialConditions();
	sim.evolve();

	// Compute test success condition
	int status = 0;
	const double error_tol = 0.01;
	if (sim.errorNorm_ > error_tol) {
		status = 1;
	}

	return status;
}
