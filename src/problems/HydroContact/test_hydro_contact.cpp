//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_contact.cpp
/// \brief Defines a test problem for a contact wave.
///

#include "AMReX_BC_TYPES.H"
#include "AMReX_BLassert.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"

#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "test_hydro_contact.hpp"
#include "util/fextract.hpp"

struct ContactProblem {
};

template <> struct quokka::EOS_Traits<ContactProblem> {
	static constexpr double gamma = 1.4;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<ContactProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 2; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

constexpr double v_contact = 0.0; // contact wave velocity

template <> void QuokkaSimulation<ContactProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	int ncomp_cc = Physics_Indices<ContactProblem>::nvarTotal_cc;
	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];

		double vx = NAN;
		double rho = NAN;
		double P = NAN;

		if (x < 0.5) {
			rho = 1.4;
			vx = v_contact;
			P = 1.0;
		} else {
			rho = 1.0;
			vx = v_contact;
			P = 1.0;
		}
		AMREX_ASSERT(!std::isnan(vx));
		AMREX_ASSERT(!std::isnan(rho));
		AMREX_ASSERT(!std::isnan(P));

		for (int n = 0; n < ncomp_cc; ++n) {
			state_cc(i, j, k, n) = 0.;
		}
		state_cc(i, j, k, HydroSystem<ContactProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<ContactProblem>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<ContactProblem>::x2Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ContactProblem>::x3Momentum_index) = 0.;
		state_cc(i, j, k, HydroSystem<ContactProblem>::energy_index) = quokka::EOS<ContactProblem>::ComputeEintFromPres(rho, P) + 0.5 * rho * (vx * vx);
		state_cc(i, j, k, HydroSystem<ContactProblem>::internalEnergy_index) = quokka::EOS<ContactProblem>::ComputeEintFromPres(rho, P);
	});
}

template <>
void QuokkaSimulation<ContactProblem>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
								amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	for (amrex::MFIter iter(ref); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = ref.array(iter);
		auto const ncomp = ref.nComp();

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			double vx = NAN;
			double rho = NAN;
			double P = NAN;
			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];

			if (x < 0.5) {
				rho = 1.4;
				vx = v_contact;
				P = 1.0;
			} else {
				rho = 1.0;
				vx = v_contact;
				P = 1.0;
			}

			for (int n = 0; n < ncomp; ++n) {
				stateExact(i, j, k, n) = 0.;
			}

			stateExact(i, j, k, HydroSystem<ContactProblem>::density_index) = rho;
			stateExact(i, j, k, HydroSystem<ContactProblem>::x1Momentum_index) = rho * vx;
			stateExact(i, j, k, HydroSystem<ContactProblem>::x2Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ContactProblem>::x3Momentum_index) = 0.;
			stateExact(i, j, k, HydroSystem<ContactProblem>::energy_index) =
			    quokka::EOS<ContactProblem>::ComputeEintFromPres(rho, P) + 0.5 * rho * (vx * vx);
			stateExact(i, j, k, HydroSystem<ContactProblem>::internalEnergy_index) = quokka::EOS<ContactProblem>::ComputeEintFromPres(rho, P);
		});
	}

#ifdef HAVE_PYTHON
	// Plot results
	auto [position, values] = fextract(state_new_cc_[0], geom[0], 0, 0.5);
	auto [pos_exact, val_exact] = fextract(ref, geom[0], 0, 0.5);
	auto const nx = position.size();

	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> x(nx);
		std::vector<double> d_final(nx);
		std::vector<double> vx_final(nx);
		std::vector<double> P_final(nx);
		std::vector<double> d_exact(nx);
		std::vector<double> vx_exact(nx);
		std::vector<double> P_exact(nx);

		for (int i = 0; i < nx; ++i) {
			amrex::Real const this_x = position[i];
			x.push_back(this_x);

			{
				const auto rho = val_exact.at(HydroSystem<ContactProblem>::density_index)[i];
				const auto xmom = val_exact.at(HydroSystem<ContactProblem>::x1Momentum_index)[i];
				const auto E = val_exact.at(HydroSystem<ContactProblem>::energy_index)[i];
				const auto vx = xmom / rho;
				const auto Eint = E - 0.5 * rho * (vx * vx);
				const auto P = quokka::EOS<ContactProblem>::ComputePressure(rho, Eint);
				d_exact.push_back(rho);
				vx_exact.push_back(vx);
				P_exact.push_back(P);
			}

			{
				const auto frho = values.at(HydroSystem<ContactProblem>::density_index)[i];
				const auto fxmom = values.at(HydroSystem<ContactProblem>::x1Momentum_index)[i];
				const auto fE = values.at(HydroSystem<ContactProblem>::energy_index)[i];
				const auto fvx = fxmom / frho;
				const auto fEint = fE - 0.5 * frho * (fvx * fvx);
				const auto fP = (quokka::EOS_Traits<ContactProblem>::gamma - 1.) * fEint;
				d_final.push_back(frho);
				vx_final.push_back(fvx);
				P_final.push_back(fP);
			}
		}

		std::unordered_map<std::string, std::string> d_args;
		std::map<std::string, std::string> dexact_args;
		d_args["label"] = "density";
		d_args["color"] = "black";
		dexact_args["label"] = "density (exact solution)";
		matplotlibcpp::scatter(x, d_final, 10.0, d_args);
		matplotlibcpp::plot(x, d_exact, dexact_args);

		matplotlibcpp::legend();
		matplotlibcpp::title(std::format("t = {:.4f}", tNew_[0]));
		matplotlibcpp::save("./hydro_contact.pdf");
	}
#endif
}

auto problem_main() -> int
{
	// Problem parameters
	const int ncomp_cc = Physics_Indices<ContactProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[0].setLo(0, amrex::BCType::int_dir); // periodic
		BCs_cc[0].setHi(0, amrex::BCType::int_dir);
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<ContactProblem> sim(BCs_cc);

	sim.stopTime_ = 2.0;
	sim.cflNumber_ = 0.8;
	sim.maxTimesteps_ = 2000;
	sim.computeReferenceSolution_ = true;
	sim.plotfileInterval_ = -1;

	// initialize and evolve
	sim.setInitialConditions();
	sim.evolve();

	// For a stationary isolated contact wave using the HLLC solver,
	// the error should be *exactly* (i.e., to *every* digit) zero.
	// [See Section 10.7 and Figure 10.20 of Toro (1998).]
	const double error_tol = 0.0; // this is not a typo
	int status = 0;
	if (sim.errorNorm_ > error_tol) {
		status = 1;
	}

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}
