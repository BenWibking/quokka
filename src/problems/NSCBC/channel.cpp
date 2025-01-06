//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file channel.cpp
/// \brief Implements a subsonic channel flow problem with Navier-Stokes
///        Characteristic Boundary Conditions (NSCBC).
///
#include <random>
#include <tuple>
#include <vector>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_BLProfiler.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"

#include "QuokkaSimulation.hpp"
#include "channel.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/HydroState.hpp"
#include "hydro/NSCBC_inflow.hpp"
#include "hydro/NSCBC_outflow.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "util/ArrayUtil.hpp"
#include "util/fextract.hpp"
#include "util/valarray.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

using amrex::Real;

struct Channel {
}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct quokka::EOS_Traits<Channel> {
	static constexpr double gamma = 1.1;
	static constexpr double mean_molecular_weight = 28.96 * C::m_u; // air
};

template <> struct Physics_Traits<Channel> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 1; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	static constexpr int nGroups = 1; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

// global variables needed for Dirichlet boundary condition and initial conditions
#if 0 // workaround AMDGPU compiler bug
namespace
{
#endif
Real rho0 = NAN;										// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
Real u0 = NAN;											// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
Real s0 = NAN;											// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real Tgas0 = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real P_outflow = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real u_inflow = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real v_inflow = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::Real w_inflow = NAN;							// NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
AMREX_GPU_MANAGED amrex::GpuArray<Real, Physics_Traits<Channel>::numPassiveScalars> s_inflow{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
#if 0												// workaround AMDGPU compiler bug
};											 // namespace
#endif

template <> void QuokkaSimulation<Channel>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	Real const rho = rho0;
	Real const xmom = rho0 * u0;
	Real const ymom = 0;
	Real const zmom = 0;
	Real const Eint = quokka::EOS<Channel>::ComputeEintFromTgas(rho0, Tgas0);
	Real const Egas = RadSystem<Channel>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint);
	Real const scalar = s0;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<Channel>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<Channel>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<Channel>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<Channel>::x3Momentum_index) = zmom;
		state_cc(i, j, k, HydroSystem<Channel>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<Channel>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<Channel>::scalar0_index) = scalar;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void AMRSimulation<Channel>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<Real> const &consVar,
											     int /*dcomp*/, int /*numcomp*/, amrex::GeometryData const &geom,
											     const Real /*time*/, const amrex::BCRec * /*bcr*/, int /*bcomp*/,
											     int /*orig_comp*/)
{
	auto [i, j, k] = iv.dim3();
	amrex::Box const &box = geom.Domain();
	const auto &domain_lo = box.loVect3d();
	const auto &domain_hi = box.hiVect3d();
	const int ilo = domain_lo[0];
	const int ihi = domain_hi[0];

	if (i < ilo) {
		NSCBC::setInflowX1Lower<Channel>(iv, consVar, geom, ::Tgas0, ::u_inflow, ::v_inflow, ::w_inflow, ::s_inflow);
	} else if (i > ihi) {
		NSCBC::setOutflowBoundary<Channel, FluxDir::X1, NSCBC::BoundarySide::Upper>(iv, consVar, geom, P_outflow);
	}
}

auto problem_main() -> int
{
	// Problem initialization
	constexpr int ncomp_cc = Physics_Indices<Channel>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir); // NSCBC inflow
		BCs_cc[n].setHi(0, amrex::BCType::ext_dir); // NSCBC outflow

		if constexpr (AMREX_SPACEDIM >= 2) {
			BCs_cc[n].setLo(1, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(1, amrex::BCType::int_dir);
		} else if (AMREX_SPACEDIM == 3) {
			BCs_cc[n].setLo(2, amrex::BCType::int_dir);
			BCs_cc[n].setHi(2, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<Channel> sim(BCs_cc);

	amrex::ParmParse const pp("channel");
	// initial condition parameters
	pp.query("rho0", ::rho0);   // initial density [g/cc]
	pp.query("Tgas0", ::Tgas0); // initial temperature [K]
	pp.query("u0", ::u0);	    // initial velocity [cm/s]
	pp.query("s0", ::s0);	    // initial passive scalar [dimensionless]
	// boundary condition parameters
	pp.query("u_inflow", ::u_inflow);    // inflow velocity along x-axis [cm/s]
	pp.query("v_inflow", ::v_inflow);    // transverse inflow velocity (v_y) [cm/s]
	pp.query("w_inflow", ::w_inflow);    // transverse inflow velocity (v_z) [cm/s]
	pp.query("s_inflow", ::s_inflow[0]); // inflow passive scalar [dimensionless]

	// compute derived parameters
	const Real Eint0 = quokka::EOS<Channel>::ComputeEintFromTgas(rho0, Tgas0);
	::P_outflow = quokka::EOS<Channel>::ComputePressure(rho0, Eint0);
	amrex::Print() << "Derived outflow pressure is " << ::P_outflow << " erg/cc.\n";

	// Set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// extract slice
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.geom[0], 0, 0., true);
	int const nx = static_cast<int>(position.size());
	std::vector<double> const xs = position;
	std::vector<double> xs_exact = position;

	// extract solution
	std::vector<double> d(nx);
	std::vector<double> vx(nx);
	std::vector<double> P(nx);
	std::vector<double> s(nx);
	std::vector<double> density_exact(nx);
	std::vector<double> velocity_exact(nx);
	std::vector<double> Pexact(nx);
	std::vector<double> sexact(nx);

	for (int i = 0; i < nx; ++i) {
		{
			amrex::Real const rho = values.at(HydroSystem<Channel>::density_index)[i];
			amrex::Real const xmom = values.at(HydroSystem<Channel>::x1Momentum_index)[i];
			amrex::Real const Egas = values.at(HydroSystem<Channel>::energy_index)[i];
			amrex::Real const scalar = values.at(HydroSystem<Channel>::scalar0_index)[i];
			amrex::Real const Eint = Egas - (xmom * xmom) / (2.0 * rho);
			amrex::Real const gamma = quokka::EOS_Traits<Channel>::gamma;
			d.at(i) = rho;
			vx.at(i) = xmom / rho;
			P.at(i) = ((gamma - 1.0) * Eint);
			s.at(i) = scalar;
		}
		{
			density_exact.at(i) = rho0;
			velocity_exact.at(i) = u_inflow;
			Pexact.at(i) = P_outflow;
			sexact.at(i) = s_inflow[0];
		}
	}
	std::vector<std::vector<double>> const sol{d, vx, P, s};
	std::vector<std::vector<double>> const sol_exact{density_exact, velocity_exact, Pexact, sexact};

	// compute error norm
	amrex::Real err_sq = 0.;
	for (size_t n = 0; n < sol.size(); ++n) {
		amrex::Real dU_k = 0.;
		amrex::Real U_k = 0;
		for (int i = 0; i < nx; ++i) {
			// Δ Uk = ∑i |Uk,in - Uk,i0| / Nx
			const amrex::Real U_k0 = sol_exact.at(n)[i];
			const amrex::Real U_k1 = sol.at(n)[i];
			dU_k += std::abs(U_k1 - U_k0) / static_cast<double>(nx);
			U_k += std::abs(U_k0) / static_cast<double>(nx);
		}
		amrex::Print() << "dU_" << n << " = " << dU_k << " U_k = " << U_k << "\n";
		// ε = || Δ U / U || = [&sum_k (ΔU_k/U_k)^2]^{1/2}
		err_sq += std::pow(dU_k / U_k, 2);
	}
	const amrex::Real epsilon = std::sqrt(err_sq);
	amrex::Print() << "rms of component-wise relative L1 error norms = " << epsilon << "\n\n";

#ifdef HAVE_PYTHON
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// Plot results
		const int skip = 4;	  // only plot every 8 elements of exact solution
		const double msize = 5.0; // marker size
		using mpl_arg = std::map<std::string, std::string>;
		using mpl_sarg = std::unordered_map<std::string, std::string>;

		matplotlibcpp::clf();
		mpl_arg d_args;
		mpl_sarg dexact_args;
		d_args["label"] = "density";
		d_args["color"] = "C0";
		dexact_args["marker"] = "o";
		dexact_args["color"] = "C0";
		matplotlibcpp::plot(xs, d, d_args);
		matplotlibcpp::scatter(strided_vector_from(xs_exact, skip), strided_vector_from(density_exact, skip), msize, dexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./channel_flow_density.pdf");

		matplotlibcpp::clf();
		std::map<std::string, std::string> vx_args;
		vx_args["label"] = "velocity";
		vx_args["color"] = "C3";
		matplotlibcpp::plot(xs, vx, vx_args);
		mpl_sarg vexact_args;
		vexact_args["marker"] = "o";
		vexact_args["color"] = "C3";
		matplotlibcpp::scatter(strided_vector_from(xs_exact, skip), strided_vector_from(velocity_exact, skip), msize, vexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./channel_flow_velocity.pdf");

		matplotlibcpp::clf();
		std::map<std::string, std::string> P_args;
		P_args["label"] = "pressure";
		P_args["color"] = "C4";
		matplotlibcpp::plot(xs, P, P_args);
		mpl_sarg Pexact_args;
		Pexact_args["marker"] = "o";
		Pexact_args["color"] = "C4";
		matplotlibcpp::scatter(strided_vector_from(xs_exact, skip), strided_vector_from(Pexact, skip), msize, Pexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./channel_flow_pressure.pdf");

		matplotlibcpp::clf();
		std::map<std::string, std::string> s_args;
		s_args["label"] = "passive scalar";
		s_args["color"] = "C4";
		matplotlibcpp::plot(xs, s, s_args);
		mpl_sarg sexact_args;
		sexact_args["marker"] = "o";
		sexact_args["color"] = "C4";
		matplotlibcpp::scatter(strided_vector_from(xs_exact, skip), strided_vector_from(sexact, skip), msize, sexact_args);
		matplotlibcpp::legend();
		matplotlibcpp::xlabel("length x");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./channel_flow_scalar.pdf");
	}
#endif

	// Compute test success condition
	int status = 0;
	const double error_tol = 3.5e-5;
	if (epsilon > error_tol) {
		status = 1;
	}
	return status;
}
