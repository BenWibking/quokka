#ifndef CLOUDYCOOLING_HPP_ // NOLINT
#define CLOUDYCOOLING_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file CloudyCooling.hpp
/// \brief Defines methods for interpolating cooling rates from Cloudy tables.
///

#include "AMReX.H"
#include "AMReX_Algorithm.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_iMultiFab.H"

#include "GrackleDataReader.hpp"
#include "fmt/core.h"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/FastMath.hpp"
#include "math/Interpolate2D.hpp"
#include "math/ODEIntegrate.hpp"
#include "math/root_finding.hpp"
#include "radiation/radiation_system.hpp"

namespace quokka::GrackleLikeCooling
{
// From Grackle source code (initialize_chemistry_data.c, line 114):
//   In fully tabulated mode, set H mass fraction according to
//   the abundances in Cloudy, which assumes n_He / n_H = 0.1.
//   This gives a value of about 0.716. Using the default value
//   of 0.76 will result in negative electron densities at low
//   temperature. Below, we set X = 1 / (1 + hydrogen_mass_cgs_e * n_He / n_H).

constexpr double cloudy_H_mass_fraction = 1. / (1. + 0.1 * 3.971);
constexpr double X = cloudy_H_mass_fraction;
constexpr double Z = 0.02; // metal fraction by mass
constexpr double Y = 1. - X - Z;
constexpr double mean_metals_A = 16.; // mean atomic weight of metals

constexpr double sigma_T = 6.6524e-25;		    // Thomson cross section (cm^2)
constexpr double electron_mass_cgs = 9.1093897e-28; // electron mass (g)
constexpr double T_cmb = 2.725;			    // * (1 + z); // K
constexpr double E_cmb = radiation_constant_cgs_ * (T_cmb * T_cmb * T_cmb * T_cmb);

struct grackleGpuConstTables {
	// these are non-owning, so can make a copy of the whole struct
	amrex::Table1D<const Real> log_nH;
	amrex::Table1D<const Real> log_Tgas;

	amrex::Table2D<const Real> primCool;
	amrex::Table2D<const Real> primHeat;
	amrex::Table2D<const Real> metalCool;
	amrex::Table2D<const Real> metalHeat;
	amrex::Table2D<const Real> meanMolWeight;

	// temperature range
	amrex::Real T_min;
	amrex::Real T_max;

	// mean molecular weight range
	amrex::Real mmw_min;
	amrex::Real mmw_max;
};

class grackle_tables
{
      public:
	std::unique_ptr<amrex::TableData<double, 1>> log_nH;
	std::unique_ptr<amrex::TableData<double, 1>> log_Tgas;

	std::unique_ptr<amrex::TableData<double, 2>> primCooling;
	std::unique_ptr<amrex::TableData<double, 2>> primHeating;
	std::unique_ptr<amrex::TableData<double, 2>> metalCooling;
	std::unique_ptr<amrex::TableData<double, 2>> metalHeating;
	std::unique_ptr<amrex::TableData<double, 2>> mean_mol_weight;

	amrex::Real T_min;
	amrex::Real T_max;
	amrex::Real mmw_min;
	amrex::Real mmw_max;

	[[nodiscard]] auto const_tables() const -> grackleGpuConstTables;
};

struct ODEUserData {
	Real rho{};
	Real gamma{};
	grackleGpuConstTables tables;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto cloudy_cooling_function(Real const rho, Real const T, grackleGpuConstTables const &tables) -> Real
{
	// interpolate cooling rates from Cloudy tables
	const Real rhoH = rho * cloudy_H_mass_fraction; // mass density of H species
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);
	const Real log_T = std::log10(T);

	const double logPrimCool = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.primCool);
	const double logPrimHeat = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.primHeat);
	const double logMetalCool = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.metalCool);
	const double logMetalHeat = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.metalHeat);

	const double netLambda_prim = FastMath::pow10(logPrimHeat) - FastMath::pow10(logPrimCool);
	const double netLambda_metals = FastMath::pow10(logMetalHeat) - FastMath::pow10(logMetalCool);
	const double netLambda = netLambda_prim + netLambda_metals;

	// multiply by the square of H mass density (**NOT number density**)
	double Edot = (rhoH * rhoH) * netLambda;

	// compute dimensionless mean mol. weight mu from mu(T) table
	const double mu = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.meanMolWeight);

	// compute electron density
	// N.B. it is absolutely critical to include the metal contribution here!
	double n_e = (rho / (C::m_p + C::m_e)) * (1.0 - mu * (X + Y / 4. + Z / mean_metals_A)) / (mu - (electron_mass_cgs / (C::m_p + C::m_e)));
	// the approximation for the metals contribution to e- fails at high densities (~1e3 or higher)
	n_e = std::max(n_e, 1.0e-4 * nH);

	// photoelectric heating term
	const double Tsqrt = std::sqrt(T);
	constexpr double phi = 0.5; // phi_PAH from Wolfire et al. (2003)
	constexpr double G_0 = 1.7; // ISRF from Wolfire et al. (2003)
	const double epsilon = 4.9e-2 / (1. + 4.0e-3 * std::pow(G_0 * Tsqrt / (n_e * phi), 0.73)) +
			       3.7e-2 * std::pow(T / 1.0e4, 0.7) / (1. + 2.0e-4 * (G_0 * Tsqrt / (n_e * phi)));
	const double Gamma_pe = 1.3e-24 * nH * epsilon * G_0;
	Edot += Gamma_pe;

	// Compton term (CMB photons)
	// [e.g., Hirata 2018: doi:10.1093/mnras/stx2854]
	constexpr double Gamma_C = (8. * sigma_T * E_cmb) / (3. * electron_mass_cgs * c_light_cgs_);
	constexpr double C_n = Gamma_C * C::k_B / (5. / 3. - 1.0);
	const double compton_CMB = -C_n * (T - T_cmb) * n_e;
	Edot += compton_CMB;

	return Edot;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeEgasFromTgas(double rho, double Tgas, double gamma, grackleGpuConstTables const &tables) -> Real
{
	// convert Egas (internal gas energy) to temperature
	const Real rhoH = rho * cloudy_H_mass_fraction;
	const Real nH = rhoH / (C::m_p + C::m_e);

	// compute mu from mu(T) table
	const Real mu = interpolate2d(std::log10(nH), std::log10(Tgas), tables.log_nH, tables.log_Tgas, tables.meanMolWeight);

	// compute thermal gas energy
	const Real Egas = (rho / ((C::m_p + C::m_e) * mu)) * C::k_B * Tgas / (gamma - 1.);
	return Egas;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto ComputeTgasFromEgas(double rho, double Egas, double gamma, grackleGpuConstTables const &tables) -> Real
{
	// convert Egas (internal gas energy) to temperature

	// check whether temperature is out-of-bounds
	const Real Tmin_table = tables.T_min;
	const Real Tmax_table = tables.T_max;
	const Real Eint_min = ComputeEgasFromTgas(rho, Tmin_table, gamma, tables);
	const Real Eint_max = ComputeEgasFromTgas(rho, Tmax_table, gamma, tables);

	if (Egas <= Eint_min) {
		return Tmin_table;
	}
	if (Egas >= Eint_max) {
		return Tmax_table;
	}

	// solve for temperature given Eint (with fixed adiabatic index gamma)
	const Real rhoH = rho * cloudy_H_mass_fraction;
	const Real nH = rhoH / (C::m_p + C::m_e);
	const Real log_nH = std::log10(nH);

	// mean molecular weight (in Grackle tables) is defined w/r/t
	// hydrogen_mass_cgs_
	const Real C = (gamma - 1.) * Egas / (C::k_B * (rho / (C::m_p + C::m_e)));

	// solve for mu(T)*C == T.
	// (Grackle does this with a fixed-point iteration. We use a more robust
	// method, similar to Brent's method, the TOMS748 method.)
	const Real reltol = 1.0e-5;
	const int maxIterLimit = 100;
	int maxIter = maxIterLimit;

	const Real log_Tmin = std::log10(Tmin_table);
	const Real log_Tmax = std::log10(Tmax_table);
	auto f = [log_nH, C, tables, log_Tmin, log_Tmax](const Real &T) noexcept {
		// compute new mu from mu(log10 T) table
		const Real log_T = clamp(std::log10(T), log_Tmin, log_Tmax);
		Real mu = interpolate2d(log_nH, log_T, tables.log_nH, tables.log_Tgas, tables.meanMolWeight);
		Real fun = C * mu - T;
		return fun;
	};

	// compute temperature bounds using physics
	const Real mu_min = tables.mmw_min;
	const Real mu_max = tables.mmw_max;
	const Real T_min = amrex::Clamp(C * mu_min, Tmin_table, Tmax_table);
	const Real T_max = amrex::Clamp(C * mu_max, Tmin_table, Tmax_table);

	// do root-finding
	quokka::math::eps_tolerance<Real> tol(reltol);
	auto bounds = quokka::math::toms748_solve(f, T_min, T_max, tol, maxIter);
	Real T_sol = 0.5 * (bounds.first + bounds.second);

	if ((maxIter >= maxIterLimit) || std::isnan(T_sol)) {
		printf("\nTgas iteration failed! rho = %.17g, Eint = %.17g, nH = %e, Tgas "
		       "= %e, "
		       "bounds.first = %e, bounds.second = %e, T_min = %e, T_max = %e, "
		       "maxIter = %d\n",
		       rho, Egas, nH, T_sol, bounds.first, bounds.second, T_min, T_max, maxIter);
		T_sol = NAN;
	}

	return T_sol;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto user_rhs(Real /*t*/, quokka::valarray<Real, 1> &y_data, quokka::valarray<Real, 1> &y_rhs, void *user_data) -> int
{
	// unpack user_data
	auto *udata = static_cast<ODEUserData *>(user_data);
	const Real rho = udata->rho;
	const Real gamma = udata->gamma;
	grackleGpuConstTables const &tables = udata->tables;

	// check whether temperature is out-of-bounds
	const Real Tmin = tables.T_min;
	const Real Tmax = tables.T_max;
	const Real Eint_min = ComputeEgasFromTgas(rho, Tmin, gamma, tables);
	const Real Eint_max = ComputeEgasFromTgas(rho, Tmax, gamma, tables);

	// compute temperature and cooling rate
	const Real Eint = y_data[0];

	if (Eint <= Eint_min) {
		// set cooling to value at Tmin
		y_rhs[0] = cloudy_cooling_function(rho, Tmin, tables);
	} else if (Eint >= Eint_max) {
		// set cooling to value at Tmax
		y_rhs[0] = cloudy_cooling_function(rho, Tmax, tables);
	} else {
		// ok, within tabulated cooling limits
		const Real T = ComputeTgasFromEgas(rho, Eint, gamma, tables);
		if (!std::isnan(T)) { // temp iteration succeeded
			y_rhs[0] = cloudy_cooling_function(rho, T, tables);
		} else { // temp iteration failed
			y_rhs[0] = NAN;
			return 1; // failed
		}
	}

	return 0; // success
}

template <typename problem_t> auto computeCooling(amrex::MultiFab &mf, const Real dt_in, grackle_tables &cloudyTables, const Real T_floor) -> bool
{
	BL_PROFILE("computeCooling()")

	const Real dt = dt_in;
	const Real reltol_floor = 0.01;
	const Real rtol = 1.0e-4; // not recommended to change this

	auto tables = cloudyTables.const_tables();

	const auto &ba = mf.boxArray();
	const auto &dmap = mf.DistributionMap();
	amrex::iMultiFab nsubstepsMF(ba, dmap, 1, 0);

	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);
		auto const &nsubsteps = nsubstepsMF.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const Real x1Mom = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const Real x2Mom = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const Real x3Mom = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const Real Egas = state(i, j, k, HydroSystem<problem_t>::energy_index);

			const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, x1Mom, x2Mom, x3Mom, Egas);
			const Real gamma = quokka::EOS_Traits<problem_t>::gamma;
			ODEUserData user_data{rho, gamma, tables};
			quokka::valarray<Real, 1> y = {Eint};
			quokka::valarray<Real, 1> const abstol = {reltol_floor * ComputeEgasFromTgas(rho, T_floor, gamma, tables)};

			// do integration with RK2 (Heun's method)
			int nsteps = 0;
			rk_adaptive_integrate(user_rhs, 0, y, dt, &user_data, rtol, abstol, nsteps);
			nsubsteps(i, j, k) = nsteps;

			// check if integration failed
			if (nsteps >= maxStepsODEIntegrate) {
				Real const T = ComputeTgasFromEgas(rho, Eint, quokka::EOS_Traits<problem_t>::gamma, tables);
				Real const Edot = cloudy_cooling_function(rho, T, tables);
				Real const t_cool = Eint / Edot;
				printf("max substeps exceeded! rho = %.17e, Eint = %.17e, T = %g, cooling "
				       "time = %g, dt = %.17e\n",
				       rho, Eint, T, t_cool, dt);
			}
			const Real Eint_new = y[0];
			const Real dEint = Eint_new - Eint;

			state(i, j, k, HydroSystem<problem_t>::energy_index) += dEint;
			state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) += dEint;
		});
	}

	int nmax = nsubstepsMF.max(0);
	Real navg = static_cast<Real>(nsubstepsMF.sum(0)) / static_cast<Real>(nsubstepsMF.boxArray().numPts());
	amrex::Print() << fmt::format("\tcooling substeps (per cell): avg {}, max {}\n", navg, nmax);

	// check if integration succeeded
	if (nmax >= maxStepsODEIntegrate) {
		amrex::Print() << "\t[GrackleLikeCooling] Reaction ODE failure! Retrying hydro update...\n";
		return false;
	}
	return true; // success
}

void readGrackleData(std::string &grackle_hdf5_file, grackle_tables &cloudyTables);

} // namespace quokka::GrackleLikeCooling

#endif // CLOUDYCOOLING_HPP_
