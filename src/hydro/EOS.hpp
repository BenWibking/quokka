#ifndef EOS_HPP_
#define EOS_HPP_
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file EOS.hpp
/// \brief A class for equation of state calculations.

#include <cmath>
#include <optional>
#include <tuple>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"
#include "physics_info.hpp"
#include <AMReX_Print.H>

#include "eos.H"

#ifdef CHEMISTRY
#include "actual_eos_data.H"
#endif

namespace quokka
{

// specify default values for ideal gamma-law EOS
//
template <typename problem_t> struct EOS_Traits {
	static constexpr double gamma = 5. / 3.;     // default value
	static constexpr double cs_isothermal = NAN; // only used when gamma = 1
	static constexpr double mean_molecular_weight = NAN;
	static constexpr double boltzmann_constant = C::k_B;
};

template <typename problem_t> class EOS
{
      private:
	static constexpr amrex::Real gamma_ = EOS_Traits<problem_t>::gamma;
	static constexpr amrex::Real mean_molecular_weight_ = EOS_Traits<problem_t>::mean_molecular_weight;

      public:
	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real;

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real;

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real;

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeEintTempDerivative(amrex::Real rho, amrex::Real Tgas, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real;

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeOtherDerivatives(amrex::Real rho, amrex::Real P, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {});

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputePressure(amrex::Real rho, amrex::Real Eint, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {}) -> amrex::Real;

	[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static auto
	ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars = {})
	    -> amrex::Real;

	static constexpr amrex::Real boltzmann_constant_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return C::k_B;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::boltzmann_constant;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// k_B / k_B_bar = u_l^2 * u_m / u_t^2 / u_T
			return C::k_B /
			       (Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_mass /
				(Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time) / Physics_Traits<problem_t>::unit_temperature);
		}
	}();
};

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeTgasFromEint(amrex::Real rho, amrex::Real Eint,
										  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
    -> amrex::Real
{
	// return temperature for an ideal gas given density and internal energy
	amrex::Real Tgas = NAN;

	const amrex::Real boltzmann_constant_g = boltzmann_constant_;

#ifdef CHEMISTRY
	eos_t chemstate;
	chemstate.rho = rho;
	chemstate.e = Eint / rho;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_re, chemstate);
	Tgas = chemstate.T;
#else
	amrex::ignore_unused(massScalars);

	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.e = Eint / rho;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_re, estate);
		// scale returned temperature in case boltzmann constant is dimensionless
		Tgas = estate.T * C::k_B / boltzmann_constant_g;
	}
#endif
	return Tgas;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromTgas(amrex::Real rho, amrex::Real Tgas,
										  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
    -> amrex::Real
{
	// return internal energy density given density and temperature
	amrex::Real Eint = NAN;

#ifdef CHEMISTRY
	eos_t chemstate;
	chemstate.rho = rho;
	// Define and initialize Tgas here
	amrex::Real const Tgas_value = Tgas;
	chemstate.T = Tgas_value;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_rt, chemstate);
	Eint = chemstate.e * chemstate.rho;
#else
	amrex::ignore_unused(massScalars);

	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.T = Tgas;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_rt, estate);
		Eint = estate.e * rho * boltzmann_constant_ / C::k_B;
	}
#endif
	return Eint;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure,
										  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
    -> amrex::Real
{
	// return internal energy density given density and pressure
	amrex::Real Eint = NAN;

#ifdef CHEMISTRY
	eos_t chemstate;
	chemstate.rho = rho;
	chemstate.p = Pressure;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_rp, chemstate);
	Eint = chemstate.e * chemstate.rho;
#else
	amrex::ignore_unused(massScalars);

	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.p = Pressure;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_rp, estate);
		Eint = estate.e * rho;
	}
#endif
	return Eint;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto
EOS<problem_t>::ComputeEintTempDerivative(const amrex::Real rho, const amrex::Real Tgas,
					  std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars) -> amrex::Real
{
	// compute derivative of internal energy w/r/t temperature, given density and temperature
	amrex::Real dEint_dT = NAN;

#ifdef CHEMISTRY
	eos_t chemstate;
	chemstate.rho = rho;
	// we don't need Tgas to find chemstate.dedT, but we still need to initialize chemstate.T because we are using the 'rt' EOS mode
	chemstate.T = NAN;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_rt, chemstate);
	dEint_dT = chemstate.dedT * chemstate.rho;
#else
	amrex::ignore_unused(massScalars);

	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.T = Tgas;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_rt, estate);
		dEint_dT = estate.dedT * rho * boltzmann_constant_ / C::k_B;
	}
#endif
	return dEint_dT;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto
EOS<problem_t>::ComputeOtherDerivatives(const amrex::Real rho, const amrex::Real P, std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
{
	// compute derivative of specific internal energy w/r/t density, given density and pressure
	amrex::Real deint_dRho = NAN;
	// compute derivative of specific internal energy w/r/t density, given density and pressure
	amrex::Real deint_dP = NAN;
	// compute derivative of density w/r/t pressure, given density and pressure
	amrex::Real dRho_dP = NAN;
	// compute derivative of pressure w/r/t density at constant entropy, given density and pressure (needed for the fundamental derivative G)
	amrex::Real dP_dRho_s = NAN;
	// fundamental derivative
	amrex::Real G = NAN;

#ifdef CHEMISTRY
	eos_t chemstate;
	chemstate.rho = rho;
	chemstate.p = P;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_rp, chemstate);
	deint_dRho = chemstate.dedr;
	deint_dP = 1.0 / chemstate.dpde;
	dRho_dP = 1.0 / (chemstate.dpdr * C::k_B / boltzmann_constant_);
	dP_dRho_s = chemstate.cs * chemstate.cs;
	G = chemstate.G;

#else
	amrex::ignore_unused(massScalars);

	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.p = P;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_rp, estate);
		deint_dRho = estate.dedr;
		deint_dP = 1.0 / estate.dpde;
		dRho_dP = 1.0 / (estate.dpdr * C::k_B / boltzmann_constant_);
		dP_dRho_s = estate.cs * estate.cs;
		G = estate.G;
	}
#endif
	return std::make_tuple(deint_dRho, deint_dP, dRho_dP, dP_dRho_s, G);
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputePressure(amrex::Real rho, amrex::Real Eint,
									      std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
    -> amrex::Real
{
	// return pressure for an ideal gas
	amrex::Real P = NAN;
#ifdef CHEMISTRY
	eos_t chemstate;
	chemstate.rho = rho;
	chemstate.e = Eint / rho;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_re, chemstate);
	P = chemstate.p;
#else
	amrex::ignore_unused(massScalars);

	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		// if rho is 0, pass 0 to state.e
		if (rho == 0.0) {
			estate.e = 0;
		} else {
			estate.e = Eint / rho;
		}
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_re, estate);
		P = estate.p;
	}
#endif
	return P;
}

template <typename problem_t>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto EOS<problem_t>::ComputeSoundSpeed(amrex::Real rho, amrex::Real Pressure,
										std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> const &massScalars)
    -> amrex::Real
{
	// return sound speed for an ideal gas
	amrex::Real cs = NAN;

#ifdef CHEMISTRY
	eos_t chemstate;
	chemstate.rho = rho;
	chemstate.p = Pressure;
	// initialize array of number densities
	for (int ii = 0; ii < NumSpec; ++ii) {
		chemstate.xn[ii] = -1.0;
	}

	if (massScalars) {
		const auto &massArray = *massScalars;
		for (int nn = 0; nn < nmscalars_; ++nn) {
			chemstate.xn[nn] = massArray[nn] / spmasses[nn]; // massScalars are partial densities (massFractions * rho)
		}
	}

	eos(eos_input_rp, chemstate);
	cs = chemstate.cs;
#else
	amrex::ignore_unused(massScalars);

	if constexpr (gamma_ != 1.0) {
		chem_eos_t estate;
		estate.rho = rho;
		estate.p = Pressure;
		estate.mu = mean_molecular_weight_ / C::m_u;
		eos(eos_input_rp, estate);
		cs = estate.cs;
	}
#endif
	return cs;
}

} // namespace quokka

#endif // EOS_HPP_
