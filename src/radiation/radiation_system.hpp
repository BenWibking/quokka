#ifndef RADIATION_SYSTEM_HPP_ // NOLINT
#define RADIATION_SYSTEM_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file radiation_system.hpp
/// \brief Defines a class for solving the (1d) radiation moment equations.
///

// c++ headers

#include <array>
#include <cmath>

// library headers
#include "AMReX.H" // IWYU pragma: keep
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

// internal headers
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hyperbolic_system.hpp"
#include "math/math_impl.hpp"
#include "physics_info.hpp"
#include "radiation/planck_integral.hpp"
#include "util/valarray.hpp"

using Real = amrex::Real;

static constexpr bool print_chat = true;

// Hyper parameters for the radiation solver
static constexpr bool add_line_cooling_to_radiation_in_jac = false;
static constexpr bool include_delta_B = true;
static constexpr bool use_diffuse_flux_mean_opacity = true;
static constexpr bool special_edge_bin_slopes = false;	    // Use 2 and -4 as the slopes for the first and last bins, respectively
static constexpr bool force_rad_floor_in_iteration = false; // force radiation energy density to be positive (and above the floor value) in the Newton iteration
static constexpr bool include_work_term_in_source = true;

static const int max_iter_to_update_alpha_E = 5; // Apply to the PPL_opacity_full_spectrum only. Only update alpha_E for the first max_iter_to_update_alpha_E
// iterations of the Newton iteration
static constexpr bool enable_dE_constrain = true;
static constexpr bool use_D_as_base = false;
static const bool PPL_free_slope_st_total = false; // PPL with free slopes for all, but subject to the constraint sum_g alpha_g B_g = - sum_g B_g. Not working
						   // well -- Newton iteration convergence issue.

// Time integration scheme
// IMEX PD-ARS
static constexpr double IMEX_a22 = 1.0;
static constexpr double IMEX_a32 = 0.5; // 0 < IMEX_a32 <= 0.5
// SSP-RK2 + implicit radiation-matter exchange
// static constexpr double IMEX_a22 = 0.0;
// static constexpr double IMEX_a32 = 0.0;

// physical constants in CGS units
static constexpr double c_light_cgs_ = C::c_light;	    // cgs
static constexpr double radiation_constant_cgs_ = C::a_rad; // cgs
static constexpr double inf = std::numeric_limits<double>::max();

// enum for opacity_model
enum class OpacityModel {
	single_group = 0, // user-defined opacity for each group, given as a function of density and temperature.
	piecewise_constant_opacity,
	PPL_opacity_fixed_slope_spectrum,
	PPL_opacity_full_spectrum // piecewise power-law opacity model with piecewise power-law fitting to a user-defined opacity function and on-the-fly
				  // piecewise power-law fitting to radiation energy density and flux.
};

// this struct is specialized by the user application code
//
template <typename problem_t> struct RadSystem_Traits {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = 0.;
	static constexpr double energy_unit = C::ev2erg;
	static constexpr amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups + 1> radBoundaries = {0., inf};
	static constexpr double beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::single_group;
};

// this struct is specialized by the user application code
//
template <typename problem_t> struct ISM_Traits {
	static constexpr bool enable_dust_gas_thermal_coupling_model = false;
	static constexpr bool enable_photoelectric_heating = false;
	static constexpr double gas_dust_coupling_threshold = 1.0e-6;
};

// A struct to hold the results of the ComputeRadPressure function.
struct RadPressureResult {
	quokka::valarray<double, 4> F; // components of radiation pressure tensor
	double S;		       // maximum wavespeed for the radiation system
};

// A struct to hold the opacity terms for the radiation-matter energy exchange, containing the following elements:
// kappaE, kappaP, kappaF, kappaPoverE, delta_nu_kappa_B_at_edge, alpha_P, alpha_E
template <typename problem_t> struct OpacityTerms {
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> kappaE;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> kappaP;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> kappaF;
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> kappaPoverE;
	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> delta_nu_kappa_B_at_edge; // Delta (nu * kappa * B)
	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> alpha_P;
	amrex::GpuArray<double, Physics_Traits<problem_t>::nGroups> alpha_E;
};

// A struct to hold the results of the Newton-Raphson iteration for energy update, containing the following elements:
// Egas, T_gas, T_d, EradVec, work, opacity_terms
template <typename problem_t> struct NewtonIterationResult {
	double Egas;							      // gas internal energy
	double T_gas;							      // gas temperature
	double T_d;							      // dust temperature
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> EradVec; // radiation energy density
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> work;    // work term
	OpacityTerms<problem_t> opacity_terms;
};

// A struct to hold the results of ComputeJacobian functions, containing the following elements:
// J00, F0, Fg_abs_sum, J0g, Jg0, Jgg, Fg
template <typename problem_t> struct JacobianResult {
	double J00;	   // (0, 0) component of the Jacobian matrix
	double F0;	   // (0) component of the residual
	double Fg_abs_sum; // sum of the absolute values of the (g) components of the residual, g = 1, 2, ..., nGroups, and tau(g) > 0
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> J0g; // (0, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Jg0; // (g, 0) components of the Jacobian matrix, g = 1, 2, ..., nGroups
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Jgg; // (g, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Jg1; // (g, 1) components of the Jacobian matrix, g = 1, 2, ..., nGroups
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Fg;  // (g) components of the residual, g = 1, 2, ..., nGroups
};

// A struct to hold the results of UpdateFlux(), containing the following elements:
// Erad, gasMomentum, Frad
template <typename problem_t> struct FluxUpdateResult {
	quokka::valarray<double, Physics_Traits<problem_t>::nGroups> Erad;			   // radiation energy density
	amrex::GpuArray<double, 3> gasMomentum;							   // gas momentum
	amrex::GpuArray<amrex::GpuArray<amrex::Real, Physics_Traits<problem_t>::nGroups>, 3> Frad; // radiation flux
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto minmod_func(double a, double b) -> double
{
	return 0.5 * (sgn(a) + sgn(b)) * std::min(std::abs(a), std::abs(b));
}

// Use SFINAE (Substitution Failure Is Not An Error) to check if opacity_model is defined in RadSystem_Traits<problem_t>
template <typename problem_t, typename = void> struct RadSystem_Has_Opacity_Model : std::false_type {
};

template <typename problem_t>
struct RadSystem_Has_Opacity_Model<problem_t, std::void_t<decltype(RadSystem_Traits<problem_t>::opacity_model)>> : std::true_type {
};

/// Class for the radiation moment equations
///
template <typename problem_t> class RadSystem : public HyperbolicSystem<problem_t>
{
      public:
	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto MC(double a, double b) -> double
	{
		return 0.5 * (sgn(a) + sgn(b)) * std::min(0.5 * std::abs(a + b), std::min(2.0 * std::abs(a), 2.0 * std::abs(b)));
	}

	static constexpr int nmscalars_ = Physics_Traits<problem_t>::numMassScalars;
	static constexpr int numRadVars_ = Physics_NumVars::numRadVars;				 // number of radiation variables for each photon group
	static constexpr int nvarHyperbolic_ = numRadVars_ * Physics_Traits<problem_t>::nGroups; // total number of radiation variables
	static constexpr int nstartHyperbolic_ = Physics_Indices<problem_t>::radFirstIndex;
	static constexpr int nvar_ = nstartHyperbolic_ + nvarHyperbolic_;

	enum gasVarIndex {
		gasDensity_index = Physics_Indices<problem_t>::hydroFirstIndex,
		x1GasMomentum_index,
		x2GasMomentum_index,
		x3GasMomentum_index,
		gasEnergy_index,
		gasInternalEnergy_index,
		scalar0_index
	};

	enum radVarIndex { radEnergy_index = nstartHyperbolic_, x1RadFlux_index, x2RadFlux_index, x3RadFlux_index };

	enum primVarIndex {
		primRadEnergy_index = 0,
		x1ReducedFlux_index,
		x2ReducedFlux_index,
		x3ReducedFlux_index,
	};

	// C++ standard does not allow constexpr to be uninitialized, even in a
	// templated class!

	static constexpr amrex::Real c_light_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return c_light_cgs_;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::c_light;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// c / c_bar = u_l / u_t
			return c_light_cgs_ / (Physics_Traits<problem_t>::unit_length / Physics_Traits<problem_t>::unit_time);
		}
	}();
	static constexpr double chat0_over_c = RadSystem_Traits<problem_t>::c_hat_over_c;
	static constexpr double chat0_ = c_light_ * chat0_over_c;

	static constexpr double radiation_constant_ = []() constexpr {
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			return C::a_rad;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
			return Physics_Traits<problem_t>::radiation_constant;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// a_rad / a_rad_bar = 1 / u_l * u_m / u_t^2 / u_T^4
			return C::a_rad / (1.0 / Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_mass /
					   (Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time) /
					   (Physics_Traits<problem_t>::unit_temperature * Physics_Traits<problem_t>::unit_temperature *
					    Physics_Traits<problem_t>::unit_temperature * Physics_Traits<problem_t>::unit_temperature));
		}
	}();

	static constexpr int beta_order_ = RadSystem_Traits<problem_t>::beta_order;

	static constexpr bool enable_dust_gas_thermal_coupling_model_ = ISM_Traits<problem_t>::enable_dust_gas_thermal_coupling_model;
	static constexpr bool enable_photoelectric_heating_ = ISM_Traits<problem_t>::enable_photoelectric_heating;

	static constexpr int nGroups_ = Physics_Traits<problem_t>::nGroups;
	static constexpr amrex::GpuArray<double, nGroups_ + 1> radBoundaries_ = []() constexpr {
		if constexpr (nGroups_ > 1) {
			return RadSystem_Traits<problem_t>::radBoundaries;
		} else {
			amrex::GpuArray<double, 2> boundaries{0., inf};
			return boundaries;
		}
	}();

	static constexpr double Erad_floor_ = RadSystem_Traits<problem_t>::Erad_floor / nGroups_;

	static constexpr OpacityModel opacity_model_ = []() constexpr {
		if constexpr (RadSystem_Has_Opacity_Model<problem_t>::value) {
			return RadSystem_Traits<problem_t>::opacity_model;
		} else {
			return OpacityModel::single_group;
		}
	}();

	// Assertion: has to use single_group when nGroups_ == 1
	static_assert(((nGroups_ > 1 && opacity_model_ != OpacityModel::single_group) || (nGroups_ == 1 && opacity_model_ == OpacityModel::single_group)),
		      "OpacityModel::single_group MUST be used when nGroups_ == 1. If nGroups_ > 1, you MUST set opacity_model."); // NOLINT

	// Assertion: PPL_opacity_full_spectrum requires at least 3 photon groups
	static_assert(!(nGroups_ < 3 && opacity_model_ == OpacityModel::PPL_opacity_full_spectrum), // NOLINT
		      "PPL_opacity_full_spectrum requires at least 3 photon groups.");

	static constexpr double mean_molecular_mass_ = quokka::EOS_Traits<problem_t>::mean_molecular_weight;
	static constexpr double gamma_ = quokka::EOS_Traits<problem_t>::gamma;

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

	// static functions

	static void ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const &cons, array_t &maxSignal, amrex::Box const &indexRange);
	static void ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange);

	static void PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
				amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);

	static void AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArrayOld,
				 amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxDiffusiveArray, double dt_in,
				 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, int nvars);

	template <FluxDir DIR>
	static void ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
				  amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in,
				  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool use_wavespeed_correction,
				  amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor_in);

	static void SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
				       amrex::Real time);

	AMREX_GPU_DEVICE static auto UpdateFlux(int i, int j, int k, arrayconst_t const &consPrev, NewtonIterationResult<problem_t> &energy, double dt,
						double gas_update_factor, double Ekin0, quokka::valarray<double, nGroups_> const &chat) -> FluxUpdateResult<problem_t>;

	static void AddSourceTermsMultiGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt, int stage,
					     double dustGasCoeff, amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor,
					     int *p_iteration_counter, int *p_iteration_failure_counter);

	static void AddSourceTermsSingleGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt, int stage,
					      double dustGasCoeff, amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor,
					      int *p_iteration_counter, int *p_iteration_failure_counter);

	static void balanceMatterRadiation(arrayconst_t &consPrev, array_t &consNew, amrex::Box const &indexRange);

	// Use an additionalr template for ComputeMassScalars as the Array type is not always the same
	template <typename ArrayType>
	AMREX_GPU_DEVICE static auto ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeEddingtonFactor(double f) -> double;

	AMREX_GPU_HOST_DEVICE static auto ComputeNumberDensityH(double rho, amrex::GpuArray<Real, nmscalars_> const &massScalars) -> double;

	// Used for single-group RHD only. Not used for multi-group RHD.
	AMREX_GPU_HOST_DEVICE static auto ComputePlanckOpacity(double rho, double Tgas) -> Real;
	AMREX_GPU_HOST_DEVICE static auto ComputeFluxMeanOpacity(double rho, double Tgas) -> Real;
	AMREX_GPU_HOST_DEVICE static auto ComputeEnergyMeanOpacity(double rho, double Tgas) -> Real;

	// For multi-group RHD, use DefineOpacityExponentsAndLowerValues to define the opacities.
	AMREX_GPU_HOST_DEVICE static auto DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double rho, double Tgas)
	    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>;

	AMREX_GPU_HOST_DEVICE static auto ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value,
								  amrex::GpuArray<double, nGroups_> const &radBoundaryRatios,
								  amrex::GpuArray<double, nGroups_> const &alpha_quant) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
								  amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value)
	    -> quokka::valarray<double, nGroups_>;
	// AMREX_GPU_HOST_DEVICE static auto
	// ComputeGroupMeanOpacityWithMinusOneSlope(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value,
	// 					 amrex::GpuArray<double, nGroups_> radBoundaryRatios) -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeEintFromEgas(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Etot) -> double;
	AMREX_GPU_HOST_DEVICE static auto ComputeEgasFromEint(double density, double X1GasMom, double X2GasMom, double X3GasMom, double Eint) -> double;
	AMREX_GPU_HOST_DEVICE static auto PlanckFunction(double nu, double T) -> double;
	AMREX_GPU_HOST_DEVICE static auto
	ComputeDiffusionFluxMeanOpacity(quokka::valarray<double, nGroups_> kappaPVec, quokka::valarray<double, nGroups_> kappaEVec,
					quokka::valarray<double, nGroups_> fourPiBoverC, amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge,
					amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge, amrex::GpuArray<double, nGroups_ + 1> kappa_slope)
	    -> quokka::valarray<double, nGroups_>;
	AMREX_GPU_HOST_DEVICE static auto ComputeFluxInDiffusionLimit(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, double T, double vel)
	    -> amrex::GpuArray<double, nGroups_>;

	template <typename ArrayType>
	AMREX_GPU_HOST_DEVICE static auto ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
	    -> amrex::GpuArray<double, nGroups_>;

	AMREX_GPU_HOST_DEVICE static void SolveLinearEqs(JacobianResult<problem_t> const &jacobian, double &x0, quokka::valarray<double, nGroups_> &xi);

	AMREX_GPU_HOST_DEVICE static void SolveLinearEqsWithLastColumn(JacobianResult<problem_t> const &jacobian, double &x0,
								       quokka::valarray<double, nGroups_> &xi);

	AMREX_GPU_HOST_DEVICE static auto Solve3x3matrix(double C00, double C01, double C02, double C10, double C11, double C12, double C20, double C21,
							 double C22, double Y0, double Y1, double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>;

	AMREX_GPU_HOST_DEVICE static auto ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries, amrex::Real temperature)
	    -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> double;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
	    -> quokka::valarray<amrex::Real, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real temperature) -> Real;

	AMREX_GPU_HOST_DEVICE static auto ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature,
											  amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
	    -> quokka::valarray<amrex::Real, nGroups_>;

	template <typename RHSFunction, typename JacFunction>
	AMREX_GPU_DEVICE static auto BackwardEulerOneVariable(RHSFunction const &rhs, JacFunction const &jac, double x0, double compare) -> double;

	AMREX_GPU_DEVICE static auto
	ComputeDustTemperatureBateKeto(double T_gas, double T_d_init, double rho, quokka::valarray<double, nGroups_> const &Erad, double N_d, double dt,
				       double R_sum, int n_step,
				       amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries = amrex::GpuArray<double, nGroups_ + 1>{}) -> double;

	AMREX_GPU_DEVICE static auto
	ComputeDustTemperatureGasOnly(double T_gas, double T_d_init, double rho, quokka::valarray<double, nGroups_> const &Erad, double N_d, double dt,
				      double R_sum, int n_step,
				      amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries = amrex::GpuArray<double, nGroups_ + 1>{},
				      amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios = amrex::GpuArray<double, nGroups_>{}) -> double;

	AMREX_GPU_HOST_DEVICE static auto DefinePhotoelectricHeatingE1Derivative(amrex::Real temperature, amrex::Real num_density) -> amrex::Real;

	AMREX_GPU_HOST_DEVICE static auto DefineBackgroundHeatingRate(amrex::Real num_density) -> amrex::Real;

	AMREX_GPU_HOST_DEVICE static auto DefineNetCoolingRate(amrex::Real temperature, amrex::Real num_density) -> quokka::valarray<double, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto DefineNetCoolingRateTempDerivative(amrex::Real temperature, amrex::Real num_density)
	    -> quokka::valarray<double, nGroups_>;

	AMREX_GPU_HOST_DEVICE static auto DefineCosmicRayHeatingRate(amrex::Real num_density) -> double;

	AMREX_GPU_DEVICE static void ComputeModelDependentKappaFAndDeltaTerms(double T, double rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
									      quokka::valarray<double, nGroups_> const &fourPiBoverC,
									      OpacityTerms<problem_t> &opacity_terms);

	AMREX_GPU_DEVICE static auto ComputeModelDependentKappaEAndKappaP(double T, double rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
									  amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
									  quokka::valarray<double, nGroups_> const &fourPiBoverC,
									  quokka::valarray<double, nGroups_> const &Erad, int n_iter,
									  amrex::GpuArray<double, nGroups_> const &alpha_E = {},
									  amrex::GpuArray<double, nGroups_> const &alpha_P = {}) -> OpacityTerms<problem_t>;

	AMREX_GPU_DEVICE static auto ComputeJacobianForGas(double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff,
							   quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src,
							   quokka::valarray<double, nGroups_> const &tau, double c_v,
							   quokka::valarray<double, nGroups_> const &kappaPoverE,
							   quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double num_den,
							   quokka::valarray<double, nGroups_> const &c_over_chat, double dt)
	    -> JacobianResult<problem_t>;

	AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDust(double T_gas, double T_d, double Egas_diff,
								  quokka::valarray<double, nGroups_> const &Erad_diff,
								  quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src,
								  double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v,
								  double lambda_gd_time_dt, quokka::valarray<double, nGroups_> const &kappaPoverE,
								  quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, quokka::valarray<double, nGroups_> const &chat_over_c, double num_den, double dt)
	    -> JacobianResult<problem_t>;

	AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDustDecoupled(
	    double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec,
	    quokka::valarray<double, nGroups_> const &Src, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double lambda_gd_time_dt,
	    quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, quokka::valarray<double, nGroups_> const &chat_over_c) -> JacobianResult<problem_t>;

	AMREX_GPU_DEVICE static auto ComputeJacobianForGasAndDustWithPE(
	    double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad, quokka::valarray<double, nGroups_> const &Erad0,
	    double PE_heating_energy_derivative, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double coeff_n,
	    quokka::valarray<double, nGroups_> const &tau, double c_v, double lambda_gd_time_dt, quokka::valarray<double, nGroups_> const &kappaPoverE,
	    quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double num_den, double dt, quokka::valarray<double, nGroups_> const &chat_over_c) -> JacobianResult<problem_t>;

	AMREX_GPU_DEVICE static auto
	SolveGasRadiationEnergyExchange(double Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double rho, double dt, quokka::valarray<double, nGroups_> const &c_over_chat,
					amrex::GpuArray<Real, nmscalars_> const &massScalars, int n_outer_iter, quokka::valarray<double, nGroups_> const &work,
					quokka::valarray<double, nGroups_> const &vel_times_F, quokka::valarray<double, nGroups_> const &Src,
					amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter, int *p_iteration_failure_counter)
	    -> NewtonIterationResult<problem_t>;

	AMREX_GPU_DEVICE static auto SolveGasDustRadiationEnergyExchange(double Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double rho,
									 double coeff_n, double dt, amrex::GpuArray<Real, nmscalars_> const &massScalars,
									 int n_outer_iter, quokka::valarray<double, nGroups_> const &work,
									 quokka::valarray<double, nGroups_> const &vel_times_F,
									 quokka::valarray<double, nGroups_> const &Src, quokka::valarray<double, nGroups_> const &chat,
									 amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter,
									 int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t>;

	AMREX_GPU_DEVICE static auto
	SolveGasDustRadiationEnergyExchangeWithPE(double Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double rho, double coeff_n, double dt,
						  amrex::GpuArray<Real, nmscalars_> const &massScalars, int n_outer_iter,
						  quokka::valarray<double, nGroups_> const &work, quokka::valarray<double, nGroups_> const &vel_times_F,
						  quokka::valarray<double, nGroups_> const &Src, quokka::valarray<double, nGroups_> const &chat,
						  amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter,
						  int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t>;

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar,
							     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k,
							     const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries)
	    -> quokka::valarray<double, nGroups_>;

	AMREX_GPU_DEVICE static auto ComputeCellOpticalDepthAllDirMin(const amrex::Array4<const amrex::Real> &consVar,
								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k,
								    const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries)
	    -> quokka::valarray<double, nGroups_>;

	AMREX_GPU_DEVICE static auto isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool;

	AMREX_GPU_DEVICE static void amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons);

	template <FluxDir DIR>
	AMREX_GPU_DEVICE static auto ComputeRadPressure(double erad_L, double Fx_L, double Fy_L, double Fz_L, double fx_L, double fy_L, double fz_L)
	    -> RadPressureResult;

	AMREX_GPU_DEVICE static auto ComputeEddingtonTensor(double fx_L, double fy_L, double fz_L) -> std::array<std::array<double, 3>, 3>;

	AMREX_GPU_DEVICE static void ComputeReducedSpeedOfLightFactor(arrayconst_t &consVar, double c_hat_over_c, array_t &reducedSpeedOfLightFactor, const amrex::Box &indexRange, const amrex::GpuArray<double, AMREX_SPACEDIM> &dx);
};

// Compute radiation energy fractions for each photon group from a Planck function, given nGroups, radBoundaries, and temperature
// This function enforces that the total fraction is 1.0, no matter what are the group boundaries
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckEnergyFractions(amrex::GpuArray<double, nGroups_ + 1> const &boundaries, amrex::Real temperature)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<amrex::Real, nGroups_> radEnergyFractions{};
	if constexpr (nGroups_ == 1) {
		radEnergyFractions[0] = 1.0;
		return radEnergyFractions;
	} else {
		amrex::Real const energy_unit_over_kT = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * temperature);
		amrex::Real y = NAN;
		amrex::Real previous = 0.0;
		for (int g = 0; g < nGroups_ - 1; ++g) {
			const amrex::Real x = boundaries[g + 1] * energy_unit_over_kT;
			if (x >= 100.) { // 100. is the upper limit of x in the table
				y = 1.0;
			} else {
				y = integrate_planck_from_0_to_x(x);
			}
			radEnergyFractions[g] = y - previous;
			previous = y;
		}
		// last group, enforcing the total fraction to be 1.0
		y = 1.0;
		radEnergyFractions[nGroups_ - 1] = y - previous;
		AMREX_ASSERT(std::abs(sum(radEnergyFractions) - 1.0) < 1.0e-10);

		return radEnergyFractions;
	}
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeNumberDensityH(double rho, amrex::GpuArray<Real, nmscalars_> const & /*massScalars*/) -> double
{
	return rho / mean_molecular_mass_;
}

// define ComputeThermalRadiation for single-group, returns the thermal radiation power = a_r * T^4
template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationSingleGroup(amrex::Real temperature) -> Real
{
	double power = radiation_constant_ * std::pow(temperature, 4);
	// set floor
	if (power < Erad_floor_) {
		power = Erad_floor_;
	}
	return power;
}

// define ComputeThermalRadiationMultiGroup, returns the thermal radiation power for each photon group. = a_r * T^4 * radEnergyFractions
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationMultiGroup(amrex::Real temperature,
										   amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	const double power = radiation_constant_ * std::pow(temperature, 4);
	const auto radEnergyFractions = ComputePlanckEnergyFractions(boundaries, temperature);
	auto Erad_g = power * radEnergyFractions;
	// set floor
	for (int g = 0; g < nGroups_; ++g) {
		if (Erad_g[g] < Erad_floor_) {
			Erad_g[g] = Erad_floor_;
		}
	}
	return Erad_g;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationTempDerivativeSingleGroup(amrex::Real temperature) -> Real
{
	// by default, d emission/dT = 4 emission / T
	return 4. * radiation_constant_ * std::pow(temperature, 3);
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature,
												 amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	// by default, d emission/dT = 4 emission / T
	auto radEnergyFractions = ComputePlanckEnergyFractions(boundaries, temperature);
	double d_power_dt = 4. * radiation_constant_ * std::pow(temperature, 3);
	return d_power_dt * radEnergyFractions;
}

// Define the background heating rate for the gas-dust-radiation system. Units in cgs: erg cm^-3 s^-1
template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineBackgroundHeatingRate(amrex::Real const /*num_density*/) -> amrex::Real
{
	return 0.0;
}

// Define the net cooling rate (line cooling + heating) for the gas-dust-radiation system. Units in cgs: erg cm^-3 s^-1
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineNetCoolingRate(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	return cooling;
}

// Define the derivative of the net cooling rate with respect to temperature for the gas-dust-radiation system. Units in cgs: erg cm^-3 s^-1 K^-1
template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineNetCoolingRateTempDerivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> cooling{};
	cooling.fillin(0.0);
	return cooling;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineCosmicRayHeatingRate(amrex::Real const /*num_density*/) -> double
{
	return 0.0;
}

// Linear equation solver for matrix with non-zeros at the first row, first column, and diagonal only.
// solve the linear system
//   [J00 J0g] [x0] - [F0] = 0
//   [Jg0 Jgg] [xg] - [Fg] = 0
// for x0 and xg, where g = 1, 2, ..., nGroups
template <typename problem_t>
AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqs(JacobianResult<problem_t> const &jacobian, double &x0, quokka::valarray<double, nGroups_> &xi)
{
	auto ratios = jacobian.J0g / jacobian.Jgg;
	x0 = (sum(ratios * jacobian.Fg) - jacobian.F0) / (-sum(ratios * jacobian.Jg0) + jacobian.J00);
	xi = (-1.0 * jacobian.Fg - jacobian.Jg0 * x0) / jacobian.Jgg;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::Solve3x3matrix(const double C00, const double C01, const double C02, const double C10, const double C11,
								const double C12, const double C20, const double C21, const double C22, const double Y0,
								const double Y1, const double Y2) -> std::tuple<amrex::Real, amrex::Real, amrex::Real>
{
	// Solve the 3x3 matrix equation: C * X = Y under the assumption that only the diagonal terms
	// are guaranteed to be non-zero and are thus allowed to be divided by.

	auto E11 = C11 - C01 * C10 / C00;
	auto E12 = C12 - C02 * C10 / C00;
	auto E21 = C21 - C01 * C20 / C00;
	auto E22 = C22 - C02 * C20 / C00;
	auto Z1 = Y1 - Y0 * C10 / C00;
	auto Z2 = Y2 - Y0 * C20 / C00;
	auto X2 = (Z2 - Z1 * E21 / E11) / (E22 - E12 * E21 / E11);
	auto X1 = (Z1 - E12 * X2) / E11;
	auto X0 = (Y0 - C01 * X1 - C02 * X2) / C00;

	return std::make_tuple(X0, X1, X2);
}

template <typename problem_t>
void RadSystem<problem_t>::SetRadEnergySource(array_t &radEnergySource, amrex::Box const &indexRange, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi, amrex::Real time)
{
	// do nothing -- user implemented
}

template <typename problem_t>
void RadSystem<problem_t>::ConservedToPrimitive(amrex::Array4<const amrex::Real> const &cons, array_t &primVar, amrex::Box const &indexRange)
{
	// keep radiation energy density as-is
	// convert (Fx,Fy,Fz) into reduced flux components (fx,fy,fx):
	//   F_x -> F_x / (c*E_r)

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// add reduced fluxes for each radiation group
		for (int g = 0; g < nGroups_; ++g) {
			const auto E_r = cons(i, j, k, radEnergy_index + numRadVars_ * g);
			const auto Fx = cons(i, j, k, x1RadFlux_index + numRadVars_ * g);
			const auto Fy = cons(i, j, k, x2RadFlux_index + numRadVars_ * g);
			const auto Fz = cons(i, j, k, x3RadFlux_index + numRadVars_ * g);

			// check admissibility of states
			AMREX_ASSERT(E_r > 0.0); // NOLINT

			primVar(i, j, k, primRadEnergy_index + numRadVars_ * g) = E_r;
			primVar(i, j, k, x1ReducedFlux_index + numRadVars_ * g) = Fx / (c_light_ * E_r);
			primVar(i, j, k, x2ReducedFlux_index + numRadVars_ * g) = Fy / (c_light_ * E_r);
			primVar(i, j, k, x3ReducedFlux_index + numRadVars_ * g) = Fz / (c_light_ * E_r);
		}
	});
}

template <typename problem_t>
void RadSystem<problem_t>::ComputeMaxSignalSpeed(amrex::Array4<const amrex::Real> const & /*cons*/, array_t &maxSignal, amrex::Box const &indexRange)
{
	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double signal_max = chat0_;
		maxSignal(i, j, k) = signal_max;
	});
}

template <typename problem_t> AMREX_GPU_DEVICE auto RadSystem<problem_t>::isStateValid(std::array<amrex::Real, nvarHyperbolic_> &cons) -> bool
{
	// check if the state variable 'cons' is a valid state
	bool isValid = true;
	for (int g = 0; g < nGroups_; ++g) {
		const auto E_r = cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fx = cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fy = cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fz = cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_];

		const auto Fnorm = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
		const auto f = Fnorm / (c_light_ * E_r);

		bool isNonNegative = (E_r > 0.);
		bool isFluxCausal = (f <= 1.);
		isValid = (isValid && isNonNegative && isFluxCausal);
	}
	return isValid;
}

template <typename problem_t> AMREX_GPU_DEVICE void RadSystem<problem_t>::amendRadState(std::array<amrex::Real, nvarHyperbolic_> &cons)
{
	// amend the state variable 'cons' to be a valid state
	for (int g = 0; g < nGroups_; ++g) {
		auto E_r = cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_];
		if (E_r < Erad_floor_) {
			E_r = Erad_floor_;
			cons[radEnergy_index + numRadVars_ * g - nstartHyperbolic_] = Erad_floor_;
		}
		const auto Fx = cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fy = cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		const auto Fz = cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_];
		if (Fx * Fx + Fy * Fy + Fz * Fz > c_light_ * c_light_ * E_r * E_r) {
			const auto Fnorm = std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
			cons[x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fx / Fnorm * c_light_ * E_r;
			cons[x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fy / Fnorm * c_light_ * E_r;
			cons[x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_] = Fz / Fnorm * c_light_ * E_r;
		}
	}
}

template <typename problem_t>
void RadSystem<problem_t>::PredictStep(arrayconst_t &consVarOld, array_t &consVarNew, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
				       amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
				       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	const auto dx = dx_in[0];
	const auto x1Flux = fluxArray[0];
	// const auto x1FluxDiffusive = fluxDiffusiveArray[0];
#if (AMREX_SPACEDIM >= 2)
	const auto dy = dx_in[1];
	const auto x2Flux = fluxArray[1];
	// const auto x2FluxDiffusive = fluxDiffusiveArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	const auto dz = dx_in[2];
	const auto x3Flux = fluxArray[2];
	// const auto x3FluxDiffusive = fluxDiffusiveArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		std::array<amrex::Real, nvarHyperbolic_> cons{};

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			cons[n] = consVarOld(i, j, k, nstartHyperbolic_ + n) + (AMREX_D_TERM((dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n)),
											     +(dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n)),
											     +(dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n))));
		}

		if (!isStateValid(cons)) {
			amendRadState(cons);
		}
		AMREX_ASSERT(isStateValid(cons));

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			consVarNew(i, j, k, nstartHyperbolic_ + n) = cons[n];
		}
	});
}

template <typename problem_t>
void RadSystem<problem_t>::AddFluxesRK2(array_t &U_new, arrayconst_t &U0, arrayconst_t &U1, amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArrayOld,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> fluxArray,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArrayOld*/,
					amrex::GpuArray<arrayconst_t, AMREX_SPACEDIM> /*fluxDiffusiveArray*/, const double dt_in,
					amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_in, amrex::Box const &indexRange, const int /*nvars*/)
{
	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

	auto const dt = dt_in;
	const auto dx = dx_in[0];
	const auto x1FluxOld = fluxArrayOld[0];
	const auto x1Flux = fluxArray[0];
#if (AMREX_SPACEDIM >= 2)
	const auto dy = dx_in[1];
	const auto x2FluxOld = fluxArrayOld[1];
	const auto x2Flux = fluxArray[1];
#endif
#if (AMREX_SPACEDIM == 3)
	const auto dz = dx_in[2];
	const auto x3FluxOld = fluxArrayOld[2];
	const auto x3Flux = fluxArray[2];
#endif

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		std::array<amrex::Real, nvarHyperbolic_> cons_new{};

		// y^n+1 = (1 - a32) y^n + a32 y^(2) + dt * (0.5 - a32) * s(y^n) + dt * 0.5 * s(y^(2)) + dt * (1 - a32) * f(y^n+1)          // the last term is
		// implicit and not used here
		for (int n = 0; n < nvarHyperbolic_; ++n) {
			const double U_0 = U0(i, j, k, nstartHyperbolic_ + n);
			const double U_1 = U1(i, j, k, nstartHyperbolic_ + n);
			const double FxU_0 = (dt / dx) * (x1FluxOld(i, j, k, n) - x1FluxOld(i + 1, j, k, n));
			const double FxU_1 = (dt / dx) * (x1Flux(i, j, k, n) - x1Flux(i + 1, j, k, n));
#if (AMREX_SPACEDIM >= 2)
			const double FyU_0 = (dt / dy) * (x2FluxOld(i, j, k, n) - x2FluxOld(i, j + 1, k, n));
			const double FyU_1 = (dt / dy) * (x2Flux(i, j, k, n) - x2Flux(i, j + 1, k, n));
#endif
#if (AMREX_SPACEDIM == 3)
			const double FzU_0 = (dt / dz) * (x3FluxOld(i, j, k, n) - x3FluxOld(i, j, k + 1, n));
			const double FzU_1 = (dt / dz) * (x3Flux(i, j, k, n) - x3Flux(i, j, k + 1, n));
#endif
			// save results in cons_new
			cons_new[n] = (1.0 - IMEX_a32) * U_0 + IMEX_a32 * U_1 + ((0.5 - IMEX_a32) * (AMREX_D_TERM(FxU_0, +FyU_0, +FzU_0))) +
				      (0.5 * (AMREX_D_TERM(FxU_1, +FyU_1, +FzU_1)));
		}

		if (!isStateValid(cons_new)) {
			amendRadState(cons_new);
		}
		AMREX_ASSERT(isStateValid(cons_new));

		for (int n = 0; n < nvarHyperbolic_; ++n) {
			U_new(i, j, k, nstartHyperbolic_ + n) = cons_new[n];
		}
	});
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEddingtonFactor(double f_in) -> double
{
	// f is the reduced flux == |F|/cE.
	// compute Levermore (1984) closure [Eq. 25]
	// the is the M1 closure that is derived from Lorentz invariance
	const double f = clamp(f_in, 0., 1.); // restrict f to be within [0, 1]
	const double f_fac = std::sqrt(4.0 - 3.0 * (f * f));
	const double chi = (3.0 + 4.0 * (f * f)) / (5.0 + 2.0 * f_fac);

#if 0 // NOLINT
      // compute Minerbo (1978) closure [piecewise approximation]
      // (For unknown reasons, this closure tends to work better
      // than the Levermore/Lorentz closure on the Su & Olson 1997 test.)
	const double chi = (f < 1. / 3.) ? (1. / 3.) : (0.5 - f + 1.5 * f*f);
#endif

	return chi;
}

template <typename problem_t>
template <typename ArrayType>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeMassScalars(ArrayType const &arr, int i, int j, int k) -> amrex::GpuArray<Real, nmscalars_>
{
	amrex::GpuArray<Real, nmscalars_> massScalars{};
	for (int n = 0; n < nmscalars_; ++n) {
		massScalars[n] = arr(i, j, k, scalar0_index + n);
	}
	return massScalars;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeCellOpticalDepthAllDirMin(const amrex::Array4<const amrex::Real> &consVar,
								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k, const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> tau{};

	const double rho = consVar(i, j, k, gasDensity_index);
	const double E_gas = consVar(i, j, k, gasInternalEnergy_index);
	const auto massScalars = RadSystem<problem_t>::ComputeMassScalars(consVar, i, j, k);
	const double T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, E_gas, massScalars);
	quokka::valarray<double, nGroups_> chi{};
	if constexpr (nGroups_ == 1) {
		chi[0] = rho * ComputePlanckOpacity(rho, T_gas);
	} else {
		const auto opacity = DefineOpacityExponentsAndLowerValues(group_boundaries, rho, T_gas);
		chi = rho * ComputeBinCenterOpacity(group_boundaries, opacity);
	}
	const double dx_min = AMREX_D_PICK(dx[0], std::min(dx[0], dx[1]), std::min({dx[0], dx[1], dx[2]}));
	tau = dx_min * chi;
	return tau;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeCellOpticalDepth(const quokka::Array4View<const amrex::Real, DIR> &consVar,
								    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, int i, int j, int k,
								    const amrex::GpuArray<double, nGroups_ + 1> &group_boundaries)
    -> quokka::valarray<double, nGroups_>
{
	// compute interface-averaged cell optical depth

	// [By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xleft_(i) is the "left"-side of the interface at
	// the left edge of zone i, and xright_(i) is the "right"-side of the
	// interface at the *left* edge of zone i.]

	// piecewise-constant reconstruction
	const double rho_L = consVar(i - 1, j, k, gasDensity_index);
	const double rho_R = consVar(i, j, k, gasDensity_index);

	const double x1GasMom_L = consVar(i - 1, j, k, x1GasMomentum_index);
	const double x1GasMom_R = consVar(i, j, k, x1GasMomentum_index);

	const double x2GasMom_L = consVar(i - 1, j, k, x2GasMomentum_index);
	const double x2GasMom_R = consVar(i, j, k, x2GasMomentum_index);

	const double x3GasMom_L = consVar(i - 1, j, k, x3GasMomentum_index);
	const double x3GasMom_R = consVar(i, j, k, x3GasMomentum_index);

	const double Egas_L = consVar(i - 1, j, k, gasEnergy_index);
	const double Egas_R = consVar(i, j, k, gasEnergy_index);

	auto massScalars_L = RadSystem<problem_t>::ComputeMassScalars(consVar, i - 1, j, k);
	auto massScalars_R = RadSystem<problem_t>::ComputeMassScalars(consVar, i, j, k);

	double Eint_L = NAN;
	double Eint_R = NAN;
	double Tgas_L = NAN;
	double Tgas_R = NAN;

	if constexpr (gamma_ != 1.0) {
		Eint_L = RadSystem<problem_t>::ComputeEintFromEgas(rho_L, x1GasMom_L, x2GasMom_L, x3GasMom_L, Egas_L);
		Eint_R = RadSystem<problem_t>::ComputeEintFromEgas(rho_R, x1GasMom_R, x2GasMom_R, x3GasMom_R, Egas_R);
		Tgas_L = quokka::EOS<problem_t>::ComputeTgasFromEint(rho_L, Eint_L, massScalars_L);
		Tgas_R = quokka::EOS<problem_t>::ComputeTgasFromEint(rho_R, Eint_R, massScalars_R);
	}

	double dl = NAN;
	if constexpr (DIR == FluxDir::X1) {
		dl = dx[0];
	} else if constexpr (DIR == FluxDir::X2) {
		dl = dx[1];
	} else if constexpr (DIR == FluxDir::X3) {
		dl = dx[2];
	}

	quokka::valarray<double, nGroups_> optical_depths{};
	if constexpr (nGroups_ == 1) {
		const double tau_L = dl * rho_L * RadSystem<problem_t>::ComputeFluxMeanOpacity(rho_L, Tgas_L);
		const double tau_R = dl * rho_R * RadSystem<problem_t>::ComputeFluxMeanOpacity(rho_R, Tgas_R);
		optical_depths[0] = (tau_L * tau_R * 2.) / (tau_L + tau_R); // harmonic mean. Alternative: 0.5*(tau_L + tau_R)
	} else {
		const auto opacity_L = DefineOpacityExponentsAndLowerValues(group_boundaries, rho_L, Tgas_L);
		const auto opacity_R = DefineOpacityExponentsAndLowerValues(group_boundaries, rho_R, Tgas_R);
		const auto tau_L = dl * rho_L * ComputeBinCenterOpacity(group_boundaries, opacity_L);
		const auto tau_R = dl * rho_R * ComputeBinCenterOpacity(group_boundaries, opacity_R);
		optical_depths = (tau_L * tau_R * 2.) / (tau_L + tau_R); // harmonic mean. Alternative: 0.5*(tau_L + tau_R)
	}

	return optical_depths;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeEddingtonTensor(const double fx, const double fy, const double fz) -> std::array<std::array<double, 3>, 3>
{
	// Compute the radiation pressure tensor

	// AMREX_ASSERT(f < 1.0); // there is sometimes a small (<1%) flux
	// limiting violation when using P1 AMREX_ASSERT(f_R < 1.0);

	auto f = std::sqrt(fx * fx + fy * fy + fz * fz);
	std::array<amrex::Real, 3> fvec = {fx, fy, fz};

	// angle between interface and radiation flux \hat{n}
	// If direction is undefined, just drop direction-dependent
	// terms.
	std::array<amrex::Real, 3> n{};

	for (int ii = 0; ii < 3; ++ii) {
		n[ii] = (f > 0.) ? (fvec[ii] / f) : 0.;
	}

	// compute radiation pressure tensors
	const double chi = RadSystem<problem_t>::ComputeEddingtonFactor(f);

	AMREX_ASSERT((chi >= 1. / 3.) && (chi <= 1.0)); // NOLINT

	// diagonal term of Eddington tensor
	const double Tdiag = (1.0 - chi) / 2.0;

	// anisotropic term of Eddington tensor (in the direction of the
	// rad. flux)
	const double Tf = (3.0 * chi - 1.0) / 2.0;

	// assemble Eddington tensor
	std::array<std::array<double, 3>, 3> T{};

	for (int ii = 0; ii < 3; ++ii) {
		for (int jj = 0; jj < 3; ++jj) {
			const double delta_ij = (ii == jj) ? 1 : 0;
			T[ii][jj] = Tdiag * delta_ij + Tf * (n[ii] * n[jj]);
		}
	}

	return T;
}

template <typename problem_t>
template <FluxDir DIR>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeRadPressure(const double erad, const double Fx, const double Fy, const double Fz, const double fx,
							       const double fy, const double fz) -> RadPressureResult
{
	// Compute the radiation pressure tensor and the maximum signal speed and return them as a struct.

	// check that states are physically admissible
	AMREX_ASSERT(erad > 0.0);

	// Compute the Eddington tensor
	auto T = ComputeEddingtonTensor(fx, fy, fz);

	// frozen Eddington tensor approximation, following Balsara
	// (1999) [JQSRT Vol. 61, No. 5, pp. 617–627, 1999], Eq. 46.
	double Tnormal = NAN;
	if constexpr (DIR == FluxDir::X1) {
		Tnormal = T[0][0];
	} else if constexpr (DIR == FluxDir::X2) {
		Tnormal = T[1][1];
	} else if constexpr (DIR == FluxDir::X3) {
		Tnormal = T[2][2];
	}

	// compute fluxes F_L, F_R
	// T_nx, T_ny, T_nz indicate components where 'n' is the direction of the
	// face normal. F_n is the radiation flux component in the direction of the
	// face normal
	double Fn = NAN;
	double Tnx = NAN;
	double Tny = NAN;
	double Tnz = NAN;

	if constexpr (DIR == FluxDir::X1) {
		Fn = Fx;

		Tnx = T[0][0];
		Tny = T[0][1];
		Tnz = T[0][2];
	} else if constexpr (DIR == FluxDir::X2) {
		Fn = Fy;

		Tnx = T[1][0];
		Tny = T[1][1];
		Tnz = T[1][2];
	} else if constexpr (DIR == FluxDir::X3) {
		Fn = Fz;

		Tnx = T[2][0];
		Tny = T[2][1];
		Tnz = T[2][2];
	}

	AMREX_ASSERT(Fn != NAN);
	AMREX_ASSERT(Tnx != NAN);
	AMREX_ASSERT(Tny != NAN);
	AMREX_ASSERT(Tnz != NAN);

	RadPressureResult result{};
	result.F = {Fn, Tnx * erad, Tny * erad, Tnz * erad};
	// It might be possible to remove this 0.1 floor without affecting the code. I tried and only the 3D RadForce failed (causing S_L = S_R = 0.0 and F[0] =
	// NAN). Read more on https://github.com/quokka-astro/quokka/pull/582 .
	result.S = std::max(0.1, std::sqrt(Tnormal));

	return result;
}

template <typename problem_t>
template <FluxDir DIR>
void RadSystem<problem_t>::ComputeFluxes(array_t &x1Flux_in, array_t &x1FluxDiffusive_in, amrex::Array4<const amrex::Real> const &x1LeftState_in,
					 amrex::Array4<const amrex::Real> const &x1RightState_in, amrex::Box const &indexRange, arrayconst_t &consVar_in,
					 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, bool const use_wavespeed_correction,
					 amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor_in)
{
	quokka::Array4View<const amrex::Real, DIR> x1LeftState(x1LeftState_in);
	quokka::Array4View<const amrex::Real, DIR> x1RightState(x1RightState_in);
	quokka::Array4View<amrex::Real, DIR> x1Flux(x1Flux_in);
	quokka::Array4View<amrex::Real, DIR> x1FluxDiffusive(x1FluxDiffusive_in);
	quokka::Array4View<const amrex::Real, DIR> consVar(consVar_in);

	// quokka::Array4View<const amrex::Real, 0> reducedSpeedOfLightFactor(reducedSpeedOfLightFactor_in);

	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	// By convention, the interfaces are defined on the left edge of each
	// zone, i.e. xinterface_(i) is the solution to the Riemann problem at
	// the left edge of zone i.

	// Indexing note: There are (nx + 1) interfaces for nx zones.

	// interface-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i_in, int j_in, int k_in) {
		auto [i, j, k] = quokka::reorderMultiIndex<DIR>(i_in, j_in, k_in);

		// const double c_hat_local = reducedSpeedOfLightFactor_in(i, j, k) * c_light_;

		amrex::GpuArray<double, nGroups_ + 1> radBoundaries_g_copy{};
		for (int g = 0; g < nGroups_ + 1; ++g) {
			radBoundaries_g_copy[g] = radBoundaries_g[g];
		}

		// HLL solver following Toro (1998) and Balsara (2017).
		// Radiation eigenvalues from Skinner & Ostriker (2013).

		// calculate cell optical depth for each photon group
		// Similar to the asymptotic-preserving flux correction in Skinner et al. (2019). Use optionally apply it here to reduce odd-even instability.
		quokka::valarray<double, nGroups_> tau_cell{};
		if (use_wavespeed_correction) {
			tau_cell = ComputeCellOpticalDepth<DIR>(consVar, dx, i, j, k, radBoundaries_g_copy);
		}

		// gather left- and right- state variables
		for (int g = 0; g < nGroups_; ++g) {
			double erad_L = x1LeftState(i, j, k, primRadEnergy_index + numRadVars_ * g);
			double erad_R = x1RightState(i, j, k, primRadEnergy_index + numRadVars_ * g);

			double fx_L = x1LeftState(i, j, k, x1ReducedFlux_index + numRadVars_ * g);
			double fx_R = x1RightState(i, j, k, x1ReducedFlux_index + numRadVars_ * g);

			double fy_L = x1LeftState(i, j, k, x2ReducedFlux_index + numRadVars_ * g);
			double fy_R = x1RightState(i, j, k, x2ReducedFlux_index + numRadVars_ * g);

			double fz_L = x1LeftState(i, j, k, x3ReducedFlux_index + numRadVars_ * g);
			double fz_R = x1RightState(i, j, k, x3ReducedFlux_index + numRadVars_ * g);

			// compute scalar reduced flux f
			double f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
			double f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);

			// Compute "un-reduced" Fx, Fy, Fz
			double Fx_L = fx_L * (c_light_ * erad_L);
			double Fx_R = fx_R * (c_light_ * erad_R);

			double Fy_L = fy_L * (c_light_ * erad_L);
			double Fy_R = fy_R * (c_light_ * erad_R);

			double Fz_L = fz_L * (c_light_ * erad_L);
			double Fz_R = fz_R * (c_light_ * erad_R);

			// check that states are physically admissible; if not, use first-order
			// reconstruction
			if ((erad_L <= 0.) || (erad_R <= 0.) || (f_L >= 1.) || (f_R >= 1.)) {
				erad_L = consVar(i - 1, j, k, radEnergy_index + numRadVars_ * g);
				erad_R = consVar(i, j, k, radEnergy_index + numRadVars_ * g);

				Fx_L = consVar(i - 1, j, k, x1RadFlux_index + numRadVars_ * g);
				Fx_R = consVar(i, j, k, x1RadFlux_index + numRadVars_ * g);

				Fy_L = consVar(i - 1, j, k, x2RadFlux_index + numRadVars_ * g);
				Fy_R = consVar(i, j, k, x2RadFlux_index + numRadVars_ * g);

				Fz_L = consVar(i - 1, j, k, x3RadFlux_index + numRadVars_ * g);
				Fz_R = consVar(i, j, k, x3RadFlux_index + numRadVars_ * g);

				// compute primitive variables
				fx_L = Fx_L / (c_light_ * erad_L);
				fx_R = Fx_R / (c_light_ * erad_R);

				fy_L = Fy_L / (c_light_ * erad_L);
				fy_R = Fy_R / (c_light_ * erad_R);

				fz_L = Fz_L / (c_light_ * erad_L);
				fz_R = Fz_R / (c_light_ * erad_R);

				f_L = std::sqrt(fx_L * fx_L + fy_L * fy_L + fz_L * fz_L);
				f_R = std::sqrt(fx_R * fx_R + fy_R * fy_R + fz_R * fz_R);
			}

			// ComputeRadPressure returns F_L_and_S_L or F_R_and_S_R
			auto [F_L, S_L] = ComputeRadPressure<DIR>(erad_L, Fx_L, Fy_L, Fz_L, fx_L, fy_L, fz_L);
			S_L *= -1.; // speed sign is -1
			auto [F_R, S_R] = ComputeRadPressure<DIR>(erad_R, Fx_R, Fy_R, Fz_R, fx_R, fy_R, fz_R);

			// set c_hat_left and c_hat_right to the minimum of the two adjacent cells
			const double c_hat_left = std::min(reducedSpeedOfLightFactor_in(i - 1, j, k, g), reducedSpeedOfLightFactor_in(i, j, k, g)) * c_light_;
			const double c_hat_right = std::min(reducedSpeedOfLightFactor_in(i, j, k, g), reducedSpeedOfLightFactor_in(i + 1, j, k, g)) * c_light_;

			// correct for reduced speed of light
			F_L[0] *= c_hat_left / c_light_;
			F_R[0] *= c_hat_right / c_light_;
			for (int n = 1; n < numRadVars_; ++n) {
				F_L[n] *= c_hat_left * c_light_;
				F_R[n] *= c_hat_right * c_light_;
			}
			S_L *= c_hat_left;
			S_R *= c_hat_right;

			const quokka::valarray<double, numRadVars_> U_L = {erad_L, Fx_L, Fy_L, Fz_L};
			const quokka::valarray<double, numRadVars_> U_R = {erad_R, Fx_R, Fy_R, Fz_R};

			// Adjusting wavespeeds is no longer necessary with the IMEX PD-ARS scheme.
			// Read more in https://github.com/quokka-astro/quokka/pull/582
			// However, we let the user optionally apply it to reduce odd-even instability.
			quokka::valarray<double, numRadVars_> epsilon = {1.0, 1.0, 1.0, 1.0};
			if (use_wavespeed_correction) {
				// no correction for odd zones
				if ((i + j + k) % 2 == 0) {
					const double S_corr = std::min(1.0, 1.0 / tau_cell[g]); // Skinner et al.
					epsilon = {S_corr, 1.0, 1.0, 1.0};			// Skinner et al. (2019)
				}
			}

			AMREX_ASSERT(std::abs(S_L) <= c_hat_left); // NOLINT
			AMREX_ASSERT(std::abs(S_R) <= c_hat_right); // NOLINT

			// in the frozen Eddington tensor approximation, we are always
			// in the star region, so F = F_star
			const quokka::valarray<double, numRadVars_> F =
			    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R + epsilon * (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

			// check states are valid
			AMREX_ASSERT(!std::isnan(F[0])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[1])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[2])); // NOLINT
			AMREX_ASSERT(!std::isnan(F[3])); // NOLINT

			x1Flux(i, j, k, radEnergy_index + numRadVars_ * g - nstartHyperbolic_) = F[0];
			x1Flux(i, j, k, x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[1];
			x1Flux(i, j, k, x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[2];
			x1Flux(i, j, k, x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = F[3];

			const quokka::valarray<double, numRadVars_> diffusiveF =
			    (S_R / (S_R - S_L)) * F_L - (S_L / (S_R - S_L)) * F_R + (S_R * S_L / (S_R - S_L)) * (U_R - U_L);

			x1FluxDiffusive(i, j, k, radEnergy_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[0];
			x1FluxDiffusive(i, j, k, x1RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[1];
			x1FluxDiffusive(i, j, k, x2RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[2];
			x1FluxDiffusive(i, j, k, x3RadFlux_index + numRadVars_ * g - nstartHyperbolic_) = diffusiveF[3];
		} // end loop over radiation groups
	});
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> Real
{
	return NAN;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEnergyMeanOpacity(const double rho, const double Tgas) -> Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/,
										      const double /*rho*/, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents_and_values[0][g] = NAN;
		exponents_and_values[1][g] = NAN;
	}
	return exponents_and_values;
}

template <typename problem_t>
template <typename ArrayType>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeRadQuantityExponents(ArrayType const &quant, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> amrex::GpuArray<double, nGroups_>
{
	// Compute the exponents for the radiation energy density, radiation flux, radiation pressure, or Planck function.

	// Note: Could save some memory by using bin_center_previous and bin_center_current
	amrex::GpuArray<double, nGroups_> bin_center{};
	amrex::GpuArray<double, nGroups_> quant_mean{};
	amrex::GpuArray<double, nGroups_ - 1> logslopes{};
	amrex::GpuArray<double, nGroups_> exponents{};
	for (int g = 0; g < nGroups_; ++g) {
		bin_center[g] = std::sqrt(boundaries[g] * boundaries[g + 1]);
		quant_mean[g] = quant[g] / (boundaries[g + 1] - boundaries[g]);
		if (g > 0) {
			AMREX_ASSERT(bin_center[g] > bin_center[g - 1]);
			if (quant_mean[g] == 0.0 && quant_mean[g - 1] == 0.0) {
				logslopes[g - 1] = 0.0;
			} else if (quant_mean[g - 1] * quant_mean[g] <= 0.0) {
				if (quant_mean[g] > quant_mean[g - 1]) {
					logslopes[g - 1] = inf;
				} else {
					logslopes[g - 1] = -inf;
				}
			} else {
				logslopes[g - 1] = std::log(std::abs(quant_mean[g] / quant_mean[g - 1])) / std::log(bin_center[g] / bin_center[g - 1]);
			}
			AMREX_ASSERT(!std::isnan(logslopes[g - 1]));
		}
	}

	for (int g = 0; g < nGroups_; ++g) {
		if (g == 0) {
			if constexpr (!special_edge_bin_slopes) {
				exponents[g] = -1.0;
			} else {
				exponents[g] = 2.0;
			}
		} else if (g == nGroups_ - 1) {
			if constexpr (!special_edge_bin_slopes) {
				exponents[g] = -1.0;
			} else {
				exponents[g] = -4.0;
			}
		} else {
			exponents[g] = minmod_func(logslopes[g - 1], logslopes[g]);
		}
		AMREX_ASSERT(!std::isnan(exponents[g]));
	}

	if constexpr (PPL_free_slope_st_total) {
		int peak_idx = 0; // index of the peak of logslopes
		for (; peak_idx < nGroups_; ++peak_idx) {
			if (peak_idx == nGroups_ - 1) {
				peak_idx += 0;
				break;
			}
			if (exponents[peak_idx] >= 0.0 && exponents[peak_idx + 1] < 0.0) {
				break;
			}
		}
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(peak_idx < nGroups_ - 1,
						 "Peak index not found. Here peak_index is the index at which the exponent changes its sign.");
		double quant_sum = 0.0;
		double part_sum = 0.0;
		for (int g = 0; g < nGroups_; ++g) {
			quant_sum += quant[g];
			if (g == peak_idx) {
				continue;
			}
			part_sum += exponents[g] * quant[g];
		}
		if (quant[peak_idx] > 0.0 && quant_sum > 0.0) {
			exponents[peak_idx] = (-quant_sum - part_sum) / quant[peak_idx];
			AMREX_ASSERT(!std::isnan(exponents[peak_idx]));
		}
	}
	return exponents;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto
RadSystem<problem_t>::ComputeGroupMeanOpacity(amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const &kappa_expo_and_lower_value,
					      amrex::GpuArray<double, nGroups_> const &radBoundaryRatios, amrex::GpuArray<double, nGroups_> const &alpha_quant)
    -> quokka::valarray<double, nGroups_>
{
	amrex::GpuArray<double, nGroups_ + 1> const &alpha_kappa = kappa_expo_and_lower_value[0];
	amrex::GpuArray<double, nGroups_ + 1> const &kappa_lower = kappa_expo_and_lower_value[1];

	quokka::valarray<double, nGroups_> kappa{};
	for (int g = 0; g < nGroups_; ++g) {
		double alpha = alpha_quant[g] + 1.0;
		if (alpha > 100.) {
			kappa[g] = kappa_lower[g] * std::pow(radBoundaryRatios[g], kappa_expo_and_lower_value[0][g]);
			continue;
		}
		if (alpha < -100.) {
			kappa[g] = kappa_lower[g];
			continue;
		}
		double part1 = 0.0;
		if (std::abs(alpha) < 1e-8) {
			part1 = std::log(radBoundaryRatios[g]);
		} else {
			part1 = (std::pow(radBoundaryRatios[g], alpha) - 1.0) / alpha;
		}
		alpha += alpha_kappa[g];
		double part2 = 0.0;
		if (std::abs(alpha) < 1e-8) {
			part2 = std::log(radBoundaryRatios[g]);
		} else {
			part2 = (std::pow(radBoundaryRatios[g], alpha) - 1.0) / alpha;
		}
		kappa[g] = kappa_lower[g] / part1 * part2;
		AMREX_ASSERT(!std::isnan(kappa[g]));
	}
	return kappa;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEintFromEgas(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom,
								     const double Etot) -> double
{
	const double p_sq = X1GasMom * X1GasMom + X2GasMom * X2GasMom + X3GasMom * X3GasMom;
	const double Ekin = p_sq / (2.0 * density);
	const double Eint = Etot - Ekin;
	AMREX_ASSERT_WITH_MESSAGE(Eint > 0., "Gas internal energy is not positive!");
	return Eint;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeEgasFromEint(const double density, const double X1GasMom, const double X2GasMom, const double X3GasMom,
								     const double Eint) -> double
{
	const double p_sq = X1GasMom * X1GasMom + X2GasMom * X2GasMom + X3GasMom * X3GasMom;
	const double Ekin = p_sq / (2.0 * density);
	const double Etot = Eint + Ekin;
	return Etot;
}

template <typename problem_t> AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::PlanckFunction(const double nu, const double T) -> double
{
	// returns 4 pi B(nu) / c
	double const coeff = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * T);
	double const x = coeff * nu;
	if (x > 100.) {
		return 0.0;
	}
	double planck_integral = NAN;
	if (x <= 1.0e-10) {
		// Taylor series
		planck_integral = x * x - x * x * x / 2.;
	} else {
		planck_integral = std::pow(x, 3) / (std::exp(x) - 1.0);
	}
	return coeff / (std::pow(PI, 4) / 15.0) * (radiation_constant_ * std::pow(T, 4)) * planck_integral;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeDiffusionFluxMeanOpacity(const quokka::valarray<double, nGroups_> kappaPVec,
										 const quokka::valarray<double, nGroups_> kappaEVec,
										 const quokka::valarray<double, nGroups_> fourPiBoverC,
										 const amrex::GpuArray<double, nGroups_> delta_nu_kappa_B_at_edge,
										 const amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge,
										 const amrex::GpuArray<double, nGroups_ + 1> kappa_slope)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappaF{};
	for (int g = 0; g < nGroups_; ++g) {
		// kappaF[g] = 4. / 3. * kappaPVec[g] * fourPiBoverC[g] + 1. / 3. * kappa_slope[g] * kappaPVec[g] * fourPiBoverC[g] - 1. / 3. *
		// delta_nu_kappa_B_at_edge[g];
		kappaF[g] = (kappaPVec[g] + 1. / 3. * kappaEVec[g]) * fourPiBoverC[g] +
			    1. / 3. * (kappa_slope[g] * kappaEVec[g] * fourPiBoverC[g] - delta_nu_kappa_B_at_edge[g]);
		auto const denom = 4. / 3. * fourPiBoverC[g] - 1. / 3. * delta_nu_B_at_edge[g];
		if (denom <= 0.0) {
			AMREX_ASSERT(kappaF[g] == 0.0);
			kappaF[g] = 0.0;
		} else {
			kappaF[g] /= denom;
		}
	}
	return kappaF;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeBinCenterOpacity(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries,
									 amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value)
    -> quokka::valarray<double, nGroups_>
{
	quokka::valarray<double, nGroups_> kappa_center{};
	for (int g = 0; g < nGroups_; ++g) {
		kappa_center[g] =
		    kappa_expo_and_lower_value[1][g] * std::pow(rad_boundaries[g + 1] / rad_boundaries[g], 0.5 * kappa_expo_and_lower_value[0][g]);
	}
	return kappa_center;
}

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::ComputeFluxInDiffusionLimit(const amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, const double T,
									     const double vel) -> amrex::GpuArray<double, nGroups_>
{
	double const coeff = RadSystem_Traits<problem_t>::energy_unit / (boltzmann_constant_ * T);
	amrex::GpuArray<double, nGroups_ + 1> edge_values{};
	amrex::GpuArray<double, nGroups_> flux{};
	for (int g = 0; g < nGroups_ + 1; ++g) {
		auto x = coeff * rad_boundaries[g];
		edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x) - 1. / 3. * x * (std::pow(x, 3) / (std::exp(x) - 1.0)) / gInf;
		// test: reproduce the Planck function
		// edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x);
	}
	for (int g = 0; g < nGroups_; ++g) {
		flux[g] = vel * radiation_constant_ * std::pow(T, 4) * (edge_values[g + 1] - edge_values[g]);
	}
	return flux;
}

template <typename problem_t>
template <typename RHSFunction, typename JacFunction>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::BackwardEulerOneVariable(RHSFunction const &rhs, JacFunction const &jac, const double x0, const double compare)
    -> double
{
	double x = x0;
	const double rel_tol = 1.0e-8;
	const double rel_change_tol = 1.0e-6;
	const int max_iter_td = 100;
	int iter_Td = 0;
	for (; iter_Td < max_iter_td; ++iter_Td) {
		const auto the_rhs = rhs(x);
		if (std::abs(the_rhs) < rel_tol * compare) {
			break;
		}

		const double dT = -the_rhs / jac(x);
		x += dT;

		if (iter_Td > 0) {
			if (std::abs(dT) < rel_change_tol * std::abs(x)) {
				break;
			}
		}
	}

	AMREX_ASSERT_WITH_MESSAGE(iter_Td < max_iter_td, "Newton iteration in IntegratorOneVariable failed to converge.");
	if (iter_Td >= max_iter_td) {
		x = -1.0;
	}

	return x;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeDustTemperatureBateKeto(double const T_gas, double const T_d_init, double const rho,
									   quokka::valarray<double, nGroups_> const &Erad, double N_d, double dt, double R_sum,
									   int n_step, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries) -> double
{
	// if (n_step > 0) {
	// 	const auto T_d = T_gas - R_sum / (N_d * std::sqrt(T_gas));
	// 	AMREX_ASSERT_WITH_MESSAGE(T_d >= 0., "Dust temperature is negative!");
	// 	return T_d;
	// }

	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};

	if constexpr (nGroups_ > 1 && opacity_model_ != OpacityModel::piecewise_constant_opacity) {
		for (int g = 0; g < nGroups_; ++g) {
			rad_boundary_ratios[g] = rad_boundaries[g + 1] / rad_boundaries[g];
		}
	}

	// the RHS of the equation 0 = c_hat_ dt rho (kappa_E * E_g - kappa_P * B_g) + N_d sqrt(T_gas) (T_gas - T_d)
	auto rhs = [=](double T_d) -> double {
		double LHS = NAN;

		if constexpr (nGroups_ == 1) {
			const auto fourPiBoverC = ComputeThermalRadiationSingleGroup(T_d);
			const auto kappaE = ComputeEnergyMeanOpacity(rho, T_d);
			const auto kappaP = ComputePlanckOpacity(rho, T_d);
			LHS = c_light_ * dt * rho * (kappaE * Erad[0] - kappaP * fourPiBoverC) + N_d * std::sqrt(T_gas) * (T_gas - T_d);
		} else {
			const auto fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, rad_boundaries);
			const auto opacity_terms = ComputeModelDependentKappaEAndKappaP(T_d, rho, rad_boundaries, rad_boundary_ratios, fourPiBoverC, Erad, 0);
			LHS =
			    c_light_ * dt * rho * sum(opacity_terms.kappaE * Erad - opacity_terms.kappaP * fourPiBoverC) + N_d * std::sqrt(T_gas) * (T_gas - T_d);
		}

		return LHS;
	};

	// the Jacobian of the RHS of the equation 0 = c_hat_ dt rho (kappa_E * E_g - kappa_P * B_g) + N_d sqrt(T_gas) (T_gas - T_d)
	auto jac = [=](double T_d) -> double {
		double dLHS_dTd = NAN;

		if constexpr (nGroups_ == 1) {
			const auto kappaP = ComputePlanckOpacity(rho, T_d);
			const auto d_fourpib_over_c_d_t = ComputeThermalRadiationTempDerivativeSingleGroup(T_d);
			dLHS_dTd = -c_light_ * dt * rho * (kappaP * d_fourpib_over_c_d_t) - N_d * std::sqrt(T_gas);
		} else {
			const auto fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, rad_boundaries);
			const auto opacity_terms = ComputeModelDependentKappaEAndKappaP(T_d, rho, rad_boundaries, rad_boundary_ratios, fourPiBoverC, Erad, 0);
			const auto d_fourpib_over_c_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, rad_boundaries);
			dLHS_dTd = -c_light_ * dt * rho * sum(opacity_terms.kappaP * d_fourpib_over_c_d_t) - N_d * std::sqrt(T_gas);
		}

		return dLHS_dTd;
	};

	const double Lambda_compare = N_d * std::sqrt(T_gas) * T_gas;

	const auto T_d = BackwardEulerOneVariable(rhs, jac, T_d_init, Lambda_compare);

	return T_d;
}

template <typename problem_t>
AMREX_GPU_DEVICE void RadSystem<problem_t>::ComputeReducedSpeedOfLightFactor(arrayconst_t &consVar_in, const double c_hat_over_c, array_t &reducedSpeedOfLightFactor,
									 const amrex::Box &indexRange, const amrex::GpuArray<double, AMREX_SPACEDIM> &dx)
{
	// quokka::Array4View<const amrex::Real, FluxDir::X1> consVarX1(consVar_in);
	// quokka::Array4View<const amrex::Real, FluxDir::X2> consVarX2(consVar_in);
	// quokka::Array4View<const amrex::Real, FluxDir::X3> consVarX3(consVar_in);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// debugging: print consVar(i - 1, j, k, radEnergy_index), consVar(i, j, k, radEnergy_index), consVar(i + 1, j, k, radEnergy_index)
		// amrex::Print() << "consVar(" << i - 1 << ", " << j << ", " << k << ", " << radEnergy_index << ") = " << consVar(i - 1, j, k, radEnergy_index) << std::endl;
		// amrex::Print() << "consVar(" << i << ", " << j << ", " << k << ", " << radEnergy_index << ") = " << consVar(i, j, k, radEnergy_index) << std::endl;
		// amrex::Print() << "consVar(" << i + 1 << ", " << j << ", " << k << ", " << radEnergy_index << ") = " << consVar(i + 1, j, k, radEnergy_index) << std::endl;
		// amrex::Print() << std::endl;

		// // compute kappaP
		// const double rho = consVar(i, j, k, gasDensity_index);
		// const double E_gas = consVar(i, j, k, gasInternalEnergy_index);
		// const auto massScalars = RadSystem<problem_t>::ComputeMassScalars(consVar, i, j, k);
		// const double T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, E_gas, massScalars);
		// const double chi = rho * ComputePlanckOpacity(rho, T_gas);

		// if (chi <= 0.0) {
		// 	reducedSpeedOfLightFactor(i, j, k) = 1.0;
		// 	return;
		// }

// 		// Calculate first derivatives of Erad
// 		double dEdx = 0.0;
// 		double dEdy = 0.0;
// 		double dEdz = 0.0;

// 		// Calculate second derivatives of Erad 
// 		double d2Edx2 = 0.0;
// 		double d2Edy2 = 0.0;
// 		double d2Edz2 = 0.0;
// 		double Eradc = 0.0;

// 		// Sum over all radiation groups
// 		for (int g = 0; g < nGroups_; ++g) {
// 			const int Erad_idx = radEnergy_index + numRadVars_ * g;

// 			Eradc += consVar(i, j, k, Erad_idx);

// 			// First derivatives (central difference)
// 			dEdx += (consVar(i+1, j, k, Erad_idx) - consVar(i-1, j, k, Erad_idx)) / (2.0 * dx[0]);
// #if AMREX_SPACEDIM > 1
// 			dEdy += (consVar(i, j+1, k, Erad_idx) - consVar(i, j-1, k, Erad_idx)) / (2.0 * dx[1]);
// #endif
// #if AMREX_SPACEDIM > 2
// 			dEdz += (consVar(i, j, k+1, Erad_idx) - consVar(i, j, k-1, Erad_idx)) / (2.0 * dx[2]);
// #endif

// 			// Second derivatives
// 			d2Edx2 += (consVar(i+1, j, k, Erad_idx) - 2.0*consVar(i, j, k, Erad_idx) + consVar(i-1, j, k, Erad_idx)) / (dx[0] * dx[0]);
// #if AMREX_SPACEDIM > 1
// 			d2Edy2 += (consVar(i, j+1, k, Erad_idx) - 2.0*consVar(i, j, k, Erad_idx) + consVar(i, j-1, k, Erad_idx)) / (dx[1] * dx[1]);
// #endif
// #if AMREX_SPACEDIM > 2
// 			d2Edz2 += (consVar(i, j, k+1, Erad_idx) - 2.0*consVar(i, j, k, Erad_idx) + consVar(i, j, k-1, Erad_idx)) / (dx[2] * dx[2]);
// #endif
// 		}

// 		// Calculate magnitudes
// 		const double grad_mag = std::sqrt(dEdx*dEdx + dEdy*dEdy + dEdz*dEdz);
// 		const double laplacian_mag = std::abs(d2Edx2 + d2Edy2 + d2Edz2);

		// // Avoid division by zero
		// // const double R = (grad_mag > 0.0) ? laplacian_mag / grad_mag / chi : 0.0;
		// const double R = 1.0 / chi * laplacian_mag / (grad_mag + 1.0e-3 * Eradc / dx[0]);
		// const double R_scaled = R / 3.0;
		// const double R_scaled4 = R_scaled * R_scaled * R_scaled * R_scaled;

		const auto tau_cell = ComputeCellOpticalDepthAllDirMin(consVar_in, dx, i, j, k, radBoundaries_);
		const int pow = 1;
		const double scaling = 1.0;
		// const double scaling = 3.0;
		const double max_chat_scaleup = 10.0;
		for (int g = 0; g < nGroups_; ++g) {
			const double scaled = std::pow(tau_cell[g] * scaling, pow);
			const double reduced_speed_of_light_factor = c_hat_over_c * (1.0 + max_chat_scaleup * scaled) / (1.0 + scaled);
			reducedSpeedOfLightFactor(i, j, k, g) = std::min(reduced_speed_of_light_factor, 1.0);
		}
	});

	if constexpr (print_chat) {
		// print reducedSpeedOfLightFactor
		amrex::Print() << "reducedSpeedOfLightFactor: ";
		for (int g = 0; g < nGroups_; ++g) {
			amrex::Print() << "group " << g << ": ";
			// Use indexRange to iterate over valid indices
			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
				amrex::Print() << reducedSpeedOfLightFactor(i, j, k, g) << " ";
			});
			amrex::Print() << std::endl;
		}
	}
}

#include "radiation/source_terms_multi_group.hpp"  // IWYU pragma: export
#include "radiation/source_terms_single_group.hpp" // IWYU pragma: export

#endif // RADIATION_SYSTEM_HPP_