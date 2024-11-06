// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef RAD_SOURCE_TERMS_MULTI_GROUP_HPP_ // NOLINT
#define RAD_SOURCE_TERMS_MULTI_GROUP_HPP_

#include "radiation/radiation_system.hpp" // IWYU pragma: keep

// Compute kappaE and kappaP based on the opacity model. The result is stored in the last five arguments: alpha_P, alpha_E, kappaP, kappaE, and kappaPoverE.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeModelDependentKappaEAndKappaP(
    double const T, double const rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, amrex::GpuArray<double, nGroups_> const &rad_boundary_ratios,
    quokka::valarray<double, nGroups_> const &fourPiBoverC, quokka::valarray<double, nGroups_> const &Erad, int const n_iter,
    amrex::GpuArray<double, nGroups_> const &alpha_E, amrex::GpuArray<double, nGroups_> const &alpha_P) -> OpacityTerms<problem_t>
{
	OpacityTerms<problem_t> result;

	const auto kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(rad_boundaries, rho, T);

	if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
		for (int g = 0; g < nGroups_; ++g) {
			result.kappaP[g] = kappa_expo_and_lower_value[1][g];
			result.kappaE[g] = kappa_expo_and_lower_value[1][g];
		}
	} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) {
		amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
		if constexpr (!special_edge_bin_slopes) {
			for (int g = 0; g < nGroups_; ++g) {
				alpha_quant_minus_one[g] = -1.0;
			}
		} else {
			alpha_quant_minus_one[0] = 2.0;
			alpha_quant_minus_one[nGroups_ - 1] = -4.0;
			for (int g = 1; g < nGroups_ - 1; ++g) {
				alpha_quant_minus_one[g] = -1.0;
			}
		}
		result.kappaP = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, alpha_quant_minus_one);
		result.kappaE = result.kappaP;
	} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
		if (n_iter < max_iter_to_update_alpha_E) {
			result.alpha_E = ComputeRadQuantityExponents(Erad, rad_boundaries);
			result.alpha_P = ComputeRadQuantityExponents(fourPiBoverC, rad_boundaries);
		} else {
			result.alpha_E = alpha_E;
			result.alpha_P = alpha_P;
		}
		result.kappaE = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, result.alpha_E);
		result.kappaP = ComputeGroupMeanOpacity(kappa_expo_and_lower_value, rad_boundary_ratios, result.alpha_P);
	}
	AMREX_ASSERT(!result.kappaP.hasnan());
	AMREX_ASSERT(!result.kappaE.hasnan());
	for (int g = 0; g < nGroups_; ++g) {
		if (result.kappaE[g] > 0.0) {
			result.kappaPoverE[g] = result.kappaP[g] / result.kappaE[g];
		} else {
			result.kappaPoverE[g] = 1.0;
		}
	}

	return result;
}

// Compute kappaF and the delta_nu_kappa_B_at_edge term. kappaF is used to compute the work term and the delta_nu_kappa_B_at_edge term is used to compute the
// transport between groups in the momentum function. Only the last two arguments (kappaFVec, delta_nu_kappa_B_at_edge) are modified in this function.
template <typename problem_t>
AMREX_GPU_DEVICE void
RadSystem<problem_t>::ComputeModelDependentKappaFAndDeltaTerms(double const T, double const rho, amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries,
							       quokka::valarray<double, nGroups_> const &fourPiBoverC, OpacityTerms<problem_t> &opacity_terms)
{
	amrex::GpuArray<double, nGroups_> delta_nu_B_at_edge{};
	const auto kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(rad_boundaries, rho, T);
	for (int g = 0; g < nGroups_; ++g) {
		auto const nu_L = rad_boundaries[g];
		auto const nu_R = rad_boundaries[g + 1];
		auto const B_L = PlanckFunction(nu_L, T); // 4 pi B(nu) / c
		auto const B_R = PlanckFunction(nu_R, T); // 4 pi B(nu) / c
		auto const kappa_L = kappa_expo_and_lower_value[1][g];
		auto const kappa_R = kappa_L * std::pow(nu_R / nu_L, kappa_expo_and_lower_value[0][g]);
		opacity_terms.delta_nu_kappa_B_at_edge[g] = nu_R * kappa_R * B_R - nu_L * kappa_L * B_L;
		delta_nu_B_at_edge[g] = nu_R * B_R - nu_L * B_L;
	}
	if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
		opacity_terms.kappaF = opacity_terms.kappaP;
	} else {
		if constexpr (use_diffuse_flux_mean_opacity) {
			opacity_terms.kappaF =
			    ComputeDiffusionFluxMeanOpacity(opacity_terms.kappaP, opacity_terms.kappaE, fourPiBoverC, opacity_terms.delta_nu_kappa_B_at_edge,
							    delta_nu_B_at_edge, kappa_expo_and_lower_value[0]);
		} else {
			// for simplicity, I assume kappaF = kappaE when opacity_model_ ==
			// OpacityModel::PPL_opacity_full_spectrum, if !use_diffuse_flux_mean_opacity. We won't use this
			// option anyway.
			opacity_terms.kappaF = opacity_terms.kappaE;
		}
	}
}

// Compute the Jacobian of energy update equations for the gas-radiation system. The result is a struct containing the following elements:
// J00: (0, 0) component of the Jacobian matrix. = d F0 / d Egas
// F0: (0) component of the residual. = Egas residual
// Fg_abs_sum: sum of the absolute values of the each component of Fg that has tau(g) > 0
// J0g: (0, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d F0 / d R_g
// Jg0: (g, 0) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d Egas
// Jgg: (g, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d R_g
// Fg: (g) components of the residual, g = 1, 2, ..., nGroups. = Erad residual
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGas(double /*T_d*/, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff,
								  quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src,
								  quokka::valarray<double, nGroups_> const &tau, double c_v,
								  quokka::valarray<double, nGroups_> const &kappaPoverE,
								  quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double const num_den,
								  double const dt) -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	const double cscale = c_light_ / c_hat_;

	// CR_heating term
	const double CR_heating = DefineCosmicRayHeatingRate(num_den) * dt;

	result.F0 = Egas_diff + cscale * sum(Rvec) - CR_heating;
	result.Fg = Erad_diff - (Rvec + Src);
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		}
	}

	// compute Jacobian elements
	// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
	// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

	auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t;

	result.J00 = 1.0;
	result.J0g.fillin(cscale);
	result.Jg0 = 1.0 / c_v * dEg_dT;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * kappaPoverE[g] / tau[g] - 1.0;
		}
	}

	return result;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasRadiationEnergyExchange(
    double const Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double const rho, double const dt,
    amrex::GpuArray<Real, nmscalars_> const &massScalars, int const n_outer_iter, quokka::valarray<double, nGroups_> const &work,
    quokka::valarray<double, nGroups_> const &vel_times_F, quokka::valarray<double, nGroups_> const &Src,
    amrex::GpuArray<double, nGroups_ + 1> const &rad_boundaries, int *p_iteration_counter, int *p_iteration_failure_counter) -> NewtonIterationResult<problem_t>
{
	// 1. Compute energy exchange

	// BEGIN NEWTON-RAPHSON LOOP
	// Define the source term: S = dt chat gamma rho (kappa_P B - kappa_E E) + dt chat c^-2 gamma rho kappa_F v * F_i, where gamma =
	// 1 / sqrt(1 - v^2 / c^2) is the Lorentz factor. Solve for the new radiation energy and gas internal energy using a
	// Newton-Raphson method using the base variables (Egas, D_0, D_1,
	// ...), where D_i = R_i / tau_i^(t) and tau_i^(t) = dt * chat * gamma * rho * kappa_{P,i}^(t) is the optical depth across chat
	// * dt for group i at time t. Compared with the old base (Egas, Erad_0, Erad_1, ...), this new base is more stable and
	// converges faster. Furthermore, the PlanckOpacityTempDerivative term is not needed anymore since we assume d/dT (kappa_P /
	// kappa_E) = 0 in the calculation of the Jacobian. Note that this assumption only affects the convergence rate of the
	// Newton-Raphson iteration and does not affect the result at all once the iteration is converged.
	//
	// The Jacobian of F(E_g, D_i) is
	//
	// dF_G / dE_g = 1
	// dF_G / dD_i = c / chat * tau0_i
	// dF_{D,i} / dE_g = 1 / (chat * C_v) * (kappa_{P,i} / kappa_{E,i}) * d/dT (4 \pi B_i)
	// dF_{D,i} / dD_i = - (1 / (chat * dt * rho * kappa_{E,i}) + 1) * tau0_i = - ((1 / tau_i)(kappa_Pi / kappa_Ei) + 1) * tau0_i

	const double c = c_light_; // make a copy of c_light_ to avoid compiler error "undefined in device code"
	const double chat = c_hat_;
	const double cscale = c / chat;

	const double H_num_den = ComputeNumberDensityH(rho, massScalars);

	// const double Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src));
	double Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src));

	double T_gas = NAN;
	double T_d = NAN; // a dummy dust temperature, T_d = T_gas for gas-only model
	double delta_x = NAN;
	quokka::valarray<double, nGroups_> delta_R{};
	quokka::valarray<double, nGroups_> Rvec{};
	quokka::valarray<double, nGroups_> tau0{};	 // optical depth across c * dt at old state
	quokka::valarray<double, nGroups_> tau{};	 // optical depth across c * dt at new state
	quokka::valarray<double, nGroups_> work_local{}; // work term used in the Newton-Raphson iteration of the current outer iteration
	quokka::valarray<double, nGroups_> fourPiBoverC{};
	amrex::GpuArray<double, nGroups_> rad_boundary_ratios{};
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
	OpacityTerms<problem_t> opacity_terms{};

	// fill kappa_expo_and_lower_value with NAN to get warned when there are uninitialized values
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < nGroups_ + 1; ++j) {
			kappa_expo_and_lower_value[i][j] = NAN;
		}
	}

	if constexpr (!(opacity_model_ == OpacityModel::piecewise_constant_opacity)) {
		for (int g = 0; g < nGroups_; ++g) {
			rad_boundary_ratios[g] = rad_boundaries[g + 1] / rad_boundaries[g];
		}
	}

	// define a list of alpha_quant for the model PPL_opacity_fixed_slope_spectrum
	amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
	if constexpr ((opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) ||
		      (gamma_ == 1.0 && opacity_model_ == OpacityModel::PPL_opacity_full_spectrum)) {
		if constexpr (!special_edge_bin_slopes) {
			for (int g = 0; g < nGroups_; ++g) {
				alpha_quant_minus_one[g] = -1.0;
			}
		} else {
			alpha_quant_minus_one[0] = 2.0;
			alpha_quant_minus_one[nGroups_ - 1] = -4.0;
			for (int g = 1; g < nGroups_ - 1; ++g) {
				alpha_quant_minus_one[g] = -1.0;
			}
		}
	}

	double Egas_guess = Egas0;
	auto EradVec_guess = Erad0Vec;

	const double resid_tol = 1.0e-11; // 1.0e-15;
	const int maxIter = 100;
	int n = 0;
	for (; n < maxIter; ++n) {
		// 1. Compute dust temperature
		// If the dust model is turned off, ComputeDustTemperature should be a function that returns T_gas.

		T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
		AMREX_ASSERT(T_gas >= 0.);
		T_d = T_gas;

		// 2. Compute kappaP and kappaE at dust temperature

		fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, rad_boundaries);

		opacity_terms = ComputeModelDependentKappaEAndKappaP(T_d, rho, rad_boundaries, rad_boundary_ratios, fourPiBoverC, EradVec_guess, n,
								     opacity_terms.alpha_E, opacity_terms.alpha_P);

		if (n == 0) {
			// Compute kappaF and the delta_nu_kappa_B term. kappaF is used to compute the work term.
			// Will update opacity_terms in place
			ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms); // update opacity_terms in place
		}

		// 3. In the first loop, calculate kappaF, work, tau0, R

		if (n == 0) {

			if constexpr ((beta_order_ == 1) && (include_work_term_in_source)) {
				// compute the work term at the old state
				// const double gamma = 1.0 / sqrt(1.0 - vsqr / (c * c));
				if (n_outer_iter == 0) {
					for (int g = 0; g < nGroups_; ++g) {
						if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
							work_local[g] = vel_times_F[g] * opacity_terms.kappaF[g] * chat / (c * c) * dt;
						} else {
							kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(rad_boundaries, rho, T_d);
							work_local[g] = vel_times_F[g] * opacity_terms.kappaF[g] * chat / (c * c) * dt *
									(1.0 + kappa_expo_and_lower_value[0][g]);
						}
					}
				} else {
					// If n_outer_iter > 0, use the work term from the previous outer iteration, which is passed as the parameter 'work'
					work_local = work;
				}
			} else {
				work_local.fillin(0.0);
			}

			tau0 = dt * rho * opacity_terms.kappaP * chat;
			tau = tau0;
			Rvec = (fourPiBoverC - EradVec_guess / opacity_terms.kappaPoverE) * tau0 + work_local;
			if constexpr (use_D_as_base) {
				// tau0 is used as a scaling factor for Rvec
				for (int g = 0; g < nGroups_; ++g) {
					if (tau0[g] <= 1.0) {
						tau0[g] = 1.0;
					}
				}
			}
		} else { // in the second and later loops, calculate tau and E (given R)
			tau = dt * rho * opacity_terms.kappaP * chat;
			for (int g = 0; g < nGroups_; ++g) {
				// If tau = 0.0, Erad_guess shouldn't change
				if (tau[g] > 0.0) {
					EradVec_guess[g] = opacity_terms.kappaPoverE[g] * (fourPiBoverC[g] - (Rvec[g] - work_local[g]) / tau[g]);
					if constexpr (force_rad_floor_in_iteration) {
						if (EradVec_guess[g] < 0.0) {
							Egas_guess -= cscale * (Erad_floor_ - EradVec_guess[g]);
							EradVec_guess[g] = Erad_floor_;
						}
					}
				}
			}
		}

		const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, rad_boundaries);
		AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());
		const double c_v = quokka::EOS<problem_t>::ComputeEintTempDerivative(rho, T_gas, massScalars); // Egas = c_v * T

		const auto Egas_diff = Egas_guess - Egas0;
		const auto Erad_diff = EradVec_guess - Erad0Vec;

		auto jacobian =
		    ComputeJacobianForGas(T_d, Egas_diff, Erad_diff, Rvec, Src, tau, c_v, opacity_terms.kappaPoverE, d_fourpiboverc_d_t, H_num_den, dt);

		if constexpr (use_D_as_base) {
			jacobian.J0g = jacobian.J0g * tau0;
			jacobian.Jgg = jacobian.Jgg * tau0;
		}

		// check relative convergence of the residuals
		if ((std::abs(jacobian.F0 / Etot0) < resid_tol) && (cscale * jacobian.Fg_abs_sum / Etot0 < resid_tol)) {
			break;
		}

#if 0
		// For debugging: print (Egas0, Erad0Vec, tau0), which defines the initial condition for a Newton-Raphson iteration
		if (n == 0) {
			std::cout << "Egas0 = " << Egas0 << ", Erad0Vec = [";
			for (int g = 0; g < nGroups_; ++g) {
				std::cout << Erad0Vec[g] << ", ";
			}
			std::cout << "], tau0 = [";
			for (int g = 0; g < nGroups_; ++g) {
				std::cout << tau0[g] << ", ";
			}
			std::cout << "]";
			std::cout << "; C_V = " << c_v << ", a_rad = " << radiation_constant_ << ", coeff_n = " << coeff_n << "\n";
		} else if (n >= 0) {
			std::cout << "n = " << n << ", Egas_guess = " << Egas_guess << ", EradVec_guess = [";
			for (int g = 0; g < nGroups_; ++g) {
				std::cout << EradVec_guess[g] << ", ";
			}
			std::cout << "], tau = [";
			for (int g = 0; g < nGroups_; ++g) {
				std::cout << tau[g] << ", ";
			}
			std::cout << "]";
			std::cout << ", F_G = " << jacobian.F0 << ", F_D_abs_sum = " << jacobian.Fg_abs_sum << ", Etot0 = " << Etot0 << "\n";
		}
#endif

		// update variables
		RadSystem<problem_t>::SolveLinearEqs(jacobian, delta_x, delta_R); // This is modify delta_x and delta_R in place
		AMREX_ASSERT(!std::isnan(delta_x));
		AMREX_ASSERT(!delta_R.hasnan());

		// Update independent variables (Egas_guess, Rvec)
		// enable_dE_constrain is used to prevent the gas temperature from dropping/increasing below/above the radiation
		// temperature
		const double T_rad = std::sqrt(std::sqrt(sum(EradVec_guess) / radiation_constant_));
		if (enable_dE_constrain && delta_x / c_v > std::max(T_gas, T_rad)) {
			Egas_guess = quokka::EOS<problem_t>::ComputeEintFromTgas(rho, T_rad);
			// Rvec.fillin(0.0);
		} else {
			Egas_guess += delta_x;
			if constexpr (use_D_as_base) {
				Rvec += tau0 * delta_R;
			} else {
				Rvec += delta_R;
			}
		}

		// check relative and absolute convergence of E_r
		// if (std::abs(deltaEgas / Egas_guess) < 1e-7) {
		// 	break;
		// }
	} // END NEWTON-RAPHSON LOOP

	AMREX_ASSERT(Egas_guess > 0.0);
	AMREX_ASSERT(min(EradVec_guess) >= 0.0);

	AMREX_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration failed to converge!");
	if (n >= maxIter) {
		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
	}

	amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);	 // total number of radiation updates. NOLINT
	amrex::Gpu::Atomic::Add(&p_iteration_counter[1], n + 1); // total number of Newton-Raphson iterations. NOLINT
	amrex::Gpu::Atomic::Max(&p_iteration_counter[2], n + 1); // maximum number of Newton-Raphson iterations. NOLINT

	NewtonIterationResult<problem_t> result;

	if (n > 0) {
		// calculate kappaF since the temperature has changed
		// Will update opacity_terms in place
		ComputeModelDependentKappaFAndDeltaTerms(T_d, rho, rad_boundaries, fourPiBoverC, opacity_terms); // update opacity_terms in place
	}

	result.Egas = Egas_guess;
	result.EradVec = EradVec_guess;
	result.work = work_local;
	result.T_gas = T_gas;
	result.T_d = T_d;
	result.opacity_terms = opacity_terms;
	return result;
}

// Update radiation flux and gas momentum. Returns FluxUpdateResult struct. The function also updates energy.Egas and energy.work.
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::UpdateFlux(int const i, int const j, int const k, arrayconst_t &consPrev, NewtonIterationResult<problem_t> &energy,
						       double const dt, double const gas_update_factor, double const Ekin0) -> FluxUpdateResult<problem_t>
{
	amrex::GpuArray<amrex::Real, 3> Frad_t0{};
	amrex::GpuArray<amrex::Real, 3> dMomentum{0., 0., 0.};
	amrex::GpuArray<amrex::GpuArray<amrex::Real, nGroups_>, 3> Frad_t1{};

	// make a copy of radBoundaries_
	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	double const rho = consPrev(i, j, k, gasDensity_index);
	const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
	const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
	const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
	const std::array<double, 3> gasMtm0 = {x1GasMom0, x2GasMom0, x3GasMom0};

	auto const fourPiBoverC = ComputeThermalRadiationMultiGroup(energy.T_d, radBoundaries_g);
	auto const kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g, rho, energy.T_d);

	const double chat = c_hat_;

	for (int g = 0; g < nGroups_; ++g) {
		Frad_t0[0] = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
		Frad_t0[1] = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
		Frad_t0[2] = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);

		if constexpr ((gamma_ == 1.0) || (beta_order_ == 0)) {
			for (int n = 0; n < 3; ++n) {
				Frad_t1[n][g] = Frad_t0[n] / (1.0 + rho * energy.opacity_terms.kappaF[g] * chat * dt);
				// Compute conservative gas momentum update
				dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c_light_ * chat);
			}
		} else {
			const auto erad = energy.EradVec[g];
			std::array<double, 3> v_terms{};

			auto fx = Frad_t0[0] / (c_light_ * erad);
			auto fy = Frad_t0[1] / (c_light_ * erad);
			auto fz = Frad_t0[2] / (c_light_ * erad);
			double F_coeff = chat * rho * energy.opacity_terms.kappaF[g] * dt;
			auto Tedd = ComputeEddingtonTensor(fx, fy, fz);

			for (int n = 0; n < 3; ++n) {
				// compute thermal radiation term
				double Planck_term = NAN;

				if constexpr (include_delta_B) {
					Planck_term =
					    energy.opacity_terms.kappaP[g] * fourPiBoverC[g] - 1.0 / 3.0 * energy.opacity_terms.delta_nu_kappa_B_at_edge[g];
				} else {
					Planck_term = energy.opacity_terms.kappaP[g] * fourPiBoverC[g];
				}

				Planck_term *= chat * dt * gasMtm0[n];

				// compute radiation pressure
				double pressure_term = 0.0;
				for (int z = 0; z < 3; ++z) {
					pressure_term += gasMtm0[z] * Tedd[n][z] * erad;
				}
				// Simplification: assuming Eddington tensors are the same for all groups, we have kappaP = kappaE
				if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
					pressure_term *= chat * dt * energy.opacity_terms.kappaE[g];
				} else {
					pressure_term *= chat * dt * (1.0 + kappa_expo_and_lower_value[0][g]) * energy.opacity_terms.kappaE[g];
				}

				v_terms[n] = Planck_term + pressure_term;
			}

			for (int n = 0; n < 3; ++n) {
				// Compute flux update
				Frad_t1[n][g] = (Frad_t0[n] + v_terms[n]) / (1.0 + F_coeff);

				// Compute conservative gas momentum update
				dMomentum[n] += -(Frad_t1[n][g] - Frad_t0[n]) / (c_light_ * chat);
			}
		}
	}

	amrex::Real x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0];
	amrex::Real x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1];
	amrex::Real x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2];

	FluxUpdateResult<problem_t> updated_flux;

	for (int g = 0; g < nGroups_; ++g) {
		updated_flux.Erad[g] = energy.EradVec[g];
	}

	// 3. Deal with the work term.
	if constexpr ((gamma_ != 1.0) && (beta_order_ == 1)) {
		// compute difference in gas kinetic energy before and after momentum update
		amrex::Real const Egastot1 = ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, energy.Egas);
		amrex::Real const Ekin1 = Egastot1 - energy.Egas;
		amrex::Real const dEkin_work = Ekin1 - Ekin0;

		if constexpr (include_work_term_in_source) {
			// New scheme: the work term is included in the source terms. The work done by radiation went to internal energy, but it
			// should go to the kinetic energy. Remove the work term from internal energy.
			energy.Egas -= dEkin_work;
			// The work term is included in the source term, but it is lagged. We update the work term here.
			for (int g = 0; g < nGroups_; ++g) {
				// compute new work term from the updated radiation flux and velocity
				// work = v * F * chi
				if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
					energy.work[g] = (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]) *
							 energy.opacity_terms.kappaF[g] * chat / (c_light_ * c_light_) * dt;
				} else if constexpr (opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum ||
						     opacity_model_ == OpacityModel::PPL_opacity_full_spectrum) {
					energy.work[g] = (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]) *
							 (1.0 + kappa_expo_and_lower_value[0][g]) * energy.opacity_terms.kappaF[g] * chat /
							 (c_light_ * c_light_) * dt;
				}
			}
		} else {
			// Old scheme: the source term does not include the work term, so we add the work term to the Erad.

			// compute loss of radiation energy to gas kinetic energy
			auto dErad_work = -(c_hat_ / c_light_) * dEkin_work;

			// apportion dErad_work according to kappaF_i * (v * F_i)
			quokka::valarray<double, nGroups_> energyLossFractions{};
			if constexpr (nGroups_ == 1) {
				energyLossFractions[0] = 1.0;
			} else {
				// compute energyLossFractions
				for (int g = 0; g < nGroups_; ++g) {
					energyLossFractions[g] = energy.opacity_terms.kappaF[g] *
								 (x1GasMom1 * Frad_t1[0][g] + x2GasMom1 * Frad_t1[1][g] + x3GasMom1 * Frad_t1[2][g]);
				}
				auto energyLossFractionsTot = sum(energyLossFractions);
				if (energyLossFractionsTot != 0.0) {
					energyLossFractions /= energyLossFractionsTot;
				} else {
					energyLossFractions.fillin(0.0);
				}
			}
			for (int g = 0; g < nGroups_; ++g) {
				auto radEnergyNew = energy.EradVec[g] + dErad_work * energyLossFractions[g];
				// AMREX_ASSERT(radEnergyNew > 0.0);
				if (radEnergyNew < Erad_floor_) {
					// return energy to Egas_guess
					energy.Egas -= (Erad_floor_ - radEnergyNew) * (c_light_ / c_hat_);
					radEnergyNew = Erad_floor_;
				}
				updated_flux.Erad[g] = radEnergyNew;
			}
		}
	}

	x1GasMom1 = consPrev(i, j, k, x1GasMomentum_index) + dMomentum[0] * gas_update_factor;
	x2GasMom1 = consPrev(i, j, k, x2GasMomentum_index) + dMomentum[1] * gas_update_factor;
	x3GasMom1 = consPrev(i, j, k, x3GasMomentum_index) + dMomentum[2] * gas_update_factor;
	updated_flux.gasMomentum = {x1GasMom1, x2GasMom1, x3GasMom1};
	updated_flux.Frad = Frad_t1;

	return updated_flux;
}

template <typename problem_t>
void RadSystem<problem_t>::AddSourceTermsMultiGroup(array_t &consVar, arrayconst_t &radEnergySource, amrex::Box const &indexRange, amrex::Real dt_radiation,
						    const int stage, double dustGasCoeff, int *p_iteration_counter, int *p_iteration_failure_counter)
{
	static_assert(beta_order_ == 0 || beta_order_ == 1);

	arrayconst_t &consPrev = consVar; // make read-only
	array_t &consNew = consVar;
	auto dt = dt_radiation;
	if (stage == 2) {
		dt = (1.0 - IMEX_a32) * dt_radiation;
	}

	amrex::GpuArray<amrex::Real, nGroups_ + 1> radBoundaries_g = radBoundaries_;

	// Add source terms

	// 1. Compute gas energy and radiation energy update following Howell &
	// Greenough [Journal of Computational Physics 184 (2003) 53–78].

	// cell-centered kernel
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// make a local reference
		auto p_iteration_counter_local = p_iteration_counter;		      // NOLINT
		auto p_iteration_failure_counter_local = p_iteration_failure_counter; // NOLINT

		const double c = c_light_;
		const double chat = c_hat_;
		const double dustGasCoeff_local = dustGasCoeff;

		// load fluid properties
		const double rho = consPrev(i, j, k, gasDensity_index);
		const double x1GasMom0 = consPrev(i, j, k, x1GasMomentum_index);
		const double x2GasMom0 = consPrev(i, j, k, x2GasMomentum_index);
		const double x3GasMom0 = consPrev(i, j, k, x3GasMomentum_index);
		const double Egastot0 = consPrev(i, j, k, gasEnergy_index);
		auto massScalars = RadSystem<problem_t>::ComputeMassScalars(consPrev, i, j, k);

		// load radiation energy
		quokka::valarray<double, nGroups_> Erad0Vec;
		for (int g = 0; g < nGroups_; ++g) {
			Erad0Vec[g] = consPrev(i, j, k, radEnergy_index + numRadVars_ * g);
		}
		AMREX_ASSERT(min(Erad0Vec) > 0.0);
		const double Erad0 = sum(Erad0Vec);

		// load radiation energy source term
		// plus advection source term (for well-balanced/SDC integrators)
		quokka::valarray<double, nGroups_> Src;
		for (int g = 0; g < nGroups_; ++g) {
			Src[g] = dt * (chat * radEnergySource(i, j, k, g));
		}

		double Egas0 = NAN;
		double Ekin0 = NAN;
		double Etot0 = NAN;
		double Egas_guess = NAN;
		quokka::valarray<double, nGroups_> work{};
		quokka::valarray<double, nGroups_> work_prev{};

		if constexpr (gamma_ != 1.0) {
			Egas0 = ComputeEintFromEgas(rho, x1GasMom0, x2GasMom0, x3GasMom0, Egastot0);
			Etot0 = Egas0 + (c / chat) * (Erad0 + sum(Src));
			Ekin0 = Egastot0 - Egas0;
		}

		// make a copy of radBoundaries_g
		amrex::GpuArray<double, nGroups_ + 1> radBoundaries_g_copy{};
		amrex::GpuArray<double, nGroups_> radBoundaryRatios_copy{};
		for (int g = 0; g < nGroups_ + 1; ++g) {
			radBoundaries_g_copy[g] = radBoundaries_g[g];
		}
		for (int g = 0; g < nGroups_; ++g) {
			radBoundaryRatios_copy[g] = radBoundaries_g_copy[g + 1] / radBoundaries_g_copy[g];
		}

		// define a list of alpha_quant for the model PPL_opacity_fixed_slope_spectrum
		amrex::GpuArray<double, nGroups_> alpha_quant_minus_one{};
		if constexpr ((opacity_model_ == OpacityModel::PPL_opacity_fixed_slope_spectrum) ||
			      (gamma_ == 1.0 && opacity_model_ == OpacityModel::PPL_opacity_full_spectrum)) {
			if constexpr (!special_edge_bin_slopes) {
				for (int g = 0; g < nGroups_; ++g) {
					alpha_quant_minus_one[g] = -1.0;
				}
			} else {
				alpha_quant_minus_one[0] = 2.0;
				alpha_quant_minus_one[nGroups_ - 1] = -4.0;
				for (int g = 1; g < nGroups_ - 1; ++g) {
					alpha_quant_minus_one[g] = -1.0;
				}
			}
		}

		amrex::Real gas_update_factor = 1.0;
		if (stage == 1) {
			gas_update_factor = IMEX_a32;
		}

		const double H_num_den = ComputeNumberDensityH(rho, massScalars);
		const double cscale = c / chat;
		double coeff_n = NAN;
		if constexpr (enable_dust_gas_thermal_coupling_model_) {
			coeff_n = dt * dustGasCoeff_local * H_num_den * H_num_den / cscale;
		}

		// Outer iteration loop to update the work term until it converges
		const int max_iter = 5;
		int iter = 0;
		for (; iter < max_iter; ++iter) {
			amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> kappa_expo_and_lower_value{};
			NewtonIterationResult<problem_t> updated_energy;

			// 1. Compute matter-radiation energy exchange for non-isothermal gas

			if constexpr (gamma_ != 1.0) {

				// 1.2. Compute a term required to calculate the work. This is only required in the first outer loop.

				quokka::valarray<double, nGroups_> vel_times_F{};
				if constexpr (include_work_term_in_source) {
					if (iter == 0) {
						for (int g = 0; g < nGroups_; ++g) {
							// Compute vel_times_F[g] = sum(vel * F_g)
							const double frad0 = consPrev(i, j, k, x1RadFlux_index + numRadVars_ * g);
							const double frad1 = consPrev(i, j, k, x2RadFlux_index + numRadVars_ * g);
							const double frad2 = consPrev(i, j, k, x3RadFlux_index + numRadVars_ * g);
							vel_times_F[g] = (x1GasMom0 * frad0 + x2GasMom0 * frad1 + x3GasMom0 * frad2);
						}
					}
				}

				// 1.3. Compute the gas and radiation energy update. This also updates the opacities. When iter == 0, this also computes
				// the work term.

				if constexpr (!enable_dust_gas_thermal_coupling_model_) {
					// gas + radiation
					updated_energy =
					    SolveGasRadiationEnergyExchange(Egas0, Erad0Vec, rho, dt, massScalars, iter, work, vel_times_F, Src,
									    radBoundaries_g_copy, p_iteration_counter_local, p_iteration_failure_counter_local);
				} else {
					if constexpr (!enable_photoelectric_heating_) {
						// gas + radiation + dust
						updated_energy = SolveGasDustRadiationEnergyExchange(
						    Egas0, Erad0Vec, rho, coeff_n, dt, massScalars, iter, work, vel_times_F, Src, radBoundaries_g_copy,
						    p_iteration_counter_local, p_iteration_failure_counter_local);
					} else {
						// gas + radiation + dust + photoelectric heating
						updated_energy = SolveGasDustRadiationEnergyExchangeWithPE(
						    Egas0, Erad0Vec, rho, coeff_n, dt, massScalars, iter, work, vel_times_F, Src, radBoundaries_g_copy,
						    p_iteration_counter_local, p_iteration_failure_counter_local);
					}
				}

				Egas_guess = updated_energy.Egas;

				// copy work to work_prev
				for (int g = 0; g < nGroups_; ++g) {
					work_prev[g] = updated_energy.work[g];
				}

				kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, updated_energy.T_d);
			} else { // constexpr (gamma_ == 1.0)
				kappa_expo_and_lower_value = DefineOpacityExponentsAndLowerValues(radBoundaries_g_copy, rho, NAN);
				if constexpr (opacity_model_ == OpacityModel::piecewise_constant_opacity) {
					for (int g = 0; g < nGroups_; ++g) {
						updated_energy.opacity_terms.kappaF[g] = kappa_expo_and_lower_value[1][g];
					}
				} else {
					updated_energy.opacity_terms.kappaF =
					    ComputeGroupMeanOpacity(kappa_expo_and_lower_value, radBoundaryRatios_copy, alpha_quant_minus_one);
				}
			}

			// Erad_guess is the new radiation energy (excluding work term)
			// Egas_guess is the new gas internal energy

			// 2. Compute radiation flux update

			// 2.1. Update flux and gas momentum
			auto updated_flux = UpdateFlux(i, j, k, consPrev, updated_energy, dt, gas_update_factor, Ekin0);

			// 2.2. Check for convergence of the work term
			bool work_converged = true;
			if constexpr ((beta_order_ == 0) || (gamma_ == 1.0) || (!include_work_term_in_source)) {
				// pass
			} else {
				work = updated_energy.work;

				// Check for convergence of the work term
				auto const Egastot1 =
				    ComputeEgasFromEint(rho, updated_flux.gasMomentum[0], updated_flux.gasMomentum[1], updated_flux.gasMomentum[2], Egas_guess);
				const double rel_lag_tol = 1.0e-8;
				const double lag_tol = 1.0e-13;
				double ref_work = rel_lag_tol * sum(abs(work));
				ref_work = std::max(ref_work, lag_tol * Egastot1 / (c_light_ / c_hat_));
				// ref_work = std::max(ref_work, lag_tol * sum(Rvec)); // comment out because Rvec is not accessible here
				if (sum(abs(work - work_prev)) > ref_work) {
					work_converged = false;
				}
			}

			// 3. If converged, store new radiation energy, gas energy
			if (work_converged) {
				consNew(i, j, k, x1GasMomentum_index) = updated_flux.gasMomentum[0];
				consNew(i, j, k, x2GasMomentum_index) = updated_flux.gasMomentum[1];
				consNew(i, j, k, x3GasMomentum_index) = updated_flux.gasMomentum[2];
				for (int g = 0; g < nGroups_; ++g) {
					consNew(i, j, k, radEnergy_index + numRadVars_ * g) = updated_flux.Erad[g];
					consNew(i, j, k, x1RadFlux_index + numRadVars_ * g) = updated_flux.Frad[0][g];
					consNew(i, j, k, x2RadFlux_index + numRadVars_ * g) = updated_flux.Frad[1][g];
					consNew(i, j, k, x3RadFlux_index + numRadVars_ * g) = updated_flux.Frad[2][g];
				}
				if constexpr (gamma_ != 1.0) {
					Egas_guess = updated_energy.Egas;
				}
				break;
			}
		} // end full-step iteration

		AMREX_ASSERT_WITH_MESSAGE(iter < max_iter, "AddSourceTerms iteration failed to converge!");
		if (iter >= max_iter) {
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter_local[2], 1); // NOLINT
		}

		// 4b. Store new radiation energy, gas energy
		// In the first stage of the IMEX scheme, the hydro quantities are updated by a fraction (defined by
		// gas_update_factor) of the time step.
		const auto x1GasMom1 = consNew(i, j, k, x1GasMomentum_index);
		const auto x2GasMom1 = consNew(i, j, k, x2GasMomentum_index);
		const auto x3GasMom1 = consNew(i, j, k, x3GasMomentum_index);

		if constexpr (gamma_ != 1.0) {
			Egas_guess = Egas0 + (Egas_guess - Egas0) * gas_update_factor;
			consNew(i, j, k, gasInternalEnergy_index) = Egas_guess;
			consNew(i, j, k, gasEnergy_index) = ComputeEgasFromEint(rho, x1GasMom1, x2GasMom1, x3GasMom1, Egas_guess);
		} else {
			amrex::ignore_unused(Egas_guess);
			amrex::ignore_unused(Egas0);
			amrex::ignore_unused(Etot0);
			amrex::ignore_unused(work);
			amrex::ignore_unused(work_prev);
		}
	});
}

#endif // RAD_SOURCE_TERMS_MULTI_GROUP_HPP_