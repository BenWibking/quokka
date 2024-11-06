// IWYU pragma: private; include "radiation/radiation_system.hpp"
#ifndef RADIATION_DUST_SYSTEM_HPP_
#define RADIATION_DUST_SYSTEM_HPP_

#include "radiation/radiation_system.hpp"

template <typename problem_t>
AMREX_GPU_HOST_DEVICE auto RadSystem<problem_t>::DefinePhotoelectricHeatingE1Derivative(amrex::Real const /*temperature*/, amrex::Real const /*num_density*/)
    -> amrex::Real
{
	return 0.0;
}

// Compute the Jacobian of energy update equations for the gas-dust-radiation system. The result is a struct containing the following elements:
// J00: (0, 0) component of the Jacobian matrix. = d F0 / d Egas
// F0: (0) component of the residual. = Egas residual
// Fg_abs_sum: sum of the absolute values of the each component of Fg that has tau(g) > 0
// J0g: (0, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d F0 / d R_g
// Jg0: (g, 0) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d Egas
// Jgg: (g, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d R_g
// Fg: (g) components of the residual, g = 1, 2, ..., nGroups. = Erad residual
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDust(
    double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec,
    quokka::valarray<double, nGroups_> const &Src, double coeff_n, quokka::valarray<double, nGroups_> const &tau, double c_v, double /*lambda_gd_time_dt*/,
    quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, const double num_den, const double dt)
    -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	const double cscale = c_light_ / c_hat_;

	// compute cooling/heating terms
	const auto cooling = DefineNetCoolingRate(T_gas, num_den) * dt;
	const auto cooling_derivative = DefineNetCoolingRateTempDerivative(T_gas, num_den) * dt;
	const double CR_heating = DefineCosmicRayHeatingRate(num_den) * dt;

	result.F0 = Egas_diff + cscale * sum(Rvec) + sum(cooling) - CR_heating;
	result.Fg = Erad_diff - (Rvec + Src);
	if constexpr (add_line_cooling_to_radiation_in_jac) {
		result.Fg -= (1.0 / cscale) * cooling;
	}
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		} else {
			result.Fg_abs_sum += std::abs(result.Fg[g] + Rvec[g]);
		}
	}

	// compute Jacobian elements
	// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
	// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

	auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t;

	result.J00 = 1.0 + sum(cooling_derivative) / c_v;
	result.J0g.fillin(cscale);
	const double d_Td_d_T = 3. / 2. - T_d / (2. * T_gas);
	dEg_dT *= d_Td_d_T;
	const double dTd_dRg = -1.0 / (coeff_n * std::sqrt(T_gas));
	const auto rg = kappaPoverE * d_fourpiboverc_d_t * dTd_dRg;
	result.Jg0 = 1.0 / c_v * dEg_dT - (1 / cscale) * cooling_derivative - 1.0 / cscale * rg * result.J00;
	// Note that Fg is modified here, but it does not change Fg_abs_sum, which is used to check the convergence.
	result.Fg = result.Fg - 1.0 / cscale * rg * result.F0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * kappaPoverE[g] / tau[g] - 1.0;
		}
	}

	return result;
}

// Compute the Jacobian of energy update equations for the gas-dust-radiation system with gas and dust decoupled. The result is a struct containing the
// following elements: J00: (0, 0) component of the Jacobian matrix. = d F0 / d T_d F0: (0) component of the residual. = sum_g R_g - lambda_gd_time_dt
// Fg_abs_sum: sum of the absolute values of the each component of Fg that has tau(g) > 0
// J0g: (0, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d F0 / d R_g
// Jg0: (g, 0) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d T_d
// Jgg: (g, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d R_g
// Fg: (g) components of the residual, g = 1, 2, ..., nGroups. = Erad residual
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDustDecoupled(
    double /*T_gas*/, double /*T_d*/, double /*Egas_diff*/, quokka::valarray<double, nGroups_> const &Erad_diff, quokka::valarray<double, nGroups_> const &Rvec,
    quokka::valarray<double, nGroups_> const &Src, double /*coeff_n*/, quokka::valarray<double, nGroups_> const &tau, double /*c_v*/, double lambda_gd_time_dt,
    quokka::valarray<double, nGroups_> const &kappaPoverE, quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t) -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	result.F0 = -lambda_gd_time_dt + sum(Rvec);
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

	result.J00 = 0.0;
	result.J0g.fillin(1.0);
	result.Jg0 = dEg_dT;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] <= 0.0) {
			result.Jgg[g] = -std::numeric_limits<double>::infinity();
		} else {
			result.Jgg[g] = -1.0 * kappaPoverE[g] / tau[g] - 1.0;
		}
	}

	return result;
}

// Compute the Jacobian of energy update equations for the gas-dust-radiation system. The result is a struct containing the following elements:
// J00: (0, 0) component of the Jacobian matrix. = d F0 / d Egas
// F0: (0) component of the residual. = Egas residual
// Fg_abs_sum: sum of the absolute values of the each component of Fg that has tau(g) > 0
// J0g: (0, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d F0 / d R_g
// Jg0: (g, 0) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d Egas
// Jgg: (g, g) components of the Jacobian matrix, g = 1, 2, ..., nGroups. = d Fg / d R_g
// Fg: (g) components of the residual, g = 1, 2, ..., nGroups. = Erad residual
template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::ComputeJacobianForGasAndDustWithPE(
    double T_gas, double T_d, double Egas_diff, quokka::valarray<double, nGroups_> const &Erad, quokka::valarray<double, nGroups_> const &Erad0,
    double PE_heating_energy_derivative, quokka::valarray<double, nGroups_> const &Rvec, quokka::valarray<double, nGroups_> const &Src, double coeff_n,
    quokka::valarray<double, nGroups_> const &tau, double c_v, double /*lambda_gd_time_dt*/, quokka::valarray<double, nGroups_> const &kappaPoverE,
    quokka::valarray<double, nGroups_> const &d_fourpiboverc_d_t, double const num_den, double const dt) -> JacobianResult<problem_t>
{
	JacobianResult<problem_t> result;

	const double cscale = c_light_ / c_hat_;

	// compute cooling/heating terms
	const auto cooling = DefineNetCoolingRate(T_gas, num_den) * dt;
	const auto cooling_derivative = DefineNetCoolingRateTempDerivative(T_gas, num_den) * dt;
	const double CR_heating = DefineCosmicRayHeatingRate(num_den) * dt;

	result.F0 = Egas_diff + cscale * sum(Rvec) + sum(cooling) - PE_heating_energy_derivative * Erad[nGroups_ - 1] - CR_heating;
	result.Fg = Erad - Erad0 - (Rvec + Src);
	if constexpr (add_line_cooling_to_radiation_in_jac) {
		result.Fg -= (1.0 / cscale) * cooling;
	}
	result.Fg_abs_sum = 0.0;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] > 0.0) {
			result.Fg_abs_sum += std::abs(result.Fg[g]);
		} else {
			result.Fg_abs_sum += std::abs(result.Fg[g] + Rvec[g]);
		}
	}

	// const auto d_fourpiboverc_d_t = ComputeThermalRadiationTempDerivativeMultiGroup(T_d, radBoundaries_g_copy);
	AMREX_ASSERT(!d_fourpiboverc_d_t.hasnan());

	// compute Jacobian elements
	// I assume (kappaPVec / kappaEVec) is constant here. This is usually a reasonable assumption. Note that this assumption
	// only affects the convergence rate of the Newton-Raphson iteration and does not affect the converged solution at all.

	auto d_Eg_d_Rg = -1.0 * kappaPoverE;
	for (int g = 0; g < nGroups_; ++g) {
		if (tau[g] <= 0.0) {
			d_Eg_d_Rg[g] = -LARGE;
		} else {
			d_Eg_d_Rg[g] /= tau[g];
		}
	}

	result.J00 = 1.0 + sum(cooling_derivative) / c_v;
	result.J0g.fillin(cscale);
	result.J0g[nGroups_ - 1] -= PE_heating_energy_derivative * d_Eg_d_Rg[nGroups_ - 1];
	const double d_Td_d_T = 3. / 2. - T_d / (2. * T_gas);
	const auto dEg_dT = kappaPoverE * d_fourpiboverc_d_t * d_Td_d_T;
	const double dTd_dRg = -1.0 / (coeff_n * std::sqrt(T_gas));
	const auto rg = kappaPoverE * d_fourpiboverc_d_t * dTd_dRg;
	result.Jg0 = 1.0 / c_v * dEg_dT - (1 / cscale) * cooling_derivative - 1.0 / cscale * rg * result.J00;
	// Note that Fg is modified here, but it does not change Fg_abs_sum, which is used to check the convergence.
	result.Fg = result.Fg - 1.0 / cscale * rg * result.F0;
	result.Jgg = d_Eg_d_Rg + (-1.0);
	result.Jgg[nGroups_ - 1] += rg[nGroups_ - 1] - (rg[nGroups_ - 1] / cscale) * PE_heating_energy_derivative * d_Eg_d_Rg[nGroups_ - 1];
	result.Jg1 = rg - 1.0 / cscale * rg * result.J0g[nGroups_ - 1]; // note that this is the (nGroups_ - 1)th column, except for the (nGroups_ - 1)th row

	return result;
}

// Linear equation solver for matrix with non-zeros at the first row, first column, and diagonal only.
// solve the linear system
//   [a00 a0i] [x0] = [y0]
//   [ai0 aii] [xi]   [yi]
// for x0 and xi, where a0i = (a01, a02, a03, ...); ai0 = (a10, a20, a30, ...); aii = (a11, a22, a33, ...), xi = (x1, x2, x3, ...), yi = (y1, y2, y3, ...)
template <typename problem_t>
AMREX_GPU_HOST_DEVICE void RadSystem<problem_t>::SolveLinearEqsWithLastColumn(JacobianResult<problem_t> const &jacobian, double &x0,
									      quokka::valarray<double, nGroups_> &xi)
{
	// Note that the following routine only works when the FUV group is the last group, i.e., pe_index = nGroups_ - 1

	// note that jacobian.Jgg[pe_index] is the right value for a[pe_index][pe_index], while jacobian.Jg1[pe_index] is NOT.
	const int pe_index = nGroups_ - 1;
	const auto ratios = jacobian.J0g / jacobian.Jgg;

	const auto a00_new = jacobian.J00 - sum(ratios * jacobian.Jg0);
	const auto y0_new = jacobian.F0 - sum(ratios * jacobian.Fg);
	auto a01_new = jacobian.J0g[pe_index] - sum(ratios * jacobian.Jg1);
	// re-accounting for the pe_index'th element of jacobian.Jg1
	a01_new = a01_new + ratios[pe_index] * jacobian.Jg1[pe_index] - ratios[pe_index] * jacobian.Jgg[pe_index];
	const auto a10 = jacobian.Jg0[pe_index];
	const auto a11 = jacobian.Jgg[pe_index];
	const auto y1 = jacobian.Fg[pe_index];
	// solve linear equations [[a00_new, a01_new], [a10, a11]] [[x0], [xi[pe_index]]] = [y0_new, y1]
	x0 = (y0_new - a01_new / a11 * y1) / (a00_new - a01_new / a11 * a10);
	const auto x1 = (y1 - a10 * x0) / a11;
	xi[pe_index] = x1;
	// xi = (jacobian.Fg - jacobian.Jg0 * x0) / jacobian.Jgg;
	for (int g = 0; g < pe_index; ++g) {
		xi[g] = (jacobian.Fg[g] - jacobian.Jg0[g] * x0 - jacobian.Jg1[g] * x1) / jacobian.Jgg[g];
	}
	x0 *= -1.0;
	xi *= -1.0;
}

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasDustRadiationEnergyExchange(
    double const Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double const rho, double const coeff_n, double const dt,
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

	int dust_model = 1;
	double T_d0 = NAN;
	double lambda_gd_times_dt = NAN;
	const double T_gas0 = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas0, massScalars);
	AMREX_ASSERT(T_gas0 >= 0.);
	T_d0 = ComputeDustTemperatureBateKeto(T_gas0, T_gas0, rho, Erad0Vec, coeff_n, dt, NAN, 0, rad_boundaries);
	AMREX_ASSERT_WITH_MESSAGE(T_d0 >= 0., "Dust temperature is negative!");
	if (T_d0 < 0.0) {
		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
	}

	const double max_Gamma_gd = coeff_n * std::max(std::sqrt(T_gas0) * T_gas0, std::sqrt(T_d0) * T_d0);
	if (cscale * max_Gamma_gd < ISM_Traits<problem_t>::gas_dust_coupling_threshold * Egas0) {
		dust_model = 2;
		lambda_gd_times_dt = coeff_n * std::sqrt(T_gas0) * (T_gas0 - T_d0);
	}

	// const double Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src));
	double Etot0 = NAN;
	if (dust_model == 1) {
		Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src));
	} else {
		// for dust_model == 2 (decoupled gas and dust), Egas0 is not involved in the iteration
		const double fourPiBoverC = sum(ComputeThermalRadiationMultiGroup(T_d0, rad_boundaries));
		Etot0 = std::abs(lambda_gd_times_dt) + fourPiBoverC + (sum(Erad0Vec) + sum(Src));
	}

	double T_gas = NAN;
	double T_d = NAN;
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

	const double H_num_den = ComputeNumberDensityH(rho, massScalars);

	T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
	AMREX_ASSERT(T_gas >= 0.);

	const double resid_tol = 1.0e-11; // 1.0e-15;
	const int maxIter = 100;
	int n = 0;
	for (; n < maxIter; ++n) {
		// 1. Compute dust temperature
		// If the dust model is turned off, ComputeDustTemperature should be a function that returns T_gas.

		if (n > 0) {
			T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
			AMREX_ASSERT(T_gas >= 0.);
		}

		if (dust_model == 1) {
			if (n == 0) {
				T_d = T_d0;
			} else {
				T_d = T_gas - sum(Rvec) / (coeff_n * std::sqrt(T_gas));
			}
		} else {
			if (n == 0) {
				T_d = T_d0;
			}
		}
		AMREX_ASSERT_WITH_MESSAGE(T_d >= 0., "Dust temperature is negative! Consider increasing ISM_Traits::gas_dust_coupling_threshold");
		if (T_d < 0.0) {
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
		}

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

		JacobianResult<problem_t> jacobian;

		if (dust_model == 1) {
			jacobian = ComputeJacobianForGasAndDust(T_gas, T_d, Egas_diff, Erad_diff, Rvec, Src, coeff_n, tau, c_v, lambda_gd_times_dt,
								opacity_terms.kappaPoverE, d_fourpiboverc_d_t, H_num_den, dt);
		} else {
			jacobian = ComputeJacobianForGasAndDustDecoupled(T_gas, T_d, Egas_diff, Erad_diff, Rvec, Src, coeff_n, tau, c_v, lambda_gd_times_dt,
									 opacity_terms.kappaPoverE, d_fourpiboverc_d_t);
		}

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
		if (dust_model == 2) {
			T_d += delta_x;
			Rvec += delta_R;
		} else {
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
		}

		// check relative and absolute convergence of E_r
		// if (std::abs(deltaEgas / Egas_guess) < 1e-7) {
		// 	break;
		// }
	} // END NEWTON-RAPHSON LOOP

	const auto cooling_tend = DefineNetCoolingRate(T_gas, H_num_den) * dt;
	if (dust_model == 2) {
		// include line cooling/heating, cosmic ray heating terms; implicitly update Egas_guess

		const double CR_heating = DefineCosmicRayHeatingRate(H_num_den) * dt;

		const double compare = Egas_guess + cscale * lambda_gd_times_dt + sum(abs(cooling_tend)) + CR_heating;

		// RHS of the equation 0 = Egas - Egas0 + cscale * lambda_gd_times_dt + sum(cooling)
		auto rhs = [=](double Egas_) -> double {
			const double T_gas_ = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
			const auto cooling_ = DefineNetCoolingRate(T_gas_, H_num_den) * dt;
			return Egas_ - Egas0 + cscale * lambda_gd_times_dt + sum(cooling_) - CR_heating;
		};

		// Jacobian of the RHS of the equation 0 = Egas - Egas0 + cscale * lambda_gd_times_dt + sum(cooling)
		auto jac = [=](double Egas_) -> double {
			const double T_gas_ = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
			const auto d_cooling_d_Tgas_ = DefineNetCoolingRateTempDerivative(T_gas_, H_num_den) * dt;
			return 1.0 + sum(d_cooling_d_Tgas_);
		};

		Egas_guess = BackwardEulerOneVariable(rhs, jac, Egas0, compare);
	}

	if constexpr (!add_line_cooling_to_radiation_in_jac) {
		AMREX_ASSERT_WITH_MESSAGE(min(cooling_tend) >= 0., "add_line_cooling_to_radiation has to be enabled when there is negative cooling rate!");
		// TODO(CCH): potential GPU-related issue here.
		EradVec_guess += (1 / cscale) * cooling_tend;
	}

	AMREX_ASSERT(Egas_guess > 0.0);
	AMREX_ASSERT(min(EradVec_guess) >= 0.0);

	AMREX_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration for matter-radiation coupling failed to converge!");
	if (n >= maxIter) {
		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
	}

	amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);	 // total number of radiation updates. NOLINT
	amrex::Gpu::Atomic::Add(&p_iteration_counter[1], n + 1); // total number of Newton-Raphson iterations. NOLINT
	amrex::Gpu::Atomic::Max(&p_iteration_counter[2], n + 1); // maximum number of Newton-Raphson iterations. NOLINT
	if (dust_model == 2) {
		amrex::Gpu::Atomic::Add(&p_iteration_counter[3], 1); // total number of decoupled gas-dust iterations. NOLINT
	}

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

template <typename problem_t>
AMREX_GPU_DEVICE auto RadSystem<problem_t>::SolveGasDustRadiationEnergyExchangeWithPE(
    double const Egas0, quokka::valarray<double, nGroups_> const &Erad0Vec, double const rho, double const coeff_n, double const dt,
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

	int dust_model = 1;
	double T_d0 = NAN;
	double lambda_gd_times_dt = NAN;
	const double T_gas0 = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas0, massScalars);
	AMREX_ASSERT(T_gas0 >= 0.);
	T_d0 = ComputeDustTemperatureBateKeto(T_gas0, T_gas0, rho, Erad0Vec, coeff_n, dt, NAN, 0, rad_boundaries);
	AMREX_ASSERT_WITH_MESSAGE(T_d0 >= 0., "Dust temperature is negative!");
	if (T_d0 < 0.0) {
		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
	}

	const double max_Gamma_gd = coeff_n * std::max(std::sqrt(T_gas0) * T_gas0, std::sqrt(T_d0) * T_d0);
	if (cscale * max_Gamma_gd < ISM_Traits<problem_t>::gas_dust_coupling_threshold * Egas0) {
		dust_model = 2;
		lambda_gd_times_dt = coeff_n * std::sqrt(T_gas0) * (T_gas0 - T_d0);
	}

	// const double Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src));
	double Etot0 = NAN;
	if (dust_model == 1) {
		Etot0 = Egas0 + cscale * (sum(Erad0Vec) + sum(Src));
	} else {
		// for dust_model == 2 (decoupled gas and dust), Egas0 is not involved in the iteration
		Etot0 = std::abs(lambda_gd_times_dt) + (sum(Erad0Vec) + sum(Src));
	}

	double T_gas = NAN;
	double T_d = NAN;
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

	// fill kappa_expo_and_lower_value with NAN to avoid mistakes caused by uninitialized values
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

	T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
	AMREX_ASSERT(T_gas >= 0.);

	// phtoelectric heating
	const double H_num_den = ComputeNumberDensityH(rho, massScalars);
	const double PE_heating_energy_derivative = dt * DefinePhotoelectricHeatingE1Derivative(T_gas, H_num_den);

	const double resid_tol = 1.0e-11; // 1.0e-15;
	const int maxIter = 100;
	int n = 0;
	for (; n < maxIter; ++n) {
		// 1. Compute dust temperature
		// If the dust model is turned off, ComputeDustTemperature should be a function that returns T_gas.

		if (n > 0) {
			T_gas = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_guess, massScalars);
			AMREX_ASSERT(T_gas >= 0.);
		}

		if (dust_model == 1) {
			if (n == 0) {
				T_d = T_d0;
			} else {
				T_d = T_gas - sum(Rvec) / (coeff_n * std::sqrt(T_gas));
			}
		} else {
			if (n == 0) {
				T_d = T_d0;
			}
		}
		AMREX_ASSERT_WITH_MESSAGE(T_d >= 0., "Dust temperature is negative! Consider increasing ISM_Traits::gas_dust_coupling_threshold");
		if (T_d < 0.0) {
			amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[1], 1); // NOLINT
		}

		// 2. Compute kappaP and kappaE at dust temperature

		fourPiBoverC = ComputeThermalRadiationMultiGroup(T_d, rad_boundaries);

		opacity_terms = ComputeModelDependentKappaEAndKappaP(T_d, rho, rad_boundaries, rad_boundary_ratios, fourPiBoverC, EradVec_guess, n,
								     opacity_terms.alpha_E, opacity_terms.alpha_P);

		if (n == 0) {
			// Compute kappaF and the delta_nu_kappa_B term. kappaF is used to compute the work term.
			// Only the last two arguments (kappaFVec, delta_nu_kappa_B_at_edge) are modified in this function.
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

		JacobianResult<problem_t> jacobian;

		if (dust_model == 1) {
			jacobian =
			    ComputeJacobianForGasAndDustWithPE(T_gas, T_d, Egas_diff, EradVec_guess, Erad0Vec, PE_heating_energy_derivative, Rvec, Src, coeff_n,
							       tau, c_v, lambda_gd_times_dt, opacity_terms.kappaPoverE, d_fourpiboverc_d_t, H_num_den, dt);
		} else {
			jacobian = ComputeJacobianForGasAndDustDecoupled(T_gas, T_d, Egas_diff, Erad_diff, Rvec, Src, coeff_n, tau, c_v, lambda_gd_times_dt,
									 opacity_terms.kappaPoverE, d_fourpiboverc_d_t);
		}

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
		RadSystem<problem_t>::SolveLinearEqsWithLastColumn(jacobian, delta_x, delta_R); // This is modify delta_x and delta_R in place
		AMREX_ASSERT(!std::isnan(delta_x));
		AMREX_ASSERT(!delta_R.hasnan());

		// Update independent variables (Egas_guess, Rvec)
		// enable_dE_constrain is used to prevent the gas temperature from dropping/increasing below/above the radiation
		// temperature
		if (dust_model == 2) {
			T_d += delta_x;
			Rvec += delta_R;
		} else {
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
		}

		// check relative and absolute convergence of E_r
		// if (std::abs(deltaEgas / Egas_guess) < 1e-7) {
		// 	break;
		// }
	} // END NEWTON-RAPHSON LOOP

	const auto cooling_tend = DefineNetCoolingRate(T_gas, H_num_den) * dt;
	if (dust_model == 2) {
		// compute cooling/heating terms; implicitly update Egas_guess

		const double CR_heating = DefineCosmicRayHeatingRate(H_num_den) * dt;
		const double compare = Egas_guess + cscale * lambda_gd_times_dt + sum(abs(cooling_tend)) + CR_heating;

		// RHS of the equation 0 = Egas - Egas0 + cscale * lambda_gd_times_dt + sum(cooling) - PE_heating_energy_derivative * EradVec_guess[nGroups_ -
		// 1];
		auto rhs = [=](double Egas_) -> double {
			const double T_gas_ = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
			const auto cooling_ = DefineNetCoolingRate(T_gas_, H_num_den) * dt;
			return Egas_ - Egas0 + cscale * lambda_gd_times_dt + sum(cooling_) - PE_heating_energy_derivative * EradVec_guess[nGroups_ - 1] -
			       CR_heating;
		};

		// Jacobian of the RHS of the equation 0 = Egas - Egas0 + cscale * lambda_gd_times_dt + sum(cooling) + PE_heating_energy_derivative *
		// EradVec_guess[nGroups_ - 1];
		auto jac = [=](double Egas_) -> double {
			const double T_gas_ = quokka::EOS<problem_t>::ComputeTgasFromEint(rho, Egas_, massScalars);
			const auto d_cooling_d_Tgas_ = DefineNetCoolingRateTempDerivative(T_gas_, H_num_den) * dt;
			return 1.0 + sum(d_cooling_d_Tgas_);
		};

		Egas_guess = BackwardEulerOneVariable(rhs, jac, Egas0, compare);
	}

	if constexpr (!add_line_cooling_to_radiation_in_jac) {
		EradVec_guess += (1 / cscale) * cooling_tend;
	}

	AMREX_ASSERT(Egas_guess > 0.0);
	AMREX_ASSERT(min(EradVec_guess) >= 0.0);

	AMREX_ASSERT_WITH_MESSAGE(n < maxIter, "Newton-Raphson iteration failed to converge!");
	if (n >= maxIter) {
		amrex::Gpu::Atomic::Add(&p_iteration_failure_counter[0], 1); // NOLINT
	}

	amrex::Gpu::Atomic::Add(&p_iteration_counter[0], 1);	 // total number of radiation updates. NOLINT
	amrex::Gpu::Atomic::Add(&p_iteration_counter[1], n + 1); // total number of Newton-Raphson iterations. NOLINT
	amrex::Gpu::Atomic::Max(&p_iteration_counter[2], n + 1); // maximum number of Newton-Raphson iterations. NOLINT
	if (dust_model == 2) {
		amrex::Gpu::Atomic::Add(&p_iteration_counter[3], 1); // total number of decoupled gas-dust iterations. NOLINT
	}

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

#endif // RADIATION_DUST_SYSTEM_HPP_