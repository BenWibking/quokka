/// \file test_radhydro_pulse_MG_int.cpp
/// \brief Defines a test problem for multigroup radiation in the diffusion regime with advection by gas using group-integrated opacity.
///

#include "test_radhydro_pulse_MG_int.hpp"
#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"
#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "physics_info.hpp"
#include "radiation/planck_integral.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"

struct MGProblem {
}; // dummy type to allow compile-type polymorphism via template specialization
struct ExactProblem {
};

// constexpr int n_groups_ = 2;
constexpr int n_groups_ = 4;
// constexpr int n_groups_ = 8;
// constexpr int n_groups_ = 16;
// constexpr OpacityModel opacity_model_ = OpacityModel::piecewise_constant_opacity;
constexpr OpacityModel opacity_model_ = OpacityModel::PPL_opacity_fixed_slope_spectrum;
// constexpr OpacityModel opacity_model_ = OpacityModel::PPL_opacity_full_spectrum;

constexpr amrex::GpuArray<double, n_groups_ + 1> rad_boundaries_ = []() constexpr {
	if constexpr (n_groups_ == 2) {
		return amrex::GpuArray<double, 3>{1e15, 1e17, 1e19};
	} else if constexpr (n_groups_ == 4) {
		return amrex::GpuArray<double, 5>{1e15, 1e16, 1e17, 1e18, 1e19};
	} else if constexpr (n_groups_ == 8) {
		return amrex::GpuArray<double, 9>{1e15, 3.16e15, 1e16, 3.16e16, 1e17, 3.16e17, 1e18, 3.16e18, 1e19};
	} else if constexpr (n_groups_ == 16) {
		return amrex::GpuArray<double, 17>{1.00000000e+15, 1.77827941e+15, 3.16227766e+15, 5.62341325e+15, 1.00000000e+16, 1.77827941e+16,
						   3.16227766e+16, 5.62341325e+16, 1.00000000e+17, 1.77827941e+17, 3.16227766e+17, 5.62341325e+17,
						   1.00000000e+18, 1.77827941e+18, 3.16227766e+18, 5.62341325e+18, 1.00000000e+19};
	} else if constexpr (n_groups_ == 32) {
		return amrex::GpuArray<double, 33>{1.00000000e+15, 1.33352143e+15, 1.77827941e+15, 2.37137371e+15, 3.16227766e+15, 4.21696503e+15,
						   5.62341325e+15, 7.49894209e+15, 1.00000000e+16, 1.33352143e+16, 1.77827941e+16, 2.37137371e+16,
						   3.16227766e+16, 4.21696503e+16, 5.62341325e+16, 7.49894209e+16, 1.00000000e+17, 1.33352143e+17,
						   1.77827941e+17, 2.37137371e+17, 3.16227766e+17, 4.21696503e+17, 5.62341325e+17, 7.49894209e+17,
						   1.00000000e+18, 1.33352143e+18, 1.77827941e+18, 2.37137371e+18, 3.16227766e+18, 4.21696503e+18,
						   5.62341325e+18, 7.49894209e+18, 1.00000000e+19};
	} else if constexpr (n_groups_ == 64) {
		return amrex::GpuArray<double, 65>{
		    1.00000000e+15, 1.15478198e+15, 1.33352143e+15, 1.53992653e+15, 1.77827941e+15, 2.05352503e+15, 2.37137371e+15, 2.73841963e+15,
		    3.16227766e+15, 3.65174127e+15, 4.21696503e+15, 4.86967525e+15, 5.62341325e+15, 6.49381632e+15, 7.49894209e+15, 8.65964323e+15,
		    1.00000000e+16, 1.15478198e+16, 1.33352143e+16, 1.53992653e+16, 1.77827941e+16, 2.05352503e+16, 2.37137371e+16, 2.73841963e+16,
		    3.16227766e+16, 3.65174127e+16, 4.21696503e+16, 4.86967525e+16, 5.62341325e+16, 6.49381632e+16, 7.49894209e+16, 8.65964323e+16,
		    1.00000000e+17, 1.15478198e+17, 1.33352143e+17, 1.53992653e+17, 1.77827941e+17, 2.05352503e+17, 2.37137371e+17, 2.73841963e+17,
		    3.16227766e+17, 3.65174127e+17, 4.21696503e+17, 4.86967525e+17, 5.62341325e+17, 6.49381632e+17, 7.49894209e+17, 8.65964323e+17,
		    1.00000000e+18, 1.15478198e+18, 1.33352143e+18, 1.53992653e+18, 1.77827941e+18, 2.05352503e+18, 2.37137371e+18, 2.73841963e+18,
		    3.16227766e+18, 3.65174127e+18, 4.21696503e+18, 4.86967525e+18, 5.62341325e+18, 6.49381632e+18, 7.49894209e+18, 8.65964323e+18,
		    1.00000000e+19};
	}
}();

static constexpr bool export_csv = true;

constexpr double T0 = 1.0e7; // K (temperature)
constexpr double T1 = 2.0e7; // K (temperature)
constexpr double rho0 = 1.2; // g cm^-3 (matter density)
constexpr double a_rad = C::a_rad;
constexpr double c = C::c_light; // speed of light (cgs)
constexpr double chat = c;
constexpr double width = 24.0; // cm, width of the pulse
constexpr double erad_floor = a_rad * T0 * T0 * T0 * T0 * 1.0e-10;
constexpr double mu = 2.33 * C::m_u;
constexpr double h_planck = C::hplanck;
constexpr double k_B = C::k_B;

// static diffusion: (for single group) tau = 2e3, beta = 3e-5, beta tau = 6e-2
constexpr double kappa0 = 180.; // cm^2 g^-1
constexpr double scaleup = 1.;
constexpr double v0_adv = 1.0e6;    // advecting pulse
constexpr double max_time = 4.8e-5; // max_time = 2 * width / v1;
// constexpr double max_time = 2e-5;   // max_time = 2 * width / v1;
constexpr int64_t max_timesteps = 1e2; // to make 3D test run fast on GPUs
// constexpr int64_t max_timesteps = 1e8; // full run

// dynamic diffusion: tau = 2e4, beta = 3e-3, beta tau = 60
// constexpr double kappa0 = 1000.; // cm^2 g^-1
// constexpr double v0_adv = 1.0e8;    // advecting pulse
// constexpr double max_time = 1.2e-4; // max_time = 2.0 * width / v1;

constexpr double T_ref = T0;
constexpr double nu_ref = 1.0e18;			     // Hz
constexpr double coeff_ = h_planck * nu_ref / (k_B * T_ref); // = 4.799243073 = 1 / 0.2083661912

template <> struct quokka::EOS_Traits<MGProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};
template <> struct quokka::EOS_Traits<ExactProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<MGProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};
template <> struct Physics_Traits<ExactProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<MGProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = erad_floor;
	static constexpr bool compute_v_over_c_terms = true;
	static constexpr double energy_unit = h_planck;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = rad_boundaries_;
	static constexpr int beta_order = 1;
	static constexpr OpacityModel opacity_model = opacity_model_;
};
template <> struct RadSystem_Traits<ExactProblem> {
	static constexpr double c_hat_over_c = 1.0;
	static constexpr double Erad_floor = erad_floor;
	static constexpr bool compute_v_over_c_terms = true;
	static constexpr int beta_order = 1;
	static constexpr OpacityModel opacity_model = OpacityModel::single_group;
};

AMREX_GPU_HOST_DEVICE
auto compute_initial_Tgas(const double x) -> double
{
	// compute temperature profile for Gaussian radiation pulse
	const double sigma = width;
	return T0 + (T1 - T0) * std::exp(-x * x / (2.0 * sigma * sigma));
}

AMREX_GPU_HOST_DEVICE
auto compute_exact_rho(const double x) -> double
{
	// compute density profile for Gaussian radiation pulse
	auto T = compute_initial_Tgas(x);
	return rho0 * T0 / T + (a_rad * mu / 3. / k_B) * (std::pow(T0, 4) / T - std::pow(T, 3));
}

AMREX_GPU_HOST_DEVICE
auto compute_kappa(const double nu, const double Tgas) -> double
{
	// cm^-1
	auto T_ = Tgas / T_ref;
	auto nu_ = nu / nu_ref;
	return scaleup * kappa0 * std::pow(T_, -0.5) * std::pow(nu_, -3) * (1.0 - std::exp(-coeff_ * nu_ / T_));
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<MGProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> const rad_boundaries, const double rho, const double Tgas)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<double, nGroups_ + 1> exponents{};
	amrex::GpuArray<double, nGroups_ + 1> kappa_lower{};

	for (int g = 0; g < nGroups_ + 1; ++g) {
		if constexpr (RadSystem_Traits<MGProblem>::opacity_model == OpacityModel::piecewise_constant_opacity) {
			exponents[g] = 0.0;
			if (g < n_groups_) {
				auto nu_center = std::sqrt(rad_boundaries[g] * rad_boundaries[g + 1]);
				kappa_lower[g] = compute_kappa(nu_center, Tgas) / rho;
			}
			// note kappa_lower[nGroups_] is not used
		} else {
			if (g == n_groups_) {
				exponents[g] = 0.;
				kappa_lower[g] = compute_kappa(rad_boundaries[g], Tgas) / rho;
			} else {
				auto kappa_up = compute_kappa(rad_boundaries[g + 1], Tgas);
				auto kappa_down = compute_kappa(rad_boundaries[g], Tgas);
				exponents[g] = std::log(kappa_up / kappa_down) / std::log(rad_boundaries[g + 1] / rad_boundaries[g]);
				kappa_lower[g] = kappa_down / rho;
			}
		}
		AMREX_ASSERT(!std::isnan(exponents[g]));
		AMREX_ASSERT(kappa_lower[g] >= 0.);
	}

	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> const exponents_and_values{exponents, kappa_lower};
	return exponents_and_values;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ExactProblem>::ComputePlanckOpacity(const double rho, const double Tgas) -> amrex::Real
{
	const double sigma = scaleup * 3063.96 * std::pow(Tgas / T0, -3.5);
	return sigma / rho;
}

template <> AMREX_GPU_HOST_DEVICE auto RadSystem<ExactProblem>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> amrex::Real
{
	const double sigma = scaleup * 101.248 * std::pow(Tgas / T0, -3.5);
	return sigma / rho;
}

template <> void QuokkaSimulation<MGProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	const auto radBoundaries_g = RadSystem_Traits<MGProblem>::radBoundaries;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		const double Trad = compute_initial_Tgas(x - x0);
		const double rho = compute_exact_rho(x - x0);
		const double Egas = quokka::EOS<MGProblem>::ComputeEintFromTgas(rho, Trad);
		const double v0 = v0_adv;

		auto Erad_g = RadSystem<MGProblem>::ComputeThermalRadiationMultiGroup(Trad, radBoundaries_g);
		auto Frad_g = RadSystem<MGProblem>::ComputeFluxInDiffusionLimit(radBoundaries_g, Trad, v0);

		for (int g = 0; g < Physics_Traits<MGProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<MGProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			// OLD, correct if you ignore the (delta nu B) term
			// state_cc(i, j, k, RadSystem<MGProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 4. / 3. * v0 * Erad_g[g];
			state_cc(i, j, k, RadSystem<MGProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = Frad_g[g];
			state_cc(i, j, k, RadSystem<MGProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<MGProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}

		state_cc(i, j, k, RadSystem<MGProblem>::gasEnergy_index) = Egas + 0.5 * rho * v0 * v0;
		state_cc(i, j, k, RadSystem<MGProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<MGProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<MGProblem>::x1GasMomentum_index) = v0 * rho;
		state_cc(i, j, k, RadSystem<MGProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<MGProblem>::x3GasMomentum_index) = 0.;
	});
}
template <> void QuokkaSimulation<ExactProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		const double Trad = compute_initial_Tgas(x - x0);
		const double Erad = a_rad * std::pow(Trad, 4);
		const double rho = compute_exact_rho(x - x0);
		const double Egas = quokka::EOS<MGProblem>::ComputeEintFromTgas(rho, Trad);
		const double v0 = v0_adv;

		// state_cc(i, j, k, RadSystem<MGProblem>::radEnergy_index) = (1. + 4. / 3. * (v0 * v0) / (c * c)) * Erad;
		state_cc(i, j, k, RadSystem<MGProblem>::radEnergy_index) = Erad;
		state_cc(i, j, k, RadSystem<MGProblem>::x1RadFlux_index) = 4. / 3. * v0 * Erad;
		state_cc(i, j, k, RadSystem<MGProblem>::x2RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<MGProblem>::x3RadFlux_index) = 0;
		state_cc(i, j, k, RadSystem<MGProblem>::gasEnergy_index) = Egas + 0.5 * rho * v0 * v0;
		state_cc(i, j, k, RadSystem<MGProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<MGProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<MGProblem>::x1GasMomentum_index) = v0 * rho;
		state_cc(i, j, k, RadSystem<MGProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<MGProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem is based on the radhydro_pulse test and is a test of interpolation for variable opacity for multigroup radiation.

	// Problem parameters
	const double CFL_number = 0.8;
	// const int nx = 32;

	const double max_dt = 1e-3; // t_cr = 2 cm / cs = 7e-8 s

	amrex::ParmParse const pp("rad");

	// Boundary conditions
	constexpr int nvars = RadSystem<MGProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem 1: advecting pulse with multigroup integration

	// Problem initialization
	QuokkaSimulation<MGProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	sim.radiationCflNumber_ = CFL_number;
	sim.cflNumber_ = CFL_number;
	sim.maxDt_ = max_dt;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	int nx = static_cast<int>(position.size());
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = sim.geom[0].ProbLoArray();
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = sim.geom[0].ProbHiArray();
	// compute the pixel size
	double dx = (prob_hi[0] - prob_lo[0]) / static_cast<double>(nx);
	double move = v0_adv * sim.tNew_[0];
	int n_p = static_cast<int>(move / dx);
	int half = static_cast<int>(nx / 2.0);
	double drift = move - static_cast<double>(n_p) * dx;
	int shift = n_p - static_cast<int>((n_p + half) / nx) * nx;

	std::vector<double> xs(nx);
	std::vector<double> Trad(nx);
	std::vector<double> Tgas(nx);
	std::vector<double> Vgas(nx);
	std::vector<double> rhogas(nx);

	for (int i = 0; i < nx; ++i) {
		int index_ = 0;
		if (shift >= 0) {
			if (i < shift) {
				index_ = nx - shift + i;
			} else {
				index_ = i - shift;
			}
		} else {
			if (i <= nx - 1 + shift) {
				index_ = i - shift;
			} else {
				index_ = i - (nx + shift);
			}
		}
		amrex::Real const x = position[i];
		double Erad_t = 0.0;
		for (int g = 0; g < Physics_Traits<MGProblem>::nGroups; ++g) {
			Erad_t += values.at(RadSystem<MGProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
		}
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto rho_t = values.at(RadSystem<MGProblem>::gasDensity_index)[i];
		const auto v_t = values.at(RadSystem<MGProblem>::x1GasMomentum_index)[i] / rho_t;
		const auto Egas = values.at(RadSystem<MGProblem>::gasInternalEnergy_index)[i];
		xs.at(i) = x - drift;
		rhogas.at(index_) = rho_t;
		Trad.at(index_) = Trad_t;
		Tgas.at(index_) = quokka::EOS<MGProblem>::ComputeTgasFromEint(rho_t, Egas);
		Vgas.at(index_) = 1e-5 * (v_t - v0_adv);
	}
	// END OF PROBLEM 1

	// Problem 2: exact opacity

	// Problem initialization
	QuokkaSimulation<ExactProblem> sim2(BCs_cc);

	sim2.radiationReconstructionOrder_ = 3; // PPM
	sim2.stopTime_ = max_time;
	sim2.radiationCflNumber_ = CFL_number;
	sim2.maxDt_ = max_dt;
	sim2.maxTimesteps_ = max_timesteps;
	sim2.plotfileInterval_ = -1;

	// initialize
	sim2.setInitialConditions();

	auto [position0, values0] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0.0);

	// evolve
	sim2.evolve();

	// read output variables
	auto [position2, values2] = fextract(sim2.state_new_cc_[0], sim2.Geom(0), 0, 0.0);
	nx = static_cast<int>(position2.size());
	prob_lo = sim2.geom[0].ProbLoArray();
	prob_hi = sim2.geom[0].ProbHiArray();
	// compute the pixel size
	dx = (prob_hi[0] - prob_lo[0]) / static_cast<double>(nx);
	move = v0_adv * sim2.tNew_[0];
	n_p = static_cast<int>(move / dx);
	half = static_cast<int>(nx / 2.0);
	drift = move - static_cast<double>(n_p) * dx;
	shift = n_p - static_cast<int>((n_p + half) / nx) * nx;

	std::vector<double> xs2(nx);
	std::vector<double> Trad2(nx);
	std::vector<double> xs0(nx);
	std::vector<double> Trad0(nx);
	std::vector<double> Tgas2(nx);
	std::vector<double> Vgas2(nx);
	std::vector<double> rhogas2(nx);

	for (int i = 0; i < nx; ++i) {
		int index_ = 0;
		if (shift >= 0) {
			if (i < shift) {
				index_ = nx - shift + i;
			} else {
				index_ = i - shift;
			}
		} else {
			if (i <= nx - 1 + shift) {
				index_ = i - shift;
			} else {
				index_ = i - (nx + shift);
			}
		}
		const amrex::Real x = position2[i];
		const auto Erad_t = values2.at(RadSystem<ExactProblem>::radEnergy_index)[i];
		const auto Erad_0 = values0.at(RadSystem<ExactProblem>::radEnergy_index)[i];
		const auto Trad_t = std::pow(Erad_t / a_rad, 1. / 4.);
		const auto Trad_0 = std::pow(Erad_0 / a_rad, 1. / 4.);
		const auto rho_t = values2.at(RadSystem<ExactProblem>::gasDensity_index)[i];
		const auto v_t = values2.at(RadSystem<ExactProblem>::x1GasMomentum_index)[i] / rho_t;
		const auto Egas = values2.at(RadSystem<ExactProblem>::gasInternalEnergy_index)[i];
		xs2.at(i) = x - drift;
		rhogas2.at(index_) = rho_t;
		Trad2.at(index_) = Trad_t;
		Tgas2.at(index_) = quokka::EOS<ExactProblem>::ComputeTgasFromEint(rho_t, Egas);
		Vgas2.at(index_) = 1e-5 * (v_t - v0_adv);
		const auto x0 = position0[i];
		xs0.at(i) = x0;
		Trad0.at(i) = Trad_0;
	}
	// END OF PROBLEM 2

	// compute error norm
	double err_norm = 0.;
	double sol_norm = 0.;
	for (size_t i = 0; i < xs2.size(); ++i) {
		err_norm += std::abs(Tgas[i] - Tgas2[i]);
		sol_norm += std::abs(Tgas2[i]);
	}
	const double error_tol = 0.02;
	const double rel_error = err_norm / sol_norm;
	amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

	// symmetry check
	double symm_err = 0.;
	double symm_norm = 0.;
	const double symm_err_tol = 0.02;
	for (size_t i = 0; i < xs.size(); ++i) {
		symm_err += std::abs(Tgas[i] - Tgas[xs.size() - 1 - i]);
		symm_norm += std::abs(Tgas[i]);
	}
	const double symm_rel_error_1 = symm_err / symm_norm;
	amrex::Print() << "Symmetry L1 error norm of the MG pulse = " << symm_rel_error_1 << std::endl;

	symm_err = 0.;
	symm_norm = 0.;
	for (size_t i = 0; i < xs2.size(); ++i) {
		symm_err += std::abs(Tgas2[i] - Tgas2[xs2.size() - 1 - i]);
		symm_norm += std::abs(Tgas2[i]);
	}
	const double symm_rel_error_2 = symm_err / symm_norm;
	amrex::Print() << "Symmetry L1 error norm of the exact (grey) pulse = " << symm_rel_error_2 << std::endl;

#ifdef HAVE_PYTHON
	// plot temperature
	matplotlibcpp::clf();
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;

	Trad_args["label"] = "Trad (MG integrated)";
	Trad_args["linestyle"] = "-";
	Trad_args["color"] = "C0";
	Tgas_args["label"] = "Tgas (MG integrated)";
	Tgas_args["linestyle"] = "--";
	Tgas_args["color"] = "C1";
	matplotlibcpp::plot(xs, Trad, Trad_args);
	matplotlibcpp::plot(xs, Tgas, Tgas_args);
	Trad_args["label"] = "Trad (grey)";
	Trad_args["linestyle"] = "-";
	Trad_args["color"] = "k";
	Tgas_args["label"] = "Tgas (grey)";
	Tgas_args["linestyle"] = "--";
	Tgas_args["color"] = "grey";
	matplotlibcpp::plot(xs2, Trad2, Trad_args);
	matplotlibcpp::plot(xs2, Tgas2, Tgas_args);
	// matplotlibcpp::ylim(0.98e7, 1.35e7);
	matplotlibcpp::ylim(0.98e7, 2.02e7);
	matplotlibcpp::grid(true);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (K)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("nGroups = {}, time t = {:.4g}", n_groups_, sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_MG_int_temperature.pdf");

	// plot gas density profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> rho_args;
	rho_args["label"] = "MG integrated";
	rho_args["linestyle"] = "-";
	rho_args["color"] = "C0";
	matplotlibcpp::plot(xs, rhogas, rho_args);
	rho_args["label"] = "grey)";
	rho_args["linestyle"] = "--";
	rho_args["color"] = "k";
	matplotlibcpp::plot(xs2, rhogas2, rho_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("gas density (g cm^-3)");
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("nGroups = {}, time t = {:.4g}", n_groups_, sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_MG_int_density.pdf");

	// plot gas velocity profile
	matplotlibcpp::clf();
	std::map<std::string, std::string> vgas_args;
	vgas_args["label"] = "MG interpolated";
	vgas_args["linestyle"] = "-";
	vgas_args["color"] = "C0";
	matplotlibcpp::plot(xs, Vgas, vgas_args);
	vgas_args["label"] = "grey";
	vgas_args["linestyle"] = "--";
	vgas_args["color"] = "k";
	matplotlibcpp::plot(xs2, Vgas2, vgas_args);
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("gas velocity (km s^-1)");
	matplotlibcpp::ylim(-5., 5.);
	matplotlibcpp::legend();
	matplotlibcpp::title(fmt::format("nGroups = {}, time t = {:.4g}", n_groups_, sim.tNew_[0]));
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radhydro_pulse_MG_int_velocity.pdf");
#endif

	// Save xs, Trad, Tgas, rhogas, Vgas, xs_mg, Trad_mg, Tgas_mg, rhogas_mg, Vgas_mg, xs2, Trad2, Tgas2, rhogas2, Vgas2
	if (export_csv) {
		std::ofstream file;
		file.open("radhydro_pulse_MG_int_temperature.csv");
		file << "xs,Trad,Tgas,xs2,Trad2,Tgas2\n";
		for (size_t i = 0; i < xs.size(); ++i) {
			file << std::scientific << std::setprecision(12) << xs[i] << "," << Trad[i] << "," << Tgas[i] << "," << xs2[i] << "," << Trad2[i] << ","
			     << Tgas2[i] << "\n";
		}
		file.close();

		// Save xs, rhogas, xs_mg, rhogas_mg, xs2, rhogas2
		file.open("radhydro_pulse_MG_int_density.csv");
		file << "xs,rhogas,xs2,rhogas2\n";
		for (size_t i = 0; i < xs.size(); ++i) {
			file << std::scientific << std::setprecision(12) << xs[i] << "," << rhogas[i] << "," << xs2[i] << "," << rhogas2[i] << "\n";
		}
		file.close();

		// Save xs, Vgas, xs_mg, Vgas_mg, xs2, Vgas2
		file.open("radhydro_pulse_MG_int_velocity.csv");
		file << "xs,Vgas,xs2,Vgas2\n";
		for (size_t i = 0; i < xs.size(); ++i) {
			file << std::scientific << std::setprecision(12) << xs[i] << "," << Vgas[i] << "," << xs2[i] << "," << Vgas2[i] << "\n";
		}
		file.close();
	}

	// Cleanup and exit
	int status = 0;
	if ((rel_error > error_tol) || std::isnan(rel_error) || (symm_rel_error_1 > symm_err_tol) || (symm_rel_error_2 > symm_err_tol) ||
	    std::isnan(symm_rel_error_1) || std::isnan(symm_rel_error_2)) {
		status = 1;
	}
	return status;
}
