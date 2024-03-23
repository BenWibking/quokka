/// \file first_star.cpp
/// \brief 
///

#include "first_star.hpp"
#include "AMReX_BC_TYPES.H"
#include "AMReX_MultiFab.H"
#include <AMReX_MultiFabUtil.H>
#include "AMReX_Print.H"
#include "RadhydroSimulation.hpp"
#include "fextract.hpp"
#include "physics_info.hpp"

#ifdef HAVE_PYTHON
#include "matplotlibcpp.h"
#endif

namespace fs = std::filesystem;

struct TheProblem {
};

AMREX_GPU_MANAGED int model = 0;
AMREX_GPU_MANAGED double init_A = NAN;
AMREX_GPU_MANAGED double init_q = NAN;
AMREX_GPU_MANAGED double init_h = NAN;
AMREX_GPU_MANAGED double r_star = NAN;
AMREX_GPU_MANAGED double rho_star = NAN;
AMREX_GPU_MANAGED double rho_bg = -1.0;
AMREX_GPU_MANAGED int v_bg = 0;
// model 0 parameters
AMREX_GPU_MANAGED double model0_gamma1 = 1.00001;
AMREX_GPU_MANAGED double model0_gamma2 = 5.0 / 3.0;

// constexpr const char* subfolder = __DATE__ " " __TIME__;
constexpr const char* subfolder = "diagnostics";
bool test_passes = true; // if one of the energy checks fails, set to false

// model 0: Lin+11
// model 1: 1/(1 + n^s) + 1/(1 + n1^s) * (n / n1)^(gamma - 1)

constexpr double G = 1.0;
constexpr double pi = M_PI;
constexpr double k_B = 1.0;
constexpr double c_iso = 1.0;
constexpr double r_c = 1.0;

// EOS parameters
constexpr double mu = 1.0;
constexpr double gamma_ad = 5.0 / 3.0;
constexpr double Cv = 1.0 / (gamma_ad - 1.0) * k_B / mu; // Specific heat at constant volume in the adiabatic phase
constexpr double T0 = c_iso * c_iso;

// model 1 parameters
constexpr double model1_s = 4.0; // jump slope. The jump equals s * log10(rho1 / rho0)
constexpr double model1_rho1_over_rho0 = 3.2;

// model 2 parameters
constexpr double rho_core = 1.0;
constexpr double rho_one = 1000.0;
constexpr double model2_jump_slope = 10.0;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = gamma_ad;
	// static constexpr double cs_isothermal = 1.0;
};

// template <> struct HydroSystem_Traits<TheProblem> {
// 	static constexpr bool reconstruct_eint = false;
// };

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = false;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 1;
};

template <> struct SimulationData<TheProblem> {

	// temporal quantities
	std::vector<amrex::Real> time{};
	std::vector<amrex::Real> mass{}; // stellar mass
	std::vector<amrex::Real> tot_mass{}; // total mass in simulation box
	std::vector<amrex::Real> position_x{};
	std::vector<amrex::Real> position_y{};
	std::vector<amrex::Real> position_z{};
	std::vector<amrex::Real> velocity_x{};
	std::vector<amrex::Real> velocity_y{};
	std::vector<amrex::Real> velocity_z{};
	std::vector<amrex::Real> rotation_radius{};
	std::vector<amrex::Real> spin_angular_mtm{};

	// init derived quantities
	amrex::Real init_x_momentum = NAN;
};

AMREX_GPU_HOST_DEVICE 
auto compute_T(const double rho) -> double
{
	// amrex::Real const A = userData_.init_A;
	// amrex::Real const r_star = userData_.init_q * r_c;
	// amrex::Real const rho_star = A * c_iso * c_iso / (4.0 * pi * G * r_star * r_star);
	if (model == 0) {
    // P = c_s^2 * rho^gamma1 (1 + (rho / rho_star)^(gamma2 - gamma1))
    auto P = c_iso * c_iso * std::pow(rho, model0_gamma1) * (1.0 + std::pow(rho / rho_star, model0_gamma2 - model0_gamma1));
    return P / (rho * k_B / mu);
  }
	if (model == 1) {
		auto rho0 = rho_star;
		// auto rho1 = 3.2 * rho0; // jump ends at this density
		const double scale = 1.0 / (1.0 + std::pow(rho / rho0, model1_s)) + 1.0 / (1.0 + std::pow(model1_rho1_over_rho0, model1_s)) * std::pow(rho / (model1_rho1_over_rho0 * rho0), gamma_ad - 1.0);
		return scale * T0;
	} 
	if (model == 2) {
		const double scale = 1.0 / (1.0 + std::exp(model2_jump_slope * (rho / rho_core - 1.0))) + std::pow(rho / rho_one, gamma_ad - 1.0);
		return scale * T0;
	} 
	return -1.0;
}

AMREX_GPU_HOST_DEVICE
auto compute_e(const double rho) -> double
{
	if (model == 0 || model == 1) {
		const auto Tgas = compute_T(rho);
		const double e = Cv * rho * Tgas;
		return e;
	} 
	if (model == 2) {
	  double Tgas = NAN;
		if (rho >= rho_core) {
			Tgas = compute_T(rho);
			return Cv * rho * Tgas;
		}
		Tgas = compute_T(rho_core);
		return Cv * rho_core * Tgas;
	}
	return -1.0;
}

// redefine EOS::ComputePressure
template <> 
AMREX_GPU_HOST_DEVICE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputePressure(amrex::Real rho, amrex::Real /*Eint*/, 
											const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/) -> amrex::Real
{
	if (model == 0 || model == 1) {
		const double e = compute_e(rho);
		return (gamma_ad - 1.0) * e;
	} 
	if (model == 2) {
		const double T = compute_T(rho);
		const double e = rho * Cv * T;
		return (gamma_ad - 1.0) * e;
	}
	return -1.0;
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeTgasFromEint(amrex::Real rho, amrex::Real /*Egas*/,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	return compute_T(rho);
	// amrex::Real const A = userData_.init_A;
	// amrex::Real const r_star = userData_.init_q * r_c;
	// amrex::Real const rho_star = A * c_iso * c_iso / (4.0 * pi * G * r_star * r_star);
	// if constexpr (model == 1) {
	// 	const double scale = 1.0 / (1.0 + std::pow(rho / rho0, model1_s)) + 1.0 / (1.0 + std::pow(rho1 / rho0, model1_s)) * std::pow(rho / rho1, gamma_ad - 1.0);
	// 	return scale * T0;
	// } else if constexpr (model == 2) {
	// 	const double scale = 1.0 / (1.0 + std::exp(model2_jump_slope * (rho / rho_core - 1.0))) + std::pow(rho / rho_one, gamma_ad - 1.0);
	// 	return scale * T0;
	// }
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeEintFromTgas(amrex::Real rho, amrex::Real /*Tgas*/,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	return compute_e(rho);
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeEintFromPres(amrex::Real rho, amrex::Real Pressure,
										  const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	return compute_e(rho);
}

template <>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto quokka::EOS<TheProblem>::ComputeSoundSpeed(amrex::Real rho, amrex::Real /*Pressure*/,
										const std::optional<amrex::GpuArray<amrex::Real, nmscalars_>> /*massScalars*/)
    -> amrex::Real
{
	const double T = compute_T(rho);
	return std::sqrt(k_B * T / mu);
}

template <> void RadhydroSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + 0.5) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + 0.5) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));
		amrex::Real const distxy = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));

		// amrex::Real const A = userData_.init_A;
		// amrex::Real const r_star = userData_.init_q * r_c;
		// amrex::Real const rho_star = A * c_iso * c_iso / (4.0 * pi * G * r_star * r_star);

		// const double rho_bg = rho_star * std::pow(r_star / r_c, 2);
		const double rho_bg_ = rho_bg > 0.0 ? rho_bg * rho_star : rho_star * (r_star / r_c) * (r_star / r_c);;

		// compute density
		double rho = NAN;
		if (r <= r_star) {
			rho = rho_star;
		} else if (r <= r_c) {
			// rho = rho_star * std::pow(r_star / r, 2);
      rho = rho_star * (r_star / r) * (r_star / r);
		} else {
			rho = rho_bg_;
		}
    AMREX_ALWAYS_ASSERT(rho > 0.0);
		const auto E_int = compute_e(rho);

		// compute azimuthal velocity
		double v_phi = 0.0;
		if (distxy <= r_star) {
			v_phi = 2 * init_A * c_iso * init_h;
			v_phi *= distxy / r_star;
		} else if (distxy <= r_c) {
			v_phi = 2 * init_A * c_iso * init_h;
		} else {
			if (v_bg == 0) {
				v_phi = 0.0;
			} else {
				v_phi = 2 * init_A * c_iso * init_h;
			}
		}

		// compute x, y, z velocity
		const double v_x = -v_phi * (y - y0) / distxy;
		const double v_y = v_phi * (x - x0) / distxy;
		const double v_z = 0.0;

		state_cc(i, j, k, HydroSystem<TheProblem>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TheProblem>::x1Momentum_index) = rho * v_x;
		state_cc(i, j, k, HydroSystem<TheProblem>::x2Momentum_index) = rho * v_y;
		state_cc(i, j, k, HydroSystem<TheProblem>::x3Momentum_index) = rho * v_z;
		state_cc(i, j, k, HydroSystem<TheProblem>::internalEnergy_index) = E_int;
		state_cc(i, j, k, HydroSystem<TheProblem>::energy_index) = E_int + 0.5 * rho * (v_x * v_x + v_y * v_y + v_z * v_z);
	});
}

template <> void RadhydroSimulation<TheProblem>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{

	// read-in jeans length refinement runtime params
	int N_J = 1e5;
	// Real jeans_density_threshold = NAN;
	Real jeans_density_threshold_over_rho_star = 0.5;

	amrex::ParmParse const pp("refine");
	pp.query("jeans_num", N_J); // inverse of the 'Jeans number' [Truelove et al. (1997)]
	// pp.query("density_threshold", jeans_density_threshold);
	pp.query("density_threshold_over_rho_star", jeans_density_threshold_over_rho_star);

	auto const rho_threshold = jeans_density_threshold_over_rho_star * rho_star;

	const amrex::Real dx = geom[lev].CellSizeArray()[0];

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto state = state_new_cc_[lev].const_array(mfi);
		const auto tag = tags.array(mfi);
		const int nidx = HydroSystem<TheProblem>::density_index;

		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			Real const rho = state(i, j, k, nidx);

			const amrex::Real rho_Jeans = std::pow(c_iso / (N_J * dx) * std::sqrt(M_PI / G), 2);

			if (rho > rho_Jeans || rho > rho_threshold) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

template <> void RadhydroSimulation<TheProblem>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	// check conservation of x-momentum
	amrex::Real const px0 = initSumCons[RadSystem<TheProblem>::x1GasMomentum_index] * vol;
	amrex::Real const py0 = initSumCons[RadSystem<TheProblem>::x2GasMomentum_index] * vol;
	amrex::Real const pz0 = initSumCons[RadSystem<TheProblem>::x3GasMomentum_index] * vol;
	amrex::Real const px = state_new_cc_[0].sum(RadSystem<TheProblem>::x1GasMomentum_index) * vol;
	amrex::Real const py = state_new_cc_[0].sum(RadSystem<TheProblem>::x2GasMomentum_index) * vol;
	amrex::Real const pz = state_new_cc_[0].sum(RadSystem<TheProblem>::x3GasMomentum_index) * vol;

	// Check X-momentum conservation

	amrex::Real const abs_err = (px - px0);
	amrex::Real const rel_err = abs_err / px0;

	amrex::Print() << "\nInitial X-momentum = " << px0 << std::endl;
	amrex::Print() << "Final X-momentum = " << px << std::endl;
	amrex::Print() << "\tabsolute conservation error = " << abs_err << std::endl;
	amrex::Print() << "\trelative conservation error = " << rel_err << std::endl;
	amrex::Print() << std::endl;

	const double err_tol = 1.0e-10;
	bool p_test_passes = true;  // does x-momentum conserve?

	if (((std::abs(rel_err) > err_tol) && (std::abs(abs_err) > 1.0e-10)) || std::isnan(abs_err)) {
		// Note that the initial x-momentum is zero, so the relative error is not meaningful
		// The RMS X-momentum is of order unity, so the absolute error should be smaller than 1.0e-10
		amrex::Print() << "X-momentum not conserved to machine precision!\n";
		p_test_passes = false;
	} else {
		amrex::Print() << "X-momentum conservation is OK.\n";
	}

	// Check Y-momentum conservation

	amrex::Real const abs_err_y = (py - py0);
	amrex::Real const rel_err_y = abs_err_y / py0;

	amrex::Print() << "\nInitial Y-momentum = " << py0 << std::endl;
	amrex::Print() << "Final Y-momentum = " << py << std::endl;
	amrex::Print() << "\tabsolute conservation error = " << abs_err_y << std::endl;
	amrex::Print() << "\trelative conservation error = " << rel_err_y << std::endl;
	amrex::Print() << std::endl;

	if (((std::abs(rel_err_y) > err_tol) && (std::abs(abs_err_y) > 1.0e-10)) || std::isnan(abs_err_y)) {
		// Note that the initial y-momentum is zero, so the relative error is not meaningful
		// The RMS y-momentum is of order unity, so the absolute error should be smaller than 1.0e-10
		amrex::Print() << "Y-momentum not conserved to machine precision!\n";
		p_test_passes = false;
	} else {
		amrex::Print() << "Y-momentum conservation is OK.\n";
	}

	// Check Z-momentum conservation

	amrex::Real const abs_err_z = (pz - pz0);
	amrex::Real const rel_err_z = abs_err_z / pz0;

	amrex::Print() << "\nInitial Z-momentum = " << pz0 << std::endl;
	amrex::Print() << "Final Z-momentum = " << pz << std::endl;
	amrex::Print() << "\tabsolute conservation error = " << abs_err_z << std::endl;
	amrex::Print() << "\trelative conservation error = " << rel_err_z << std::endl;
	amrex::Print() << std::endl;

	if (((std::abs(rel_err_z) > err_tol) && (std::abs(abs_err_z) > 1.0e-10)) || std::isnan(abs_err_z)) {
		// Note that the initial z-momentum is zero, so the relative error is not meaningful
		// The RMS z-momentum is of order unity, so the absolute error should be smaller than 1.0e-10
		amrex::Print() << "Z-momentum not conserved to machine precision!\n";
		p_test_passes = false;
	} else {
		amrex::Print() << "Z-momentum conservation is OK.\n";
	}

	// if both tests pass, then overall pass
  test_passes = p_test_passes;
}

template <> void RadhydroSimulation<TheProblem>::computeAfterTimestep(std::optional<int> global_step)
{

	static int local_step = -1;
  int step = 0;
  if (global_step.has_value()) {
    step = global_step.value();
  } else {
	  ++local_step;
    step = local_step;
  }

	// --------- Compute diagonostics every computeInterval_ steps -----------------
  if ((computeInterval_ > 0) && (step % computeInterval_ == 0)) {

		// get the finest level 
    const int fine_level = finestLevel();
		const int dim = 3;

		const double rho_threshold = rho_star;
		// const double rho_threshold = -1.0;

		double mass_tot = 0.0;
		double position_x_tot = 0.0;
		double position_y_tot = 0.0;
		double position_z_tot = 0.0;
		double velocity_x_tot = 0.0;
		double velocity_y_tot = 0.0;
		double velocity_z_tot = 0.0;
		double rot_radius_tot = 0.0;
		double spin_L_tot = 0.0;
		double max_rho = -1.0;

		for (int ilev = 0; ilev <= fine_level; ++ilev) {
			// compute SimulationData
			auto const &dx = geom[ilev].CellSizeArray();
			amrex::Real const vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
			auto const prob_lo = geom[ilev].ProbLoArray();

			amrex::MultiFab rho_star_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);
			amrex::MultiFab rho_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);
			amrex::MultiFab position_x_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);
			amrex::MultiFab position_y_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);
			amrex::MultiFab position_z_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);
			amrex::MultiFab velocity_x_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);
			amrex::MultiFab velocity_y_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);
			amrex::MultiFab velocity_z_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);

			if (ilev < fine_level) {
        const amrex::IntVect ratio{refRatio(ilev)};
				const amrex::iMultiFab mask = makeFineMask(boxArray(ilev), DistributionMap(ilev), boxArray(ilev+1), ratio);
				auto const& ima = mask.const_arrays();

				for (amrex::MFIter iter(state_new_cc_[ilev]); iter.isValid(); ++iter) {
					const amrex::Box &indexRange = iter.validbox();

					const auto& m = mask.array(iter);

					auto const &state = state_new_cc_[ilev].const_array(iter);
					auto const &mass = rho_star_mf.array(iter);
					auto const &rho_all = rho_mf.array(iter);
					auto const &position_x = position_x_mf.array(iter);
					auto const &position_y = position_y_mf.array(iter);
					auto const &position_z = position_z_mf.array(iter);
					auto const &velocity_x = velocity_x_mf.array(iter);
					auto const &velocity_y = velocity_y_mf.array(iter);
					auto const &velocity_z = velocity_z_mf.array(iter);

					amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
						Real const rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
						rho_all(i, j, k) = rho;
						if ((m(i, j, k) == 0) && (rho > rho_threshold)) {
							Real const x = prob_lo[0] + (i + 0.5) * dx[0];
							Real const y = prob_lo[1] + (j + 0.5) * dx[1];
							Real const z = prob_lo[2] + (k + 0.5) * dx[2];
							Real const px = state(i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
							Real const py = state(i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
							Real const pz = state(i, j, k, HydroSystem<TheProblem>::x3Momentum_index);
							mass(i, j, k) = rho;
							position_x(i, j, k) = x * rho;
							position_y(i, j, k) = y * rho;
							position_z(i, j, k) = z * rho;
							velocity_x(i, j, k) = px;
							velocity_y(i, j, k) = py;
							velocity_z(i, j, k) = pz;
						} else {
							mass(i, j, k) = 0.0;
							position_x(i, j, k) = 0.0;
							position_y(i, j, k) = 0.0;
							position_z(i, j, k) = 0.0;
							velocity_x(i, j, k) = 0.0;
							velocity_y(i, j, k) = 0.0;
							velocity_z(i, j, k) = 0.0;
						}
					});
				}
			} else { // ilev == fine_level
				for (amrex::MFIter iter(state_new_cc_[ilev]); iter.isValid(); ++iter) {
					const amrex::Box &indexRange = iter.validbox();

					auto const &state = state_new_cc_[ilev].const_array(iter);
					auto const &mass = rho_star_mf.array(iter);
					auto const &rho_all = rho_mf.array(iter);
					auto const &position_x = position_x_mf.array(iter);
					auto const &position_y = position_y_mf.array(iter);
					auto const &position_z = position_z_mf.array(iter);
					auto const &velocity_x = velocity_x_mf.array(iter);
					auto const &velocity_y = velocity_y_mf.array(iter);
					auto const &velocity_z = velocity_z_mf.array(iter);

					amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
						Real const rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
						rho_all(i, j, k) = rho;
						if (rho > rho_threshold) {
							Real const x = prob_lo[0] + (i + 0.5) * dx[0];
							Real const y = prob_lo[1] + (j + 0.5) * dx[1];
							Real const z = prob_lo[2] + (k + 0.5) * dx[2];
							Real const px = state(i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
							Real const py = state(i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
							Real const pz = state(i, j, k, HydroSystem<TheProblem>::x3Momentum_index);
							mass(i, j, k) = rho;
							position_x(i, j, k) = x * rho;
							position_y(i, j, k) = y * rho;
							position_z(i, j, k) = z * rho;
							velocity_x(i, j, k) = px;
							velocity_y(i, j, k) = py;
							velocity_z(i, j, k) = pz;
						} else {
							mass(i, j, k) = 0.0;
							position_x(i, j, k) = 0.0;
							position_y(i, j, k) = 0.0;
							position_z(i, j, k) = 0.0;
							velocity_x(i, j, k) = 0.0;
							velocity_y(i, j, k) = 0.0;
							velocity_z(i, j, k) = 0.0;
						}
					});
				}
			}

			mass_tot += rho_star_mf.sum(0) * vol;
			max_rho = std::max(max_rho, rho_mf.max(0));
			position_x_tot += position_x_mf.sum(0) * vol;
			position_y_tot += position_y_mf.sum(0) * vol;
			position_z_tot += position_z_mf.sum(0) * vol;
			velocity_x_tot += velocity_x_mf.sum(0) * vol;
			velocity_y_tot += velocity_y_mf.sum(0) * vol;
			velocity_z_tot += velocity_z_mf.sum(0) * vol;
		}

		const amrex::Real current_time = tNew_[0];
		const amrex::Real Mass = mass_tot;
		const amrex::Real Position_x = position_x_tot / Mass;
		const amrex::Real Position_y = position_y_tot / Mass;
		const amrex::Real Position_z = position_z_tot / Mass;
		const amrex::Real Velocity_x = velocity_x_tot / Mass;
		const amrex::Real Velocity_y = velocity_y_tot / Mass;
		const amrex::Real Velocity_z = velocity_z_tot / Mass;

		for (int ilev = 0; ilev <= fine_level; ++ilev) {
			// compute SimulationData
			auto const &dx = geom[ilev].CellSizeArray();
			amrex::Real const vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
			auto const prob_lo = geom[ilev].ProbLoArray();

			amrex::MultiFab rot_radius_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);
			amrex::MultiFab spin_L_mf(boxArray(ilev), DistributionMap(ilev), 1, 0);

			if (ilev < fine_level) {
        const amrex::IntVect ratio{refRatio(ilev)};
				const amrex::iMultiFab mask = makeFineMask(boxArray(ilev), DistributionMap(ilev), boxArray(ilev+1), ratio);
				auto const& ima = mask.const_arrays();

				for (amrex::MFIter iter(state_new_cc_[ilev]); iter.isValid(); ++iter) {
					const amrex::Box &indexRange = iter.validbox();

					const auto& m = mask.array(iter);
					auto const &state = state_new_cc_[ilev].const_array(iter);
					auto const &rot_radius = rot_radius_mf.array(iter);
					auto const &spin_L = spin_L_mf.array(iter);

					amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
						Real const rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
						if ((m(i, j, k) == 0) && (rho > rho_threshold)) {
							Real const x = prob_lo[0] + (i + 0.5) * dx[0];
							Real const y = prob_lo[1] + (j + 0.5) * dx[1];
							Real const px = state(i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
							Real const py = state(i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
							Real const distSqr = std::pow(x - Position_x, 2) + std::pow(y - Position_y, 2);
							rot_radius(i, j, k) = distSqr * rho;
							spin_L(i, j, k) = (x - Position_x) * py - (y - Position_y) * px;
						} else {
							rot_radius(i, j, k) = 0.0;
							spin_L(i, j, k) = 0.0;
						}
					});
				}
			} else { // ilev = fine level
				for (amrex::MFIter iter(state_new_cc_[ilev]); iter.isValid(); ++iter) {
					const amrex::Box &indexRange = iter.validbox();

					auto const &state = state_new_cc_[ilev].const_array(iter);
					auto const &rot_radius = rot_radius_mf.array(iter);
					auto const &spin_L = spin_L_mf.array(iter);

					amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
						Real const rho = state(i, j, k, HydroSystem<TheProblem>::density_index);
						if (rho > rho_threshold) {
							Real const x = prob_lo[0] + (i + 0.5) * dx[0];
							Real const y = prob_lo[1] + (j + 0.5) * dx[1];
							Real const px = state(i, j, k, HydroSystem<TheProblem>::x1Momentum_index);
							Real const py = state(i, j, k, HydroSystem<TheProblem>::x2Momentum_index);
							Real const distSqr = std::pow(x - Position_x, 2) + std::pow(y - Position_y, 2);
							rot_radius(i, j, k) = distSqr * rho;
							spin_L(i, j, k) = (x - Position_x) * py - (y - Position_y) * px;
						} else {
							rot_radius(i, j, k) = 0.0;
							spin_L(i, j, k) = 0.0;
						}
					});
				}
			}

			rot_radius_tot += rot_radius_mf.sum(0) * vol;
			spin_L_tot += spin_L_mf.sum(0) * vol;
		}

		amrex::Real const Rot_radius = std::sqrt(rot_radius_tot / Mass);
		amrex::Real const Spin_L = spin_L_tot / Mass;

		// Write to userData_
		userData_.time.push_back(current_time);
		userData_.mass.push_back(Mass);
		userData_.position_x.push_back(Position_x);
		userData_.position_y.push_back(Position_y);
		userData_.position_z.push_back(Position_z);
		userData_.velocity_x.push_back(Velocity_x);
		userData_.velocity_y.push_back(Velocity_y);
		userData_.velocity_z.push_back(Velocity_z);
		userData_.rotation_radius.push_back(Rot_radius);
		userData_.spin_angular_mtm.push_back(Spin_L);

		amrex::Real const tot_density = state_new_cc_[0].sum(HydroSystem<TheProblem>::density_index);
		auto const &dx0 = Geom(0).CellSizeArray();
		amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
		const double tot_mass = tot_density * vol;
		userData_.tot_mass.push_back(tot_mass);

		amrex::Print() << "Max density = " << fmt::format("{:.12g}", max_rho) << std::endl;

    // Save user data to file
    if (amrex::ParallelDescriptor::MyProc() == 0) {
      std::ofstream file(fmt::format("./{}/diagnostics-step{:05d}.txt", subfolder, step));
      if (file.is_open()) {
        file << "time tot_mass mass position_x position_y position_z velocity_x velocity_y velocity_z rotation_radius spin_angular_mtm\n";
        file << fmt::format("{:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}\n",
                            current_time, tot_mass, Mass, Position_x, Position_y, Position_z, Velocity_x, Velocity_y, Velocity_z, Rot_radius, Spin_L);
        file.close();
      } else {
        std::cerr << "Unable to open file" << std::endl;
      }
    }
	} // End of computeInterval_

	// ------------ Plot radial profiles every plotfileInterval_ steps -------------
  if ((computeInterval_ > 0) && (step % computeInterval_ == 0)) {
    // read output variables
    // Extract the data at the final time at the center of the y-z plane (center=true) 
    auto [position, values] = fextract(state_new_cc_[0], Geom(0), 0, 0.0, true);
    const int nx = static_cast<int>(position.size());

    std::vector<double> xs(nx);
    std::vector<double> Tgas(nx);
    std::vector<double> Egas(nx);
    std::vector<double> Vgas(nx);
    std::vector<double> rhogas(nx);
    std::vector<double> pressure(nx);

    for (int i = 0; i < nx; ++i) {
      amrex::Real const x = position[i];
      xs.at(i) = x;
      const auto rho_t = values.at(HydroSystem<TheProblem>::density_index)[i];
      const auto v_t = values.at(HydroSystem<TheProblem>::x2Momentum_index)[i] / rho_t; // v_y
      const auto Egas_t = values.at(HydroSystem<TheProblem>::internalEnergy_index)[i];
      const auto pressure_t = quokka::EOS<TheProblem>::ComputePressure(rho_t, Egas_t);
      const auto T_t = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho_t, Egas_t);
      rhogas.at(i) = rho_t;
      Tgas.at(i) = T_t;
      Egas.at(i) = Egas_t;
      Vgas.at(i) = v_t;
      pressure.at(i) = pressure_t;
    }

    // Save user data to file

		// save rho
    std::ofstream file(fmt::format("./{}/radial-profile-rho-step{:05d}.txt", subfolder, step));
    if (file.is_open()) {
      file << "time = " << tNew_[0] << "\n";
      file << "x, rho" << "\n";
      for (int i = 0; i < nx; ++i) {
        file << fmt::format("{:.12g}, {:.12g}\n", xs.at(i), rhogas.at(i));
      }
      file.close();
    } else {
      std::cerr << "Unable to open file" << std::endl;
    }

		// save v
		file.open(fmt::format("./{}/radial-profile-vy-step{:05d}.txt", subfolder, step));
		if (file.is_open()) {
			file << "time = " << tNew_[0] << "\n";
			file << "x, vy" << "\n";
			for (int i = 0; i < nx; ++i) {
				file << fmt::format("{:.12g}, {:.12g}\n", xs.at(i), Vgas.at(i));
			}
			file.close();
		} else {
			std::cerr << "Unable to open file" << std::endl;
		}

		// save T
		file.open(fmt::format("./{}/radial-profile-T-step{:05d}.txt", subfolder, step));
		if (file.is_open()) {
			file << "time = " << tNew_[0] << "\n";
			file << "x, T" << "\n";
			for (int i = 0; i < nx; ++i) {
				file << fmt::format("{:.12g}, {:.12g}\n", xs.at(i), Tgas.at(i));
			}
			file.close();
		} else {
			std::cerr << "Unable to open file" << std::endl;
		}

#ifdef HAVE_PYTHON
    // // plot temperature
    // matplotlibcpp::clf();
    // std::map<std::string, std::string> Tgas_args;
    // Tgas_args["label"] = "gas temperature";
    // Tgas_args["linestyle"] = "-";
    // matplotlibcpp::plot(xs, Tgas, Tgas_args);
    // matplotlibcpp::xlabel("x");
    // matplotlibcpp::ylabel("temperature");
    // // matplotlibcpp::legend();
    // matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
    // matplotlibcpp::tight_layout();
    // // matplotlibcpp::save("./first-star-T.png");
    // matplotlibcpp::save(fmt::format("./{}/first-star-T-s{:06d}.png", subfolder, step));

    // // plot internal energy
    // matplotlibcpp::clf();
    // std::map<std::string, std::string> Egas_args;
    // Egas_args["label"] = "gas internal energy";
    // Egas_args["linestyle"] = "-";
    // matplotlibcpp::plot(xs, Egas, Egas_args);
    // matplotlibcpp::xlabel("x");
    // matplotlibcpp::ylabel("internal energy");
    // // matplotlibcpp::legend();
    // matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
    // matplotlibcpp::tight_layout();
    // matplotlibcpp::save(fmt::format("./{}/first-star-E-s{:06d}.png", subfolder, step));

    // // plot pressure
    // matplotlibcpp::clf();
    // std::map<std::string, std::string> pressure_args;
    // pressure_args["label"] = "gas pressure";
    // pressure_args["linestyle"] = "-";
    // matplotlibcpp::plot(xs, pressure, pressure_args);
    // matplotlibcpp::xlabel("x");
    // matplotlibcpp::ylabel("pressure");
    // // matplotlibcpp::legend();
    // matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
    // matplotlibcpp::tight_layout();
    // // matplotlibcpp::save("./first-star-P.png");
    // matplotlibcpp::save(fmt::format("./{}/first-star-P-s{:06d}.png", subfolder, step));

    // // plot gas velocity profile
    // matplotlibcpp::clf();
    // std::map<std::string, std::string> vgas_args;
    // vgas_args["label"] = "gas velocity";
    // vgas_args["linestyle"] = "-";
    // matplotlibcpp::plot(xs, Vgas, vgas_args);
    // matplotlibcpp::xlabel("x");
    // matplotlibcpp::ylabel("v_y");
    // // matplotlibcpp::legend();
    // matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
    // matplotlibcpp::tight_layout();
    // // matplotlibcpp::save("./first-star-v.png");
    // matplotlibcpp::save(fmt::format("./{}/first-star-v-s{:06d}.png", subfolder, step));

    // plot density profile
    matplotlibcpp::clf();
    std::map<std::string, std::string> rhogas_args;
    rhogas_args["label"] = "gas density";
    rhogas_args["linestyle"] = "-";
    matplotlibcpp::plot(xs, rhogas, rhogas_args);
    matplotlibcpp::xlabel("x");
    matplotlibcpp::ylabel("density");
    // matplotlibcpp::legend();
    matplotlibcpp::title(fmt::format("time t = {:.4g}", tNew_[0]));
    matplotlibcpp::tight_layout();
    // matplotlibcpp::save("./first-star-rho.png");
    matplotlibcpp::save(fmt::format("./{}/radial-profile-rho-s{:06d}.png", subfolder, step));
#endif
    }
}

auto problem_main() -> int
{

	const double max_dt = 1e0;
	double init_dt = 0.00046875;

	// Boundary conditions
	const int ncomp_cc = Physics_Indices<TheProblem>::nvarTotal_cc;
	amrex::Vector<amrex::BCRec> BCs_cc(ncomp_cc);
	for (int n = 0; n < ncomp_cc; ++n) {
		for (int i = 0; i < AMREX_SPACEDIM; ++i) {
			// BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			// BCs_cc[n].setHi(i, amrex::BCType::int_dir);
			BCs_cc[n].setLo(i, amrex::BCType::foextrap);
			BCs_cc[n].setHi(i, amrex::BCType::foextrap);
		}
	}

	// Problem initialization
	RadhydroSimulation<TheProblem> sim(BCs_cc);

	sim.doPoissonSolve_ = 1; // enable self-gravity

	sim.reconstructionOrder_ = 3; // 2=PLM, 3=PPM
	sim.maxDt_ = max_dt;

	// read problem parameters
	amrex::ParmParse const pp("eos");
	pp.query("model", model);
	pp.query("model0_gamma1", model0_gamma1);
	pp.query("model0_gamma2", model0_gamma2);

	amrex::ParmParse const ppi("init");
	ppi.query("A", init_A);
	ppi.query("q", init_q);
	ppi.query("h", init_h);
	ppi.query("rho_bg", rho_bg);
	ppi.query("v_bg", v_bg);
	ppi.query("init_dt", init_dt);

	r_star = init_q * r_c;
	rho_star = init_A * c_iso * c_iso / (4.0 * pi * G * r_star * r_star);

	// initialize
	sim.initDt_ = init_dt;
	sim.setInitialConditions();

	// Check if the directory already exists
	if (amrex::ParallelDescriptor::MyProc() == 0) {
		const std::string directory_name = subfolder;
		if (fs::exists(directory_name)) {
			std::cout << "Directory already exists." << std::endl;
		} else {
			// Create the directory
			if (fs::create_directory(directory_name)) {
				std::cout << "Directory created successfully." << std::endl;
			} else {
				std::cerr << "Failed to create directory." << std::endl;
			}
		}
	}

	// calcualte initial x-momentum
	auto const px0 = sim.state_new_cc_[0].sum(HydroSystem<TheProblem>::x1Momentum_index);
	auto const &dx0 = sim.Geom(0).CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);
	sim.userData_.init_x_momentum = px0 * vol;

  sim.computeAfterTimestep(0);

	// evolve
	sim.evolve();

	// Save user data to file
	if (amrex::ParallelDescriptor::MyProc() == 0) {
		std::ofstream file(fmt::format("./{}/diagnostics-temperal-all.txt", subfolder));
		if (file.is_open()) {
			file << "time tot_mass mass position_x position_y position_z velocity_x velocity_y velocity_z rotation_radius spin_angular_mtm\n";
			for (size_t i = 0; i < sim.userData_.time.size(); ++i) {
				file << fmt::format("{:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}, {:.12g}\n",
														sim.userData_.time[i], sim.userData_.tot_mass[i], sim.userData_.mass[i], sim.userData_.position_x[i], sim.userData_.position_y[i], sim.userData_.position_z[i],
														sim.userData_.velocity_x[i], sim.userData_.velocity_y[i], sim.userData_.velocity_z[i], sim.userData_.rotation_radius[i], sim.userData_.spin_angular_mtm[i]);
			}
			file.close();
		} else {
			std::cerr << "Unable to open file" << std::endl;
		}
	}

	// Plot user data
#ifdef HAVE_PYTHON
	// Plot mass
	matplotlibcpp::clf();
	std::map<std::string, std::string> mass_args;
	mass_args["label"] = "mass";
	mass_args["linestyle"] = "-";
	matplotlibcpp::plot(sim.userData_.time, sim.userData_.mass, mass_args);
	matplotlibcpp::xlabel("time");
	matplotlibcpp::ylabel("mass");
	// matplotlibcpp::legend();
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./{}/diagnostics-temperal-mass.png", subfolder));

	// Plot position
	matplotlibcpp::clf();
	std::map<std::string, std::string> position_x_args;
	position_x_args["label"] = "position_x";
	position_x_args["linestyle"] = "-";
	matplotlibcpp::plot(sim.userData_.time, sim.userData_.position_x, position_x_args);
	matplotlibcpp::xlabel("time");
	matplotlibcpp::ylabel("position_x");
	// matplotlibcpp::legend();
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./{}/diagnostics-temperal-position_x.png", subfolder));

	// Plot velocity
	matplotlibcpp::clf();
	std::map<std::string, std::string> velocity_x_args;
	velocity_x_args["label"] = "velocity_x";
	velocity_x_args["linestyle"] = "-";
	matplotlibcpp::plot(sim.userData_.time, sim.userData_.velocity_x, velocity_x_args);
	matplotlibcpp::xlabel("time");
	matplotlibcpp::ylabel("velocity_x");
	// matplotlibcpp::legend();
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./{}/diagnostics-temperal-velocity_x.png", subfolder));

	// Plot rotation radius
	matplotlibcpp::clf();
	std::map<std::string, std::string> rotation_radius_args;
	rotation_radius_args["label"] = "rotation_radius";
	rotation_radius_args["linestyle"] = "-";
	matplotlibcpp::plot(sim.userData_.time, sim.userData_.rotation_radius, rotation_radius_args);
	matplotlibcpp::xlabel("time");
	matplotlibcpp::ylabel("rotation_radius");
	// matplotlibcpp::legend();
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./{}/diagnostics-temperal-rotation_radius.png", subfolder));

	// Plot spin angular momentum
	matplotlibcpp::clf();
	std::map<std::string, std::string> spin_angular_mtm_args;
	spin_angular_mtm_args["label"] = "spin_angular_mtm";
	spin_angular_mtm_args["linestyle"] = "-";
	matplotlibcpp::plot(sim.userData_.time, sim.userData_.spin_angular_mtm, spin_angular_mtm_args);
	matplotlibcpp::xlabel("time");
	matplotlibcpp::ylabel("spin_angular_mtm");
	// matplotlibcpp::legend();
	matplotlibcpp::tight_layout();
	matplotlibcpp::save(fmt::format("./{}/diagnostics-temperal-spin_angular_mtm.png", subfolder));
#endif

	// Cleanup and exit
	int status = 0;
	// if (test_passes) {
	// 	status = 0;
	// } else {
	// 	status = 1;
	// }
	return status;
}
