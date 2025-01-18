/// \file test_radiation_marshak_vaytet.cpp
/// \brief Defines a Marshak wave problem with variable opacity.
///

#include "AMReX_BLassert.H"

#include "QuokkaSimulation.hpp"
#include "hydro/EOS.hpp"
#include "radiation/radiation_system.hpp"
#include "test_radiation_marshak_Vaytet.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

// const double chat_over_c_ = 0.1;
constexpr bool variable_density_IC = true;
constexpr double rho0 = 1.0e-3; // g cm^-3
// variable density
constexpr double rhoL = rho0;	    // g cm^-3
constexpr double rhoR = 0.1 * rho0; // g cm^-3
// constant density
// constexpr double rhoL = rho0;
// constexpr double rhoR = rho0;

// constexpr int n_groups_ = 2; // Be careful
// constexpr int n_groups_ = 4;
constexpr int n_groups_ = 8;
// constexpr int n_groups_ = 16;
// constexpr int n_groups_ = 64;
constexpr OpacityModel the_opacity_model = OpacityModel::piecewise_constant_opacity;
// constexpr OpacityModel the_opacity_model = OpacityModel::PPL_opacity_fixed_slope_spectrum;
// constexpr OpacityModel the_opacity_model = OpacityModel::PPL_opacity_full_spectrum;

constexpr double kappa0 = 2000.0;   // cm^2 g^-1 (opacity).
constexpr double nu_pivot = 4.0e13; // Powerlaw, kappa = kappa0 (nu/nu_pivot)^{-2}
constexpr int n_coll = 4;	    // number of collections = 6, to align with Vaytet
constexpr int the_model = 10;
// 0: constant opacity (Vaytet et al. Sec 3.2.1), 1: nu-dependent opacity (Vaytet et al. Sec 3.2.2), 2: nu-and-T-dependent opacity
// (Vaytet et al. Sec 3.2.3)
// 10: bin-centered method with opacities propto nu^-2, kappa = kappa0 * (nu_c/nu_pivot)^{-2}, nu_c = sqrt(nu_i * nu_i+1)

// OLD
// constexpr int the_model = 0; // 0: constant opacity (Vaytet et al. Sec 3.2.1), 1: nu-dependent opacity (Vaytet et al. Sec 3.2.2), 2: nu-and-T-dependent
// opacity (Vaytet et al. Sec 3.2.3) constexpr int n_groups_ = 6; constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = {0.3e12, 0.3e14, 0.6e14,
// 0.9e14, 1.2e14, 1.5e14, 1.5e16}; constexpr amrex::GpuArray<double, n_groups_> group_opacities_ = {1000., 750., 500., 250., 10., 10.};

// NEW
constexpr amrex::GpuArray<double, n_groups_ + 1> group_edges_ = []() constexpr {
	// bins cover four orders of magnitutde from 6.0e10 to 6.0e14, with N bins logarithmically spaced
	// in the space of x = h nu / (k_B T) where T = 1000 K, this roughly corresponds to x = 3e-3 to 3e1
	// n_groups_ = 2, 4, 8, 16, 128, 256
	if constexpr (n_groups_ == 2) {
		return amrex::GpuArray<double, 3>{6.0e10, 6.0e12, 6.0e14};
	} else if constexpr (n_groups_ == 4) {
		return amrex::GpuArray<double, 5>{6.0e10, 6.0e11, 6.0e12, 6.0e13, 6.0e14};
	} else if constexpr (n_groups_ == 8) {
		return amrex::GpuArray<double, 9>{6.0000000e+10, 1.8973666e+11, 6.0000000e+11, 1.8973666e+12, 6.0000000e+12,
						  1.8973666e+13, 6.0000000e+13, 1.8973666e+14, 6.0000000e+14};
	} else if constexpr (n_groups_ == 16) {
		return amrex::GpuArray<double, 17>{6.00000000e+10, 1.06696765e+11, 1.89736660e+11, 3.37404795e+11, 6.00000000e+11, 1.06696765e+12,
						   1.89736660e+12, 3.37404795e+12, 6.00000000e+12, 1.06696765e+13, 1.89736660e+13, 3.37404795e+13,
						   6.00000000e+13, 1.06696765e+14, 1.89736660e+14, 3.37404795e+14, 6.00000000e+14};
	} else if constexpr (n_groups_ == 64) {
		return amrex::GpuArray<double, 65>{
		    6.00000000e+10, 6.92869191e+10, 8.00112859e+10, 9.23955916e+10, 1.06696765e+11, 1.23211502e+11, 1.42282422e+11, 1.64305178e+11,
		    1.89736660e+11, 2.19104476e+11, 2.53017902e+11, 2.92180515e+11, 3.37404795e+11, 3.89628979e+11, 4.49936526e+11, 5.19578594e+11,
		    6.00000000e+11, 6.92869191e+11, 8.00112859e+11, 9.23955916e+11, 1.06696765e+12, 1.23211502e+12, 1.42282422e+12, 1.64305178e+12,
		    1.89736660e+12, 2.19104476e+12, 2.53017902e+12, 2.92180515e+12, 3.37404795e+12, 3.89628979e+12, 4.49936526e+12, 5.19578594e+12,
		    6.00000000e+12, 6.92869191e+12, 8.00112859e+12, 9.23955916e+12, 1.06696765e+13, 1.23211502e+13, 1.42282422e+13, 1.64305178e+13,
		    1.89736660e+13, 2.19104476e+13, 2.53017902e+13, 2.92180515e+13, 3.37404795e+13, 3.89628979e+13, 4.49936526e+13, 5.19578594e+13,
		    6.00000000e+13, 6.92869191e+13, 8.00112859e+13, 9.23955916e+13, 1.06696765e+14, 1.23211502e+14, 1.42282422e+14, 1.64305178e+14,
		    1.89736660e+14, 2.19104476e+14, 2.53017902e+14, 2.92180515e+14, 3.37404795e+14, 3.89628979e+14, 4.49936526e+14, 5.19578594e+14,
		    6.00000000e+14};
	}
}();

constexpr amrex::GpuArray<double, n_groups_> group_opacities_{};

struct SuOlsonProblemCgs {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int max_step_ = 1e6;
constexpr double T_initial = 300.0; // K
constexpr double T_L = 1000.0;	    // K
constexpr double T_R = 300.0;	    // K
constexpr double rho_C_V = 1.0e-3;  // erg cm^-3 K^-1
constexpr double c_v = rho_C_V / rho0;
constexpr double mu = 1.0 / (5. / 3. - 1.) * C::k_B / c_v;

constexpr double a_rad = radiation_constant_cgs_;
constexpr double Erad_floor_ = a_rad * T_initial * T_initial * T_initial * T_initial * 1e-20;

template <> struct quokka::EOS_Traits<SuOlsonProblemCgs> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<SuOlsonProblemCgs> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_; // number of radiation groups
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<SuOlsonProblemCgs> {
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = C::hplanck; // set boundary unit to Hz
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries = group_edges_;
	static constexpr OpacityModel opacity_model = the_opacity_model;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<SuOlsonProblemCgs>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> rad_boundaries, const double /*rho*/,
								   const double Tgas) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		if constexpr (the_model < 10) {
			exponents_and_values[0][i] = 0.0;
		} else {
			exponents_and_values[0][i] = -2.0;
		}
	}
	if constexpr (the_model == 0) {
		for (int i = 0; i < nGroups_; ++i) {
			exponents_and_values[1][i] = kappa0;
		}
	} else if constexpr (the_model == 1) {
		for (int i = 0; i < nGroups_; ++i) {
			exponents_and_values[1][i] = group_opacities_[i];
		}
	} else if constexpr (the_model == 2) {
		for (int i = 0; i < nGroups_; ++i) {
			exponents_and_values[1][i] = group_opacities_[i] * std::pow(Tgas / T_initial, 3. / 2.);
		}
	} else if constexpr (the_model == 10) {
		if constexpr (the_opacity_model == OpacityModel::piecewise_constant_opacity) {
			for (int i = 0; i < nGroups_; ++i) {
				auto const bin_center = std::sqrt(rad_boundaries[i] * rad_boundaries[i + 1]);
				exponents_and_values[1][i] = kappa0 * std::pow(bin_center / nu_pivot, -2.);
			}
		} else {
			for (int i = 0; i < nGroups_ + 1; ++i) {
				exponents_and_values[1][i] = kappa0 * std::pow(rad_boundaries[i] / nu_pivot, -2.);
			}
		}
	}
	return exponents_and_values;
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<SuOlsonProblemCgs>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/,
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

	auto const radBoundaries_g = RadSystem<SuOlsonProblemCgs>::radBoundaries_;

	if (i < lo[0] || i >= hi[0]) {
		double T_H = NAN;
		double rho_H = NAN;
		if (i < lo[0]) {
			T_H = T_L;
			rho_H = rhoL;
		} else {
			T_H = T_R;
			rho_H = rhoR;
		}

		auto Erad_g = RadSystem<SuOlsonProblemCgs>::ComputeThermalRadiationMultiGroup(T_H, radBoundaries_g);
		const double Egas = quokka::EOS<SuOlsonProblemCgs>::ComputeEintFromTgas(rho_H, T_initial);

		for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}

		// gas boundary conditions are the same on both sides
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasDensity_index) = rho_H;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<SuOlsonProblemCgs>::x3GasMomentum_index) = 0.;
	}
}

template <> void QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;

	const double Lx = prob_hi[0] - prob_lo[0];

	const auto radBoundaries_g = RadSystem<SuOlsonProblemCgs>::radBoundaries_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];

		const double rho = rhoL + (rhoR - rhoL) * (x / Lx);
		const double Egas = quokka::EOS<SuOlsonProblemCgs>::ComputeEintFromTgas(rho, T_initial);
		// const double Erad = a_rad * std::pow(T_initial, 4);
		auto Erad_g = RadSystem<SuOlsonProblemCgs>::ComputeThermalRadiationMultiGroup(T_initial, radBoundaries_g);

		for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<SuOlsonProblemCgs>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// This problem tests whether the M1 closure can handle a Marshak wave problem with variable opacity
	// as accurately as ray-tracing method.

	// Problem parameters
	const int max_timesteps = max_step_;
	const double CFL_number = 0.8;
	// const double initial_dt = 5.0e-12; // s
	const double max_dt = 1.0; // s
	const double max_time = 1.36e-7;

	constexpr int nvars = RadSystem<SuOlsonProblemCgs>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0,
				amrex::BCType::ext_dir);     // custom (Marshak) x1
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<SuOlsonProblemCgs> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.stopTime_ = max_time;
	// sim.initDt_ = initial_dt;
	sim.maxDt_ = max_dt;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position.size());

	// compare against diffusion solution
	const int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> Trad(nx);
		std::vector<double> Tgas(nx);
		std::vector<double> vgas(nx);
		std::vector<double> rho_gas(nx);
		// define a vector of n_groups_ vectors
		std::vector<std::vector<double>> Trad_g(n_groups_);

		int const n_item = n_groups_ / n_coll;
		std::vector<std::vector<double>> Trad_coll(n_coll);

		for (int i = 0; i < nx; ++i) {
			double Erad_t = 0.;
			// const double Erad_t = values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index)[i];
			for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
				Erad_t += values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
			}
			const double Egas_t = values.at(RadSystem<SuOlsonProblemCgs>::gasInternalEnergy_index)[i];
			const double rho = values.at(RadSystem<SuOlsonProblemCgs>::gasDensity_index)[i];
			rho_gas.at(i) = rho;
			amrex::Real const x = position[i];
			xs.at(i) = x;
			Tgas.at(i) = quokka::EOS<SuOlsonProblemCgs>::ComputeTgasFromEint(rho, Egas_t);
			Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.);

			// vgas
			const auto x1GasMomentum = values.at(RadSystem<SuOlsonProblemCgs>::x1GasMomentum_index)[i];
			vgas.at(i) = x1GasMomentum / rho;

			int counter = 0;
			int group_counter = 0;
			double Erad_sum = 0.;
			for (int g = 0; g < Physics_Traits<SuOlsonProblemCgs>::nGroups; ++g) {
				auto Erad_g = values.at(RadSystem<SuOlsonProblemCgs>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
				Trad_g[g].push_back(std::pow(Erad_g / a_rad, 1. / 4.));

				if (counter == 0) {
					Erad_sum = 0.0;
				}
				Erad_sum += Erad_g;
				if (counter == n_item - 1) {
					Trad_coll[group_counter].push_back(std::pow(Erad_sum / a_rad, 1. / 4.));
					group_counter++;
					counter = 0;
				} else {
					counter++;
				}
			}
		}

		// // read in exact solution

		// std::vector<double> xs_exact;
		// std::vector<double> Tmat_exact;

		// std::string filename = "../extern/marshak_similarity.csv";
		// std::ifstream fstream(filename, std::ios::in);
		// AMREX_ALWAYS_ASSERT(fstream.is_open());

		// std::string header;
		// std::getline(fstream, header);

		// for (std::string line; std::getline(fstream, line);) {
		// 	std::istringstream iss(line);
		// 	std::vector<double> values;

		// 	for (double value = NAN; iss >> value;) {
		// 		values.push_back(value);
		// 	}
		// 	auto x_val = values.at(0);
		// 	auto Tmat_val = values.at(1);

		// 	xs_exact.push_back(x_val);
		// 	Tmat_exact.push_back(Tmat_val);
		// }

		// // compute error norm

		// // interpolate numerical solution onto exact tabulated solution
		// std::vector<double> Tmat_interp(xs_exact.size());
		// interpolate_arrays(xs_exact.data(), Tmat_interp.data(), static_cast<int>(xs_exact.size()), xs.data(), Tgas.data(),
		// static_cast<int>(xs.size()));

		// double err_norm = 0.;
		// double sol_norm = 0.;
		// for (size_t i = 0; i < xs_exact.size(); ++i) {
		// 	err_norm += std::abs(Tmat_interp[i] - Tmat_exact[i]);
		// 	sol_norm += std::abs(Tmat_exact[i]);
		// }

		// const double error_tol = 0.09;
		// const double rel_error = err_norm / sol_norm;
		// amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;

		// save data to file
		std::ofstream fstream;
		fstream.open("marshak_wave_Vaytet.csv");
		fstream << "# x, Tgas, Trad";
		for (int i = 0; i < n_groups_; ++i) {
			fstream << ", "
				<< "Trad_" << i;
		}
		for (int i = 0; i < nx; ++i) {
			fstream << std::endl;
			fstream << std::scientific << std::setprecision(14) << xs[i] << ", " << Tgas[i] << ", " << Trad[i];
			for (int j = 0; j < n_groups_; ++j) {
				fstream << ", " << Trad_g[j][i];
			}
		}
		fstream.close();

		// save Trad_coll to file
		// Trad_coll is the effective temperature of the total radiation energy in the n_coll (4) set of groups
		std::ofstream fstream_coll;
		fstream_coll.open("marshak_wave_Vaytet_coll.csv");
		fstream_coll << "# x, Tgas, Trad";
		for (int i = 0; i < n_coll; ++i) {
			fstream_coll << ", "
				     << "Trad_" << i;
		}
		for (int i = 0; i < nx; ++i) {
			fstream_coll << std::endl;
			fstream_coll << std::scientific << std::setprecision(14) << xs[i] << ", " << Tgas[i] << ", " << Trad[i];
			for (int j = 0; j < n_coll; ++j) {
				fstream_coll << ", " << Trad_coll[j][i];
			}
		}
		fstream_coll.close();

		// // check if velocity is strictly zero
		// const double error_v_tol = 1.0e-10;
		// double error_v = 0.0;
		// const double cs = std::sqrt(5. / 3. * C::k_B / mu * T_initial); // sound speed
		// for (size_t i = 0; i < xs.size(); ++i) {
		// 	error_v += std::abs(vgas[i]) / cs;
		// }
		// amrex::Print() << "Sum of abs(v) / cs = " << error_v << std::endl;
		// if ((error_v > error_v_tol) || std::isnan(error_v)) {
		// 	status = 1;
		// }

#ifdef HAVE_PYTHON
		// plot results
		matplotlibcpp::clf();
		std::map<std::string, std::string> args;
		args["label"] = "gas";
		args["linestyle"] = "-";
		args["color"] = "k";
		matplotlibcpp::plot(xs, Tgas, args);
		args["label"] = "radiation";
		args["linestyle"] = "--";
		args["color"] = "k";
		// args["marker"] = "x";
		matplotlibcpp::plot(xs, Trad, args);

		for (int g = 0; g < n_coll; ++g) {
			std::map<std::string, std::string> Trad_coll_args;
			Trad_coll_args["label"] = fmt::format("group {}", g);
			Trad_coll_args["linestyle"] = "-";
			Trad_coll_args["color"] = "C" + std::to_string(g);
			matplotlibcpp::plot(xs, Trad_coll[g], Trad_coll_args);
		}

		// Tgas_exact_args["label"] = "gas temperature (exact)";
		// Tgas_exact_args["color"] = "C0";
		// // Tgas_exact_args["marker"] = "x";
		// matplotlibcpp::plot(xs_exact, Tmat_exact, Tgas_exact_args);

		matplotlibcpp::xlim(0.0, 20.0);	  // cm
		matplotlibcpp::ylim(0.0, 1000.0); // K
		matplotlibcpp::xlabel("length x (cm)");
		matplotlibcpp::ylabel("temperature (K)");
		matplotlibcpp::legend();
		// matplotlibcpp::title(fmt::format("time t = {:.4g}", sim.tNew_[0]));
		if (the_opacity_model == OpacityModel::piecewise_constant_opacity) {
			matplotlibcpp::title("PC");
		} else if (the_opacity_model == OpacityModel::PPL_opacity_fixed_slope_spectrum) {
			matplotlibcpp::title("PPL-fixed");
		} else if (the_opacity_model == OpacityModel::PPL_opacity_full_spectrum) {
			matplotlibcpp::title("PPL-free");
		}
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./marshak_wave_Vaytet.pdf");

		// plot rho_gas
		matplotlibcpp::clf();
		matplotlibcpp::plot(xs, rho_gas);
		matplotlibcpp::xlabel("length x (cm)");
		matplotlibcpp::ylabel("density (g cm^-3)");
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./marshak_wave_Vaytet_rho.pdf");
#endif // HAVE_PYTHON
	}

	// Cleanup and exit
	std::cout << "Finished." << std::endl;

	// if ((rel_error > error_tol) || std::isnan(rel_error)) {
	// 	status = 1;
	// }
	return status;
}
