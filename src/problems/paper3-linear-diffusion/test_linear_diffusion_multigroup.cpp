//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_marshak.cpp
/// \brief Defines a test problem for radiation in the diffusion regime.
///


// step 1: make it CGS unit


#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"

#include "util/fextract.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "test_linear_diffusion_multigroup.hpp"

struct TheProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

constexpr int max_timesteps_ = 1e6;

// nu_0 = 1.0
// u_0 = 1.0
constexpr double a_rad = 1.0;
// constexpr double a_rad = 6.0; // because of the modified Planck function
constexpr double c = 1.7320508075688772;  // = sqrt(3.0)
constexpr double h = c * c * c / (8.0 * M_PI); // B_0 = 8 * pi * h / c^3 = 1
constexpr double k_B = h;
constexpr double C_v = 1.0;
constexpr double mu = 3.0 / 2.0 * k_B;    // C_v = 3/2 k_B / mu = 1, so mu = 3/2 * k_B
constexpr double kappa_0 = 1.0 / 1.7320508075688772;  // = 1.0 / sqrt(3.0)
// constexpr double kappa_0 = 1.0e20;

constexpr double rho0 = 1.0;
constexpr double x_max = 4.0;
constexpr double x0 = 0.5;
constexpr double t1 = 1.0;
constexpr double T0 = 1.0;
constexpr double T_f = 0.1;
// constexpr double T_f = 1.0;
constexpr double T_floor = 1.0e-12 * T0;
constexpr double Egas_floor = C_v * rho0 * T_floor;
constexpr double Erad_floor_ = 1.0e-12;

constexpr int n_groups_ = 64;
// constexpr int n_groups_ = 7;

template <> struct quokka::EOS_Traits<TheProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<TheProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = false;
	static constexpr bool is_chemistry_enabled = false;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = n_groups_;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gravitational_constant = 1.0;
	static constexpr double c_light = c;
	static constexpr double radiation_constant = a_rad;
};

template <> struct RadSystem_Traits<TheProblem> {
	static constexpr double c_light = c;
	static constexpr double c_hat = c;
	static constexpr double radiation_constant = a_rad;
	static constexpr double mean_molecular_mass = mu;
	static constexpr double boltzmann_constant = k_B;
	static constexpr double gamma = 5. / 3.;
	static constexpr double Erad_floor = Erad_floor_;
	// static constexpr bool compute_v_over_c_terms = true;
	static constexpr int beta_order = 0;
	static constexpr double energy_unit = h;
	static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries{1.0e-8, 0.0005, 0.0010500000000000002, 0.0016550000000000002, 
		0.0023205000000000005, 0.0030525500000000007, 0.003857805000000001, 0.004743585500000001, 0.0057179440500000015, 
		0.006789738455000002, 0.007968712300500003, 0.009265583530550004, 0.010692141883605006, 0.012261356071965508, 
		0.01398749167916206, 0.015886240847078265, 0.017974864931786093, 0.020272351424964703, 0.022799586567461175, 
		0.025579545224207294, 0.028637499746628027, 0.032001249721290835, 0.03570137469341992, 0.03977151216276192, 
		0.044248663379038114, 0.04917352971694193, 0.05459088268863613, 0.06054997095749974, 0.06710496805324973, 
		0.07431546485857471, 0.08224701134443219, 0.09097171247887541, 0.10056888372676295, 0.11112577209943926, 
		0.1227383493093832, 0.13551218424032152, 0.1495634026643537, 0.16501974293078908, 0.182021717223868, 0.2007238889462548, 
		0.22129627784088032, 0.24392590562496838, 0.26881849618746523, 0.2962003458062118, 0.326320380386833, 0.3594524184255163, 
		0.39589766026806794, 0.43598742629487475, 0.4800861689243623, 0.5285947858167985, 0.5819542643984784, 0.6406496908383263, 
		0.705214659922159, 0.7762361259143751, 0.8543597385058127, 0.9402957123563941, 1.0348252835920335, 1.138807811951237, 
		1.2531885931463609, 1.3790074524609972, 1.517408197707097, 1.669649017477807, 1.8371139192255879, 2.021325311148147, 
		2.2239578422629616
		// 1.0e5
		// 2.1
		};
	// static constexpr amrex::GpuArray<double, n_groups_ + 1> radBoundaries{1.0e-6, 0.0005, 0.005, 0.05, 0.5, 5.0, 50.0, 500.0};
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr bool disable_force = false;
};

// old
template <>
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::ComputeThermalRadiationMultiGroup(amrex::Real temperature, amrex::GpuArray<double, nGroups_ + 1> const &boundaries)
    -> quokka::valarray<amrex::Real, nGroups_>
{
	quokka::valarray<double, nGroups_> B_g{};
  double nu_g = NAN;
  const double coeff = 1.0;   // = B0 * h / k_B
	for (int g = 0; g < nGroups_; ++g) {
    if (g == 0) {
      nu_g = 0.5 * RadSystem_Traits<TheProblem>::radBoundaries[1];
    } else {
      // take the geometrical mean
      nu_g = std::sqrt(RadSystem_Traits<TheProblem>::radBoundaries[g] * RadSystem_Traits<TheProblem>::radBoundaries[g + 1]);
    }
    auto tmp = coeff * std::pow(nu_g, 3);
    auto frac = std::exp(- boundaries[g] / T_f) - std::exp(- boundaries[g + 1] / T_f);
    B_g[g] = tmp * frac * temperature;
	}
	return B_g;   // = a_r * T^4 = 4 * pi * B / c
}

template <>
AMREX_GPU_HOST_DEVICE auto
RadSystem<TheProblem>::ComputeThermalRadiationTempDerivativeMultiGroup(amrex::Real temperature,
							    amrex::GpuArray<double, nGroups_ + 1> const &boundaries) -> quokka::valarray<amrex::Real, nGroups_>
{
	// by default, d emission/dT = 4 emission / T
	auto erad = ComputeThermalRadiationMultiGroup(temperature, boundaries);
	return 4. * erad / temperature;
}

template <> 
AMREX_GPU_HOST_DEVICE auto RadSystem<TheProblem>::PlanckFunction(const double nu, const double T) -> double
{
	// returns 4 pi B(nu) / c
	// return std::pow(nu, 3) * std::exp(- nu / T);
	return 0.0;
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<TheProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double rho, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
  double nu_g = NAN;
	for (int g = 0; g < nGroups_ + 1; ++g) {
		exponents_and_values[0][g] = 0.0;
    if (g == 0) {
      nu_g = 0.5 * RadSystem_Traits<TheProblem>::radBoundaries[1];
    } else if (g < nGroups_) {
      // take the geometrical mean
      nu_g = std::sqrt(RadSystem_Traits<TheProblem>::radBoundaries[g] * RadSystem_Traits<TheProblem>::radBoundaries[g + 1]);
    } else {
			nu_g = RadSystem_Traits<TheProblem>::radBoundaries[nGroups_];
		}
    auto const kappa = kappa_0 * std::pow(nu_g, -3);
		exponents_and_values[1][g] = kappa / rho;
	}
	return exponents_and_values;
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<TheProblem>::ComputeEddingtonFactor(double /*f*/) -> double
{
	return (1. / 3.); // Eddington approximation
}

template <> void QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + 0.5) * dx[0];

    double const T = std::abs(x) > x0 ? T_floor : T0;
    auto Egas = quokka::EOS<TheProblem>::ComputeEintFromTgas(rho0, T);
    
		for (int g = 0; g < n_groups_; ++g) {
			state_cc(i, j, k, RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_floor_;
			state_cc(i, j, k, RadSystem<TheProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TheProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}

		state_cc(i, j, k, RadSystem<TheProblem>::gasEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::gasDensity_index) = rho0;
		state_cc(i, j, k, RadSystem<TheProblem>::gasInternalEnergy_index) = Egas;
		state_cc(i, j, k, RadSystem<TheProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TheProblem>::x3GasMomentum_index) = 0.;
	});
}

auto problem_main() -> int
{
	// Problem parameters

	const int max_timesteps = max_timesteps_;
	// const double CFL_number = 0.8;
	const double initial_dt = 0.005;
	// const double max_dt = 0.005;
	const double max_time = 1.0;

	auto isNormalComp = [=](int n, int dim) {
		if ((n == RadSystem<TheProblem>::x1RadFlux_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x2RadFlux_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x3RadFlux_index) && (dim == 2)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x1GasMomentum_index) && (dim == 0)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x2GasMomentum_index) && (dim == 1)) {
			return true;
		}
		if ((n == RadSystem<TheProblem>::x3GasMomentum_index) && (dim == 2)) {
			return true;
		}
		return false;
	};

	// Boundary conditions
	constexpr int nvars = RadSystem<TheProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		if (isNormalComp(n, 0)) {
			BCs_cc[n].setLo(0, amrex::BCType::reflect_odd);
		} else {
			BCs_cc[n].setLo(0, amrex::BCType::reflect_even);
		}
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // extrapolate
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {	    // x2- and x3- directions
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<TheProblem> sim(BCs_cc);

	// sim.cflNumber_ = CFL_number;
	// sim.radiationCflNumber_ = CFL_number;
	sim.stopTime_ = max_time;
	sim.maxTimesteps_ = max_timesteps;
	sim.initDt_ = initial_dt;
	// sim.maxDt_ = max_dt;
	sim.plotfileInterval_ = -1;

	// evolve
	sim.setInitialConditions();
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	int nx = static_cast<int>(position.size());

	// compare with exact solution
	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {

		// exact solution
		std::vector<double> x_exact = {0.0000000E+00 , 2.0000000E-01 , 4.0000000E-01 , 4.6000000E-01 , 4.7000000E-01 , 4.8000000E-01 , 4.9000000E-01 , 5.0000000E-01 , 5.1000000E-01 , 5.2000000E-01 , 5.3000000E-01 , 5.4000000E-01 , 6.0000000E-01 , 8.0000000E-01 , 1.0000000E+00};
		std::vector<double> T_exact = {9.9373253E-01, 9.9339523E-01, 9.8969664E-01, 9.8060848E-01, 9.7609654E-01, 9.6819424E-01, 9.5044751E-01, 4.9704000E-01, 4.3632445E-02, 2.5885608E-02, 1.7983134E-02, 1.3470947E-02, 4.3797848E-03, 6.4654865E-04, 1.9181546E-04};
		std::vector<double> Er_exact = {5.6401674E-03, 5.5646351E-03, 5.1047352E-03, 4.5542134E-03, 4.3744933E-03, 4.1294850E-03, 3.7570008E-03, 2.9096931E-03, 2.0623647E-03, 1.6898183E-03, 1.4447063E-03, 1.2648409E-03, 7.1255738E-04, 2.3412650E-04, 1.0934921E-04};

		std::vector<double> xs(nx);
		std::vector<double> Trad(nx);
		std::vector<double> Tgas(nx);
		std::vector<double> Erad(nx);
		std::vector<double> Egas(nx);

		for (int i = 0; i < nx; ++i) {
			xs.at(i) = position[i];
      double Erad_t = 0.;
      for (int g = 0; g < n_groups_; ++g) {
        Erad_t += values.at(RadSystem<TheProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
      }
      Erad.at(i) = Erad_t;
			const auto rho = values.at(RadSystem<TheProblem>::gasDensity_index)[i];
      const auto Egas_t = values.at(RadSystem<TheProblem>::gasEnergy_index)[i];
      const auto Tgas_t = quokka::EOS<TheProblem>::ComputeTgasFromEint(rho, Egas_t);
			Egas.at(i) = Egas_t;
			Tgas.at(i) = Tgas_t;
		}

		// save data to file
		std::ofstream fstream;
		fstream.open(fmt::format("./LinearDiffusionMP_nx{}.csv", nx));
		fstream << "# x,Tgas,Erad\n";
		for (int i = 0; i < nx; ++i) {
			fstream << xs.at(i) << "," << Tgas.at(i) << "," << Erad.at(i) << "\n";
		}
		fstream.close();
		// exact
		fstream.open("./LinearDiffusionMP_exact.csv");
		fstream << "# x,Tgas,Erad\n";
		for (int i = 0; i < x_exact.size(); ++i) {
			fstream << x_exact.at(i) << "," << T_exact.at(i) << "," << Er_exact.at(i) << "\n";
		}
		fstream.close();

#ifdef HAVE_PYTHON
		// Plot solution

		matplotlibcpp::clf();

		std::map<std::string, std::string> args;

    // temperature
    matplotlibcpp::clf();
		std::unordered_map<std::string, std::string> exact_args;
    args["label"] = "numerical solution";
		exact_args["label"] = "exact diffusion solution";
		exact_args["color"] = "C1";
    matplotlibcpp::plot(xs, Tgas, args);
    matplotlibcpp::scatter(x_exact, T_exact, 10., exact_args);
		// matplotlibcpp::legend();
		matplotlibcpp::xlabel("x (dimensionless)");
		matplotlibcpp::ylabel(R"($T_{\rm gas}$ (dimensionless))");
		// matplotlibcpp::title(fmt::format("t = {:.4g}", sim.tNew_[0]));
		matplotlibcpp::xlim(0.0, 1.0);
		matplotlibcpp::ylim(-0.05, 1.2);	// dimensionless
    matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save(fmt::format("./LinearDiffusionMP_Tgas_nx{}.pdf", nx));

    matplotlibcpp::clf();
    args["label"] = "numerical solution";
		exact_args["label"] = "exact diffusion solution";
		exact_args["color"] = "C1";
    matplotlibcpp::plot(xs, Erad, args);
		matplotlibcpp::scatter(x_exact, Er_exact, 10., exact_args);
    matplotlibcpp::xlabel("x (dimensionless)");
    matplotlibcpp::ylabel(R"($E_{\rm rad}$ (dimensionless))");
    // matplotlibcpp::title(fmt::format("t = {:.4g}", sim.tNew_[0]));
    matplotlibcpp::xlim(0.0, 1.0);
    matplotlibcpp::ylim(0.0, 0.007);	// dimensionless
    matplotlibcpp::legend();
    matplotlibcpp::tight_layout();
		matplotlibcpp::save(fmt::format("./LinearDiffusionMP_Erad_nx{}.pdf", nx));
#endif
	}

	return status;
}
