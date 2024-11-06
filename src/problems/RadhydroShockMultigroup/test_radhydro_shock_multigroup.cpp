/// \file test_radhydro_shock.cpp
/// \brief Defines a test problem for a multigroup gray-opacity non-equilibrium radiative shock.
///

#include <cmath>

#include "AMReX_Array.H"
#include "AMReX_BC_TYPES.H"

#include "radiation/radiation_system.hpp"
#include "test_radhydro_shock_multigroup.hpp"
#include "util/ArrayUtil.hpp"
#include "util/fextract.hpp"

struct ShockProblem {
}; // dummy type to allow compile-type polymorphism via template specialization

// parameters taken from Section 9.5 of Skinner et al. (2019)
// https://doi.org/10.3847/1538-4365/ab007f

constexpr double a_rad = C::a_rad; // erg cm^-3 K^-4
constexpr double c = C::c_light;   // cm s^-1
constexpr double k_B = C::k_B;	   // erg K^-1

constexpr double c_s0 = 1.73e7; // adiabatic sound speed [cm s^-1]
constexpr double kappa = 577.0; // "opacity" == rho*kappa [cm^-1] (!!)
constexpr double gamma_gas = (5. / 3.);
constexpr double c_v = k_B / ((C::m_p + C::m_e) * (gamma_gas - 1.0)); // specific heat [erg g^-1 K^-1]
constexpr double T0 = 2.18e6;					      // K
constexpr double rho0 = 5.69;					      // g cm^-3
constexpr double v0 = 5.19e7;					      // cm s^-1
constexpr double T1 = 7.98e6;					      // K [7.98297e6]
constexpr double rho1 = 17.1;					      // g cm^-3 [17.08233]
constexpr double v1 = 1.73e7;					      // cm s^-1 [1.72875e7]
constexpr double chat = 10.0 * (v0 + c_s0);			      // reduced speed of light

constexpr double Erad0 = a_rad * (T0 * T0 * T0 * T0); // erg cm^-3
constexpr double Erad_floor_ = Erad0 * 1e-12;
constexpr double Egas0 = rho0 * c_v * T0; // erg cm^-3
constexpr double Egas1 = rho1 * c_v * T1; // erg cm^-3

constexpr double shock_position = 0.0130; // 0.0132; // cm (shock position drifts to the right slightly during the simulation, so
					  // we initialize slightly to the left...)
constexpr double Lx = 0.01575;		  // cm

template <> struct Physics_Traits<ShockProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	static constexpr int nGroups = 5;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<ShockProblem> {
	static constexpr double c_hat_over_c = chat / c;
	static constexpr double Erad_floor = Erad_floor_;
	static constexpr double energy_unit = C::hplanck; // set boundary unit to Hz
	static constexpr amrex::GpuArray<double, Physics_Traits<ShockProblem>::nGroups + 1> radBoundaries{1.00000000e+15, 1.00000000e+16, 1.00000000e+17,
													  1.00000000e+18, 1.00000000e+19, 1.00000000e+20};
	static constexpr int beta_order = 1;
	// static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_fixed_slope_spectrum;
	// static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_full_spectrum;
};

template <> struct quokka::EOS_Traits<ShockProblem> {
	static constexpr double mean_molecular_weight = C::m_p + C::m_e;
	static constexpr double gamma = gamma_gas;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<ShockProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double rho, const double /*Tgas*/)
    -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = kappa / rho;
	}
	return exponents_and_values;
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<ShockProblem>::ComputeEddingtonFactor(double /*f*/) -> double
{
	return (1. / 3.); // Eddington approximation
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<ShockProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							 amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec *bcr, int /*bcomp*/,
							 int /*orig_comp*/)
{
	if (!((bcr->lo(0) == amrex::BCType::ext_dir) || (bcr->hi(0) == amrex::BCType::ext_dir))) { // NOLINT
		return;
	}

#if (AMREX_SPACEDIM == 1)
	auto i = iv.toArray()[0];
	int const j = 0;
	int const k = 0;
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

	auto const radBoundaries_g = RadSystem<ShockProblem>::radBoundaries_;

	if (i < lo[0]) {
		const double px_L = rho0 * v0;
		const double Egas_L = Egas0;

		auto Erad_g = RadSystem<ShockProblem>::ComputeThermalRadiationMultiGroup(T0, radBoundaries_g);

		// x1 left side boundary -- constant
		consVar(i, j, k, RadSystem<ShockProblem>::gasDensity_index) = rho0;
		consVar(i, j, k, RadSystem<ShockProblem>::gasInternalEnergy_index) = 0.;
		consVar(i, j, k, RadSystem<ShockProblem>::x1GasMomentum_index) = px_L;
		consVar(i, j, k, RadSystem<ShockProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<ShockProblem>::x3GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<ShockProblem>::gasEnergy_index) = Egas_L + (px_L * px_L) / (2 * rho0);
		consVar(i, j, k, RadSystem<ShockProblem>::gasInternalEnergy_index) = Egas_L;
		for (int g = 0; g < Physics_Traits<ShockProblem>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<ShockProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			consVar(i, j, k, RadSystem<ShockProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<ShockProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<ShockProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
	} else if (i >= hi[0]) {
		const double px_R = rho1 * v1;
		const double Egas_R = Egas1;

		auto Erad_g = RadSystem<ShockProblem>::ComputeThermalRadiationMultiGroup(T1, radBoundaries_g);

		// x1 right-side boundary -- constant
		consVar(i, j, k, RadSystem<ShockProblem>::gasDensity_index) = rho1;
		consVar(i, j, k, RadSystem<ShockProblem>::gasInternalEnergy_index) = 0.;
		consVar(i, j, k, RadSystem<ShockProblem>::x1GasMomentum_index) = px_R;
		consVar(i, j, k, RadSystem<ShockProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<ShockProblem>::x3GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<ShockProblem>::gasEnergy_index) = Egas_R + (px_R * px_R) / (2 * rho1);
		consVar(i, j, k, RadSystem<ShockProblem>::gasInternalEnergy_index) = Egas_R;
		for (int g = 0; g < Physics_Traits<ShockProblem>::nGroups; ++g) {
			consVar(i, j, k, RadSystem<ShockProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			consVar(i, j, k, RadSystem<ShockProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<ShockProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			consVar(i, j, k, RadSystem<ShockProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
	}
}

template <> void QuokkaSimulation<ShockProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const auto radBoundaries_g = RadSystem_Traits<ShockProblem>::radBoundaries;

	// loop over the cell-centered quantities and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];

		amrex::Real x1RadFlux = NAN;
		amrex::Real energy = NAN;
		amrex::Real density = NAN;
		amrex::Real x1Momentum = NAN;
		amrex::Real temp = NAN;

		if (x < shock_position) {
			x1RadFlux = 0.0;
			energy = Egas0 + 0.5 * rho0 * (v0 * v0);
			density = rho0;
			x1Momentum = rho0 * v0;
			temp = T0;
		} else {
			x1RadFlux = 0.0;
			energy = Egas1 + 0.5 * rho1 * (v1 * v1);
			density = rho1;
			x1Momentum = rho1 * v1;
			temp = T1;
		}

		auto Erad_g = RadSystem<ShockProblem>::ComputeThermalRadiationMultiGroup(temp, radBoundaries_g);

		state_cc(i, j, k, RadSystem<ShockProblem>::gasDensity_index) = density;
		state_cc(i, j, k, RadSystem<ShockProblem>::gasInternalEnergy_index) = 0.;
		state_cc(i, j, k, RadSystem<ShockProblem>::x1GasMomentum_index) = x1Momentum;
		state_cc(i, j, k, RadSystem<ShockProblem>::x2GasMomentum_index) = 0;
		state_cc(i, j, k, RadSystem<ShockProblem>::x3GasMomentum_index) = 0;
		state_cc(i, j, k, RadSystem<ShockProblem>::gasEnergy_index) = energy;
		state_cc(i, j, k, RadSystem<ShockProblem>::gasInternalEnergy_index) = energy - (x1Momentum * x1Momentum) / (2 * density);
		for (int g = 0; g < Physics_Traits<ShockProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<ShockProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad_g[g];
			state_cc(i, j, k, RadSystem<ShockProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = x1RadFlux;
			state_cc(i, j, k, RadSystem<ShockProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<ShockProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
		}
	});
}

auto problem_main() -> int
{
	// Problem parameters
	const int max_timesteps = 2e4;
	const double CFL_number = 0.4;
	// const int nx = 512;
	//  const double initial_dtau = 1.0e-3;	  // dimensionless time
	//  const double max_dtau = 1.0e-3;		  // dimensionless time
	//  const double initial_dt = initial_dtau / c_s0;
	//  const double max_dt = max_dtau / c_s0;
	const double max_time = 1.0e-9; // 9.08e-10; // s

	constexpr int nvars = RadSystem<ShockProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);	    // custom x1
		BCs_cc[n].setHi(0, amrex::BCType::ext_dir);	    // custom x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {	    // x2- and x3- directions
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<ShockProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.reconstructionOrder_ = 3;	       // PPM
	sim.cflNumber_ = CFL_number;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = -1;

	// run
	sim.setInitialConditions();
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	int const nx = static_cast<int>(position.size());

	// Plot results
	int status = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		std::vector<double> xs(nx);
		std::vector<double> Trad(nx);
		std::vector<double> Tgas(nx);
		std::vector<double> Erad(nx);
		std::vector<double> Egas(nx);

		for (int i = 0; i < nx; ++i) {
			const double x = Lx * ((i + 0.5) / static_cast<double>(nx));
			xs.at(i) = x; // cm

			double Erad_t = 0.0;
			for (int g = 0; g < Physics_Traits<ShockProblem>::nGroups; ++g) {
				Erad_t += values.at(RadSystem<ShockProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
			}
			Erad.at(i) = Erad_t / a_rad;			     // scaled
			Trad.at(i) = std::pow(Erad_t / a_rad, 1. / 4.) / T0; // dimensionless

			const double Etot_t = values.at(RadSystem<ShockProblem>::gasEnergy_index)[i];
			const double rho = values.at(RadSystem<ShockProblem>::gasDensity_index)[i];
			const double x1GasMom = values.at(RadSystem<ShockProblem>::x1GasMomentum_index)[i];
			const double Ekin = (x1GasMom * x1GasMom) / (2.0 * rho);

			const double Egas_t = (Etot_t - Ekin);
			Egas.at(i) = Egas_t;
			Tgas.at(i) = quokka::EOS<ShockProblem>::ComputeTgasFromEint(rho, Egas_t) / T0; // dimensionless
		}

		// read in exact solution

		std::vector<double> xs_exact;
		std::vector<double> Trad_exact;
		std::vector<double> Tmat_exact;

		std::string const filename = "../extern/LowrieEdwards/shock.txt";
		std::ifstream fstream(filename, std::ios::in);

		const double error_tol = 0.008;
		double rel_error = NAN;
		if (fstream.is_open()) {

			std::string header;
			std::getline(fstream, header);

			for (std::string line; std::getline(fstream, line);) {
				std::istringstream iss(line);
				std::vector<double> values;

				for (double value = NAN; iss >> value;) {
					values.push_back(value);
				}
				auto x_val = values.at(0);    // cm
				auto Tmat_val = values.at(3); // dimensionless
				auto Trad_val = values.at(4); // dimensionless

				if ((x_val > 0.0) && (x_val < Lx)) {
					xs_exact.push_back(x_val);
					Tmat_exact.push_back(Tmat_val);
					Trad_exact.push_back(Trad_val);
				}
			}

			// compute error norm
			std::vector<double> Trad_interp(xs_exact.size());
			amrex::Print() << "xs min/max = " << xs[0] << ", " << xs[xs.size() - 1] << std::endl;
			amrex::Print() << "xs_exact min/max = " << xs_exact[0] << ", " << xs_exact[xs_exact.size() - 1] << std::endl;

			interpolate_arrays(xs_exact.data(), Trad_interp.data(), static_cast<int>(xs_exact.size()), xs.data(), Trad.data(),
					   static_cast<int>(xs.size()));

			double err_norm = 0.;
			double sol_norm = 0.;
			for (size_t i = 0; i < xs_exact.size(); ++i) {
				err_norm += std::abs(Trad_interp[i] - Trad_exact[i]);
				sol_norm += std::abs(Trad_exact[i]);
			}

			rel_error = err_norm / sol_norm;
			amrex::Print() << "Error norm = " << err_norm << std::endl;
			amrex::Print() << "Solution norm = " << sol_norm << std::endl;
			amrex::Print() << "Relative L1 error norm = " << rel_error << std::endl;
		}

		if ((rel_error > error_tol) || std::isnan(rel_error)) {
			status = 1;
		}

#ifdef HAVE_PYTHON
		std::vector<double> xs_scaled(xs.size());
		std::vector<double> xs_exact_scaled(xs_exact.size());
		for (size_t i = 0; i < xs.size(); ++i) {
			xs_scaled.at(i) = xs.at(i) / Lx;
		}
		for (size_t i = 0; i < xs_exact.size(); ++i) {
			xs_exact_scaled.at(i) = xs_exact.at(i) / Lx;
		}

		// plot results
		int const s = 48; // stride
		std::map<std::string, std::string> Trad_args;
		Trad_args["label"] = "radiation";
		Trad_args["color"] = "C1";
		matplotlibcpp::plot(xs_scaled, Trad, Trad_args);

		if (fstream.is_open()) {
			std::unordered_map<std::string, std::string> Trad_exact_args;
			// Trad_exact_args["label"] = "Trad (diffusion ODE)";
			Trad_exact_args["color"] = "C1";
			Trad_exact_args["marker"] = "o";
			// Trad_exact_args["edgecolors"] = "k";
			matplotlibcpp::scatter(strided_vector_from(xs_exact_scaled, s), strided_vector_from(Trad_exact, s), 10.0, Trad_exact_args);
		}

		std::map<std::string, std::string> Tgas_args;
		Tgas_args["label"] = "gas";
		Tgas_args["color"] = "C2";
		matplotlibcpp::plot(xs_scaled, Tgas, Tgas_args);

		if (fstream.is_open()) {
			std::unordered_map<std::string, std::string> Tgas_exact_args;
			// Tgas_exact_args["label"] = "Tmat (diffusion ODE)";
			Tgas_exact_args["color"] = "C2";
			Tgas_exact_args["marker"] = "o";
			// Tgas_exact_args["edgecolors"] = "k";
			matplotlibcpp::scatter(strided_vector_from(xs_exact_scaled, s), strided_vector_from(Tmat_exact, s), 10.0, Tgas_exact_args);
		}

		matplotlibcpp::xlabel("length x (dimensionless)");
		matplotlibcpp::ylabel("temperature (dimensionless)");
		matplotlibcpp::legend();
		matplotlibcpp::tight_layout();
		matplotlibcpp::save("./radshock_multigroup_temperature.pdf");
#endif
	}

	return status;
}
