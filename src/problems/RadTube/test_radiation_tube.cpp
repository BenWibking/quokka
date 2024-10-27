//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_tube.cpp
/// \brief Defines a test problem for radiation pressure terms. This is also a trivial test for the PPL_fixed_slope opacity model.
///

#include <string>
#include <vector>

#include "AMReX.H"
#include "AMReX_BC_TYPES.H"

#include "AMReX_ValLocPair.H"
#include "QuokkaSimulation.hpp"
#include "math/interpolate.hpp"
#include "physics_info.hpp"
#include "radiation/radiation_system.hpp"
#include "test_radiation_tube.hpp"
#include "util/ArrayUtil.hpp"
#include "util/fextract.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

struct TubeProblem {
};

constexpr double kappa0 = 100.;	     // cm^2 g^-1
constexpr double mu = 2.33 * C::m_u; // g
constexpr double gamma_gas = 5. / 3.;

constexpr double rho0 = 1.0;		    // g cm^-3
constexpr double T0 = 2.75e7;		    // K
constexpr double rho1 = 2.1940476649492044; // g cm^-3
constexpr double T1 = 2.2609633884436745e7; // K
constexpr double a_rad = C::a_rad;

constexpr double a0 = 4.0295519855200705e7; // cm s^-1

template <> struct quokka::EOS_Traits<TubeProblem> {
	static constexpr double mean_molecular_weight = mu;
	static constexpr double gamma = gamma_gas;
};

template <> struct Physics_Traits<TubeProblem> {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;		     // number of mass scalars
	static constexpr int numPassiveScalars = numMassScalars + 0; // number of passive scalars
	static constexpr bool is_radiation_enabled = true;
	// face-centred
	static constexpr bool is_mhd_enabled = false;
	// number of radiation groups
	static constexpr int nGroups = 2;
	static constexpr UnitSystem unit_system = UnitSystem::CGS;
};

template <> struct RadSystem_Traits<TubeProblem> {
	static constexpr double c_hat = 10.0 * a0;
	static constexpr double Erad_floor = 0.;
	static constexpr double energy_unit = C::k_B;
	static constexpr amrex::GpuArray<double, Physics_Traits<TubeProblem>::nGroups + 1> radBoundaries{0.01 * T0, 3.3 * T0, 1000. * T0}; // Kelvin
	// static constexpr amrex::GpuArray<double, Physics_Traits<TubeProblem>::nGroups + 1> radBoundaries{0.01 * T0, 1000. * T0}; // Kelvin
	static constexpr int beta_order = 1;
	// static constexpr OpacityModel opacity_model = OpacityModel::single_group;
	static constexpr OpacityModel opacity_model = OpacityModel::piecewise_constant_opacity;
	// static constexpr OpacityModel opacity_model = OpacityModel::PPL_opacity_fixed_slope_spectrum;
};

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto
RadSystem<TubeProblem>::DefineOpacityExponentsAndLowerValues(amrex::GpuArray<double, nGroups_ + 1> /*rad_boundaries*/, const double /*rho*/,
							     const double /*Tgas*/) -> amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2>
{
	amrex::GpuArray<amrex::GpuArray<double, nGroups_ + 1>, 2> exponents_and_values{};
	for (int i = 0; i < nGroups_ + 1; ++i) {
		exponents_and_values[0][i] = 0.0;
		exponents_and_values[1][i] = kappa0;
	}
	return exponents_and_values;
}

template <> struct SimulationData<TubeProblem> {
	amrex::Gpu::DeviceVector<double> x_arr_g;
	amrex::Gpu::DeviceVector<double> rho_arr_g;
	amrex::Gpu::DeviceVector<double> Pgas_arr_g;
	amrex::Gpu::DeviceVector<double> Erad_arr_g;
};

template <> void QuokkaSimulation<TubeProblem>::preCalculateInitialConditions()
{
	// map initial conditions to the global variables
	std::string filename = "../extern/pressure_tube/initial_conditions.txt";
	std::ifstream fstream(filename, std::ios::in);
	AMREX_ALWAYS_ASSERT(fstream.is_open());
	std::string header;
	std::getline(fstream, header);

	std::vector<double> x_arr;
	std::vector<double> rho_arr;
	std::vector<double> Pgas_arr;
	std::vector<double> Erad_arr;

	for (std::string line; std::getline(fstream, line);) {
		std::istringstream iss(line);
		std::vector<double> values;
		for (double value = NAN; iss >> value;) {
			values.push_back(value);
		}
		auto x = values.at(0);	  // position
		auto rho = values.at(1);  // density
		auto Pgas = values.at(2); // gas pressure
		auto Erad = values.at(3); // radiation energy density

		x_arr.push_back(x);
		rho_arr.push_back(rho);
		Pgas_arr.push_back(Pgas);
		Erad_arr.push_back(Erad);
	}

	userData_.x_arr_g.resize(x_arr.size());
	userData_.rho_arr_g.resize(rho_arr.size());
	userData_.Pgas_arr_g.resize(Pgas_arr.size());
	userData_.Erad_arr_g.resize(Erad_arr.size());

	// copy to device
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, x_arr.begin(), x_arr.end(), userData_.x_arr_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, rho_arr.begin(), rho_arr.end(), userData_.rho_arr_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Pgas_arr.begin(), Pgas_arr.end(), userData_.Pgas_arr_g.begin());
	amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Erad_arr.begin(), Erad_arr.end(), userData_.Erad_arr_g.begin());
	amrex::Gpu::streamSynchronizeAll();
}

template <> void QuokkaSimulation<TubeProblem>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract variables required from the geom object
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	auto const &x_ptr = userData_.x_arr_g.dataPtr();
	auto const &rho_ptr = userData_.rho_arr_g.dataPtr();
	auto const &Pgas_ptr = userData_.Pgas_arr_g.dataPtr();
	auto const &Erad_ptr = userData_.Erad_arr_g.dataPtr();
	int x_size = static_cast<int>(userData_.x_arr_g.size());

	const auto radBoundaries_g = RadSystem_Traits<TubeProblem>::radBoundaries;

	// loop over the grid and set the initial condition
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];

		amrex::Real const rho = interpolate_value(x, x_ptr, rho_ptr, x_size);
		amrex::Real const Pgas = interpolate_value(x, x_ptr, Pgas_ptr, x_size);
		amrex::Real const Erad = interpolate_value(x, x_ptr, Erad_ptr, x_size);
		amrex::Real const Tgas = Pgas / C::k_B * mu / rho;
		AMREX_ASSERT(!std::isnan(rho));

		// calculate radEnergyFractions based on the boundary conditions
		auto radEnergyFractions = RadSystem<TubeProblem>::ComputePlanckEnergyFractions(radBoundaries_g, Tgas);

		for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<TubeProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad * radEnergyFractions[g];
			state_cc(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			state_cc(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0;
			AMREX_ASSERT(state_cc(i, j, k, RadSystem<TubeProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) > 0.);
		}

		state_cc(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = Pgas / (gamma_gas - 1.0);
		state_cc(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho;
		state_cc(i, j, k, RadSystem<TubeProblem>::gasInternalEnergy_index) = Pgas / (gamma_gas - 1.0);
		state_cc(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
		state_cc(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;
	});
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<TubeProblem>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
							amrex::GeometryData const &geom, const amrex::Real /*time*/, const amrex::BCRec * /*bcr*/,
							int /*bcomp*/, int /*orig_comp*/)
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

	auto const radBoundaries_g = RadSystem<TubeProblem>::radBoundaries_;

	// calculate radEnergyFractions
	auto radEnergyFractionsT0 = RadSystem<TubeProblem>::ComputePlanckEnergyFractions(radBoundaries_g, T0);
	auto radEnergyFractionsT1 = RadSystem<TubeProblem>::ComputePlanckEnergyFractions(radBoundaries_g, T1);

	if (i < lo[0]) {
		// left side boundary -- constant
		const double Erad = RadSystem<TubeProblem>::radiation_constant_ * std::pow(T0, 4);
		for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
			const double Frad = consVar(lo[0], j, k, RadSystem<TubeProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g);
			consVar(i, j, k, RadSystem<TubeProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad * radEnergyFractionsT0[g];
			consVar(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = Frad;
			consVar(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}

		const double Egas = (C::k_B / mu) * rho0 * T0 / (gamma_gas - 1.0);
		const double x1Mom = consVar(lo[0], j, k, RadSystem<TubeProblem>::x1GasMomentum_index);
		const double Ekin = 0.5 * (x1Mom * x1Mom) / rho0;
		consVar(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = Egas + Ekin;
		consVar(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho0;
		consVar(i, j, k, RadSystem<TubeProblem>::gasInternalEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = x1Mom;
		consVar(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;
	} else if (i > hi[0]) {
		// right-side boundary -- constant
		const double Erad = RadSystem<TubeProblem>::radiation_constant_ * std::pow(T1, 4);
		for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
			const double Frad = consVar(hi[0], j, k, RadSystem<TubeProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g);
			consVar(i, j, k, RadSystem<TubeProblem>::radEnergy_index + Physics_NumVars::numRadVars * g) = Erad * radEnergyFractionsT1[g];
			consVar(i, j, k, RadSystem<TubeProblem>::x1RadFlux_index + Physics_NumVars::numRadVars * g) = Frad;
			consVar(i, j, k, RadSystem<TubeProblem>::x2RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
			consVar(i, j, k, RadSystem<TubeProblem>::x3RadFlux_index + Physics_NumVars::numRadVars * g) = 0.;
		}

		const double Egas = (C::k_B / mu) * rho1 * T1 / (gamma_gas - 1.0);
		const double x1Mom = consVar(hi[0], j, k, RadSystem<TubeProblem>::x1GasMomentum_index);
		const double Ekin = 0.5 * (x1Mom * x1Mom) / rho1;
		consVar(i, j, k, RadSystem<TubeProblem>::gasEnergy_index) = Egas + Ekin;
		consVar(i, j, k, RadSystem<TubeProblem>::gasDensity_index) = rho1;
		consVar(i, j, k, RadSystem<TubeProblem>::gasInternalEnergy_index) = Egas;
		consVar(i, j, k, RadSystem<TubeProblem>::x1GasMomentum_index) = x1Mom;
		consVar(i, j, k, RadSystem<TubeProblem>::x2GasMomentum_index) = 0.;
		consVar(i, j, k, RadSystem<TubeProblem>::x3GasMomentum_index) = 0.;
	}
}

auto problem_main() -> int
{
	// Problem parameters
	// const int nx = 128;
	constexpr double Lx = 128.0;
	constexpr double CFL_number = 0.4;
	constexpr double tmax = Lx / a0;
	constexpr int max_timesteps = 2000;

	// Boundary conditions
	constexpr int nvars = RadSystem<TubeProblem>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir); // Dirichlet x1
		BCs_cc[n].setHi(0, amrex::BCType::ext_dir); // Dirichlet x1
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Problem initialization
	QuokkaSimulation<TubeProblem> sim(BCs_cc);

	sim.radiationReconstructionOrder_ = 3; // PPM
	sim.reconstructionOrder_ = 3;	       // PPM
	sim.stopTime_ = tmax;
	sim.cflNumber_ = CFL_number;
	sim.radiationCflNumber_ = CFL_number;
	sim.maxTimesteps_ = max_timesteps;
	sim.plotfileInterval_ = -1;

	// initialize
	sim.setInitialConditions();
	auto [position0, values0] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);

	// evolve
	sim.evolve();

	// read output variables
	auto [position, values] = fextract(sim.state_new_cc_[0], sim.Geom(0), 0, 0.0);
	const int nx = static_cast<int>(position0.size());

	// compute error norm
	std::vector<std::vector<double>> Erad_arr_at_group(Physics_Traits<TubeProblem>::nGroups, std::vector<double>(nx));
	std::vector<double> Trad_arr(nx);
	std::vector<double> Erad_arr(nx);
	std::vector<double> Erad_exact_arr(nx);
	std::vector<double> Trad_exact_arr(nx);
	std::vector<double> Trad_err(nx);
	std::vector<double> Tgas_arr(nx);
	std::vector<double> Tgas_err(nx);
	std::vector<double> rho_err(nx);
	std::vector<double> xs(nx);

	for (int i = 0; i < nx; ++i) {
		xs[i] = position[i];

		double rho_exact = values0.at(RadSystem<TubeProblem>::gasDensity_index)[i];
		double rho = values.at(RadSystem<TubeProblem>::gasDensity_index)[i];
		rho_err[i] = (rho - rho_exact) / rho_exact;

		double Erad_0 = 0.0;
		double Erad_t = 0.0;
		for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
			Erad_0 += values0.at(RadSystem<TubeProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
			Erad_arr_at_group[g][i] = values.at(RadSystem<TubeProblem>::radEnergy_index + Physics_NumVars::numRadVars * g)[i];
			Erad_t += Erad_arr_at_group[g][i];
		}
		Erad_exact_arr[i] = Erad_0;
		Erad_arr[i] = Erad_t;
		const double Trad_exact = std::pow(Erad_0 / a_rad, 1. / 4.);
		const double Trad = std::pow(Erad_t / a_rad, 1. / 4.);
		Trad_arr[i] = Trad;
		Trad_exact_arr[i] = Trad_exact;
		Trad_err[i] = (Trad - Trad_exact) / Trad_exact;

		double const Egas_exact = values0.at(RadSystem<TubeProblem>::gasEnergy_index)[i];
		double const x1GasMom_exact = values0.at(RadSystem<TubeProblem>::x1GasMomentum_index)[i];
		double const x2GasMom_exact = values0.at(RadSystem<TubeProblem>::x2GasMomentum_index)[i];
		double const x3GasMom_exact = values0.at(RadSystem<TubeProblem>::x3GasMomentum_index)[i];

		double const Egas = values.at(RadSystem<TubeProblem>::gasEnergy_index)[i];
		double const x1GasMom = values.at(RadSystem<TubeProblem>::x1GasMomentum_index)[i];
		double const x2GasMom = values.at(RadSystem<TubeProblem>::x2GasMomentum_index)[i];
		double const x3GasMom = values.at(RadSystem<TubeProblem>::x3GasMomentum_index)[i];

		double const Eint_exact = RadSystem<TubeProblem>::ComputeEintFromEgas(rho_exact, x1GasMom_exact, x2GasMom_exact, x3GasMom_exact, Egas_exact);
		double const Tgas_exact = quokka::EOS<TubeProblem>::ComputeTgasFromEint(rho_exact, Eint_exact);

		double const Eint = RadSystem<TubeProblem>::ComputeEintFromEgas(rho, x1GasMom, x2GasMom, x3GasMom, Egas);
		double const Tgas = quokka::EOS<TubeProblem>::ComputeTgasFromEint(rho, Eint);

		Tgas_arr[i] = Tgas;
		Tgas_err[i] = (Tgas - Tgas_exact) / Tgas_exact;

		// For benchmarking: print x, Tgas_exact, Erad_exact_arr. This is used to calculate E_1_exact and E_2_exact
		// std::cout << xs[i] << ", " << Trad_exact << ", " << Erad_0 << std::endl;
	}

	// define xs_exact, E1_exact, E2_exact
	std::vector<double> xs_exact = {5.00000000000000e-01, 5.50000000000000e+00, 1.05000000000000e+01, 1.55000000000000e+01, 2.05000000000000e+01,
					2.55000000000000e+01, 3.05000000000000e+01, 3.55000000000000e+01, 4.05000000000000e+01, 4.55000000000000e+01,
					5.05000000000000e+01, 5.55000000000000e+01, 6.05000000000000e+01, 6.55000000000000e+01, 7.05000000000000e+01,
					7.55000000000000e+01, 8.05000000000000e+01, 8.55000000000000e+01, 9.05000000000000e+01, 9.55000000000000e+01,
					1.00500000000000e+02, 1.05500000000000e+02, 1.10500000000000e+02, 1.15500000000000e+02, 1.20500000000000e+02,
					1.25500000000000e+02};
	std::vector<double> E1_exact = {1.97806231974620e+15, 1.96003267738932e+15, 1.94139375399209e+15, 1.92211477326756e+15, 1.90216201239978e+15,
					1.88149879792274e+15, 1.86008566953792e+15, 1.83786164032564e+15, 1.81475431543605e+15, 1.79075351115540e+15,
					1.76583321387339e+15, 1.73994821924481e+15, 1.71303490596213e+15, 1.68501210246915e+15, 1.65578211109368e+15,
					1.62523187429012e+15, 1.59323434543658e+15, 1.55965009717319e+15, 1.52432919817267e+15, 1.48711344303233e+15,
					1.44783897852076e+15, 1.40633941779824e+15, 1.36244954047942e+15, 1.31590909579761e+15, 1.26631043035030e+15,
					1.21323876205627e+15};
	std::vector<double> E2_exact = {2.34197994225380e+15, 2.29654950261068e+15, 2.25010503500791e+15, 2.20262123173244e+15, 2.15407068960022e+15,
					2.10442459607726e+15, 2.05365387846208e+15, 2.00168645967436e+15, 1.94843446056395e+15, 1.89396262984460e+15,
					1.83830481312661e+15, 1.78145970875519e+15, 1.72339649303787e+15, 1.66406088653085e+15, 1.60338168190632e+15,
					1.54127777970988e+15, 1.47766576756342e+15, 1.41246806782681e+15, 1.34562168082733e+15, 1.27708749196767e+15,
					1.20686010247924e+15, 1.13497806420176e+15, 1.06153434252058e+15, 9.86527809202386e+14, 9.09819537649705e+14,
					8.31394523943729e+14};

	// interpolate numerical solution onto exact solution tabulated points
	std::vector<double> Erad_arr_numerical_interp_at_group_1(xs_exact.size());
	std::vector<double> Erad_arr_numerical_interp_at_group_2(xs_exact.size());
	interpolate_arrays(xs_exact.data(), Erad_arr_numerical_interp_at_group_1.data(), static_cast<int>(xs_exact.size()), xs.data(),
			   Erad_arr_at_group[0].data(), static_cast<int>(xs.size()));
	interpolate_arrays(xs_exact.data(), Erad_arr_numerical_interp_at_group_2.data(), static_cast<int>(xs_exact.size()), xs.data(),
			   Erad_arr_at_group[1].data(), static_cast<int>(xs.size()));

	double err_norm = 0.;
	double sol_norm = 0.;
	for (int i = 0; i < nx; ++i) {
		err_norm += std::abs(Trad_arr[i] - Trad_exact_arr[i]);
		sol_norm += std::abs(Trad_exact_arr[i]);
	}
	// for (int i = 0; i < static_cast<int>(xs_exact.size()); ++i) {
	// 	err_norm += std::abs(Erad_arr_numerical_interp_at_group_1[i] - E1_exact[i]);
	// 	sol_norm += std::abs(E1_exact[i]);
	// 	err_norm += std::abs(Erad_arr_numerical_interp_at_group_2[i] - E2_exact[i]);
	// 	sol_norm += std::abs(E2_exact[i]);
	// }

	const double rel_err_norm = err_norm / sol_norm;
	const double rel_err_tol = 0.003;
	int status = 1;
	if (rel_err_norm < rel_err_tol) {
		status = 0;
	}
	amrex::Print() << "Relative L1 norm = " << rel_err_norm << std::endl;

#ifdef HAVE_PYTHON
	// Plot results: temperature
	const int s = 4; // stride
	std::map<std::string, std::string> Trad_args;
	std::map<std::string, std::string> Tgas_args;
	std::unordered_map<std::string, std::string> Texact_args;
	Trad_args["label"] = "radiation";
	Trad_args["color"] = "C1";
	Tgas_args["label"] = "gas";
	Tgas_args["color"] = "C2";
	Texact_args["label"] = "exact";
	Texact_args["marker"] = "o";
	Texact_args["color"] = "black";

	matplotlibcpp::plot(xs, Trad_arr, Trad_args);
	matplotlibcpp::plot(xs, Tgas_arr, Tgas_args);
	matplotlibcpp::scatter(strided_vector_from(xs, s), strided_vector_from(Trad_exact_arr, s), 10.0, Texact_args);

	matplotlibcpp::legend();
	// matplotlibcpp::title(fmt::format("t = {:.4g} s", sim.tNew_[0]));
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("temperature (Kelvins)");
	matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radiation_pressure_tube.pdf");

	// Plot results: energy density
	matplotlibcpp::clf();
	matplotlibcpp::xlabel("length x (cm)");
	matplotlibcpp::ylabel("energy density (erg/cm^3)");
	Trad_args["label"] = "E_tot";
	Trad_args["color"] = "k";
	matplotlibcpp::plot(xs, Erad_arr, Trad_args);
	for (int g = 0; g < Physics_Traits<TubeProblem>::nGroups; ++g) {
		Trad_args["label"] = fmt::format("E_{}", g);
		Trad_args["color"] = fmt::format("C{}", g);
		// matplotlibcpp::plot(xs, strided_vector_from(Erad_arr_at_group, s, g), Trad_args);
		matplotlibcpp::plot(xs, Erad_arr_at_group[g], Trad_args);
	}
	std::unordered_map<std::string, std::string> E_tot_args;
	E_tot_args["label"] = "E_tot (exact)";
	E_tot_args["marker"] = "o";
	E_tot_args["color"] = "r";
	matplotlibcpp::scatter(strided_vector_from(xs, s), strided_vector_from(Erad_exact_arr, s), 10.0, E_tot_args);

	std::unordered_map<std::string, std::string> E_0_args;
	E_0_args["label"] = "E_0 (exact)";
	E_0_args["marker"] = "o";
	E_0_args["color"] = "C0";
	matplotlibcpp::scatter(xs_exact, E1_exact, 10.0, E_0_args);

	std::unordered_map<std::string, std::string> E_1_args;
	E_1_args["label"] = "E_1 (exact)";
	E_1_args["marker"] = "o";
	E_1_args["color"] = "C1";
	matplotlibcpp::scatter(xs_exact, E2_exact, 10.0, E_1_args);

	matplotlibcpp::legend();
	// matplotlibcpp::tight_layout();
	matplotlibcpp::save("./radiation_pressure_tube_energy_density.pdf");
	matplotlibcpp::yscale("log");
	matplotlibcpp::save("./radiation_pressure_tube_energy_density_log.pdf");
#endif // HAVE_PYTHON

	// Cleanup and exit
	amrex::Print() << "Finished." << std::endl;
	return status;
}
