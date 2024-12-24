#ifndef RADIATION_SIMULATION_HPP_ // NOLINT
#define RADIATION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file QuokkaSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for radiation moments.

#include <array>
#include <filesystem>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "AMReX.H"
#include "AMReX_AmrParticles.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArray.H"
#include "AMReX_FabFactory.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuControl.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_YAFluxRegister.H"

#ifdef AMREX_USE_ASCENT
#include "AMReX_Conduit_Blueprint.H"
#include <ascent.hpp>
#include <conduit_node.hpp>
#endif

#include "SimulationData.hpp"
#include "chemistry/Chemistry.hpp"
#include "cooling/GrackleLikeCooling.hpp"
#include "cooling/TabulatedCooling.hpp"
#include "eos.H"
#include "hydro/hydro_system.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"
#include "radiation/radiation_system.hpp"
#include "simulation.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class QuokkaSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_cc_;
	using AMRSimulation<problem_t>::state_new_cc_;
	using AMRSimulation<problem_t>::max_signal_speed_;
	using AMRSimulation<problem_t>::state_old_fc_;
	using AMRSimulation<problem_t>::state_new_fc_;
	using AMRSimulation<problem_t>::TracerPC;

	using AMRSimulation<problem_t>::nghost_cc_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::BCs_cc_;
	using AMRSimulation<problem_t>::BCs_fc_;
	using AMRSimulation<problem_t>::componentNames_cc_;
	using AMRSimulation<problem_t>::componentNames_fc_;
	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::fillBoundaryConditions;
	using AMRSimulation<problem_t>::CustomPlotFileName;
	using AMRSimulation<problem_t>::geom;
	using AMRSimulation<problem_t>::grids;
	using AMRSimulation<problem_t>::dmap;
	using AMRSimulation<problem_t>::istep;
	using AMRSimulation<problem_t>::flux_reg_;
	using AMRSimulation<problem_t>::incrementFluxRegisters;
	using AMRSimulation<problem_t>::finest_level;
	using AMRSimulation<problem_t>::finestLevel;
	using AMRSimulation<problem_t>::do_reflux;
	using AMRSimulation<problem_t>::do_tracers;
	using AMRSimulation<problem_t>::Verbose;
	using AMRSimulation<problem_t>::constantDt_;
	using AMRSimulation<problem_t>::boxArray;
	using AMRSimulation<problem_t>::DistributionMap;
	using AMRSimulation<problem_t>::refRatio;
	using AMRSimulation<problem_t>::cellUpdates_;
	using AMRSimulation<problem_t>::CountCells;
	using AMRSimulation<problem_t>::WriteCheckpointFile;
	using AMRSimulation<problem_t>::GetData;
	using AMRSimulation<problem_t>::FillPatchWithData;
	using AMRSimulation<problem_t>::Gconst_;

	using AMRSimulation<problem_t>::densityFloor_;
	using AMRSimulation<problem_t>::tempFloor_;

	SimulationData<problem_t> userData_;

	int enableCooling_ = 0;
	int enableChemistry_ = 0;
	Real max_density_allowed = std::numeric_limits<amrex::Real>::max();
	Real min_density_allowed = std::numeric_limits<amrex::Real>::min();

	quokka::GrackleLikeCooling::grackle_tables grackleTables_;
	quokka::TabulatedCooling::cloudy_tables cloudyTables_;
	std::string coolingTableType_{};
	std::string coolingTableFilename_{};

	static constexpr int nvarTotal_cc_ = Physics_Indices<problem_t>::nvarTotal_cc;
	static constexpr int ncompHydro_ = HydroSystem<problem_t>::nvar_; // hydro
	static constexpr int ncompHyperbolic_ = RadSystem<problem_t>::nvarHyperbolic_;
	static constexpr int nstartHyperbolic_ = RadSystem<problem_t>::nstartHyperbolic_;

	amrex::Real radiationCflNumber_ = 0.3;
	amrex::Real chat_over_c_ = 1.0;
	amrex::Real variable_chat_param1_ = -1.0; // variable_chat_param1_ < 0.0 turns off variable reduced speed of light
	amrex::Real variable_chat_param2_ = 1.0;
	int maxSubsteps_ = 10;				// maximum number of radiation subcycles per hydro step
	amrex::Real dustGasInteractionCoeff_ = 2.5e-34; // erg cm^3 s^−1 K^−3/2

	bool computeReferenceSolution_ = false;
	amrex::Real errorNorm_ = NAN;
	amrex::Real pressureFloor_ = 0.;

	bool use_wavespeed_correction_ = false;
	bool print_rad_counter_ = false;

	int lowLevelDebuggingOutput_ = 0;	// 0 == do nothing; 1 == output intermediate multifabs used in hydro each timestep (ONLY USE FOR DEBUGGING)
	int integratorOrder_ = 2;		// 1 == forward Euler; 2 == RK2-SSP (default)
	int reconstructionOrder_ = 3;		// 1 == donor cell; 2 == PLM; 3 == PPM (default)
	int radiationReconstructionOrder_ = 3;	// 1 == donor cell; 2 == PLM; 3 == PPM (default)
	int useDualEnergy_ = 1;			// 0 == disabled; 1 == use auxiliary internal energy equation (default)
	int abortOnFofcFailure_ = 1;		// 0 == keep going, 1 == abort hydro advance if FOFC fails
	amrex::Real artificialViscosityK_ = 0.; // artificial viscosity coefficient (default == None)

	amrex::Long radiationCellUpdates_ = 0; // total number of radiation cell-updates

	// member functions
	explicit QuokkaSimulation(amrex::Vector<amrex::BCRec> &BCs_cc, amrex::Vector<amrex::BCRec> &BCs_fc) : AMRSimulation<problem_t>(BCs_cc, BCs_fc)
	{
		initialize();
	}

	explicit QuokkaSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : AMRSimulation<problem_t>(BCs_cc) { initialize(); }

	inline void initialize()
	{
		defineComponentNames();
		// read in runtime parameters
		readParmParse();
		// set gamma
		amrex::ParmParse eos("eos");
		eos.add("eos_gamma", quokka::EOS_Traits<problem_t>::gamma);
		// initialize Microphysics params
		init_extern_parameters();
		// initialize Microphysics EOS
		amrex::Real small_temp = 1e-10;
		amrex::Real small_dens = 1e-100;
		eos_init(small_temp, small_dens);
	}

	[[nodiscard]] static auto getScalarVariableNames() -> std::vector<std::string>;
	void defineComponentNames();
	void readParmParse();

	void checkHydroStates(amrex::MultiFab &mf, char const *file, int line);
	void computeMaxSignalLocal(int level) override;
	auto computeExtraPhysicsTimestep(int lev) -> amrex::Real override;
	void preCalculateInitialConditions() override;
	void setInitialConditionsOnGrid(quokka::grid const &grid_elem) override;
	void setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) override;
	void createInitialParticles() override;
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle) override;
	void computeBeforeTimestep() override;
	void computeAfterTimestep() override;
	void computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int /*ncycle*/);
	void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;
	void computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo);

	// compute derived variables
	void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;

	// compute projected vars
	[[nodiscard]] auto ComputeProjections(const amrex::Direction dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> override;

	// compute statistics
	auto ComputeStatistics() -> std::map<std::string, amrex::Real> override;

	// fix-up states
	void FixupState(int level) override;

	// implement FillPatch function
	void FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir,
		       FillPatchType fptype) override;

	// functions to operate on state vector before/after interpolating between levels
	static void PreInterpState(amrex::MultiFab &mf, int scomp, int ncomp);
	static void PostInterpState(amrex::MultiFab &mf, int scomp, int ncomp);

	// compute axis-aligned 1D profile of user_f(x, y, z)
	template <typename F> auto computeAxisAlignedProfile(int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real>;

	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	// fill rhs for Poisson solve
	void fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) override;

	// add gravitational acceleration to hydro state
	void applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) override;

	void addFluxArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> &dstfluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &srcfluxes, int srccomp,
			   int dstcomp);

	auto expandFluxArrays(std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, int nstartNew, int ncompNew)
	    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>;

	void printCoordinates(int lev, const amrex::IntVect &cell_idx);

	void advanceHydroAtLevelWithRetries(int lev, amrex::Real time, amrex::Real dt_lev, amrex::YAFluxRegister *fr_as_crse,
					    amrex::YAFluxRegister *fr_as_fine);

	auto advanceHydroAtLevel(amrex::MultiFab &state_old_cc_tmp, amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine, int lev,
				 amrex::Real time, amrex::Real dt_lev) -> bool;

	void addStrangSplitSources(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt_lev);
	auto addStrangSplitSourcesWithBuiltin(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt_lev) -> bool;

	auto isCflViolated(int lev, amrex::Real time, amrex::Real dt_actual) -> bool;

	// radiation subcycle
	void swapRadiationState(amrex::MultiFab &stateOld, amrex::MultiFab const &stateNew);
	auto computeNumberOfRadiationSubsteps(int lev, amrex::Real dt_lev_hydro) -> int;
	void advanceRadiationForwardEuler(int lev, amrex::Real time, amrex::Real dt_radiation, int iter_count, int nsubsteps, amrex::YAFluxRegister *fr_as_crse,
					  amrex::YAFluxRegister *fr_as_fine, amrex::MultiFab const &reducedSpeedOfLightFactor);
	void advanceRadiationMidpointRK2(int lev, amrex::Real time, amrex::Real dt_radiation, int iter_count, int nsubsteps, amrex::YAFluxRegister *fr_as_crse,
					 amrex::YAFluxRegister *fr_as_fine, amrex::MultiFab const &reducedSpeedOfLightFactor);

	void subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro, amrex::YAFluxRegister *fr_as_crse,
				      amrex::YAFluxRegister *fr_as_fine);

	void operatorSplitSourceTerms(amrex::Array4<amrex::Real> const &stateNew, const amrex::Box &indexRange, amrex::Real time, double dt, int stage,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
				      amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor, int *p_iteration_counter,
				      int *p_iteration_failure_counter);

	auto computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, int nvars,
				    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor_)
	    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>, std::array<amrex::FArrayBox, AMREX_SPACEDIM>>;

	auto computeHydroFluxes(amrex::MultiFab const &consVar, int nvars, int lev)
	    -> std::pair<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>;

	auto computeFOHydroFluxes(amrex::MultiFab const &consVar, int nvars, int lev)
	    -> std::pair<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>;

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive,
			  const amrex::Box &indexRange, int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
			  amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor);

	template <FluxDir DIR>
	void hydroFluxFunction(amrex::MultiFab const &primVar, amrex::MultiFab &leftState, amrex::MultiFab &rightState, amrex::MultiFab &x1Flux,
			       amrex::MultiFab &x1FaceVel, amrex::MultiFab const &x1Flat, amrex::MultiFab const &x2Flat, amrex::MultiFab const &x3Flat,
			       int ng_reconstruct, int nvars);

	template <FluxDir DIR>
	void hydroFOFluxFunction(amrex::MultiFab const &primVar, amrex::MultiFab &leftState, amrex::MultiFab &rightState, amrex::MultiFab &x1Flux,
				 amrex::MultiFab &x1FaceVel, int ng_reconstruct, int nvars);

	void replaceFluxes(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes,
			   amrex::iMultiFab &redoFlag);
};

template <typename problem_t> void QuokkaSimulation<problem_t>::defineComponentNames()
{

	// cell-centred
	// add hydro state variables
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		std::vector<std::string> hydroNames = {"gasDensity", "x-GasMomentum", "y-GasMomentum", "z-GasMomentum", "gasEnergy", "gasInternalEnergy"};
		componentNames_cc_.insert(componentNames_cc_.end(), hydroNames.begin(), hydroNames.end());
	}
	// add passive scalar variables
	if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
		std::vector<std::string> scalarNames = getScalarVariableNames();
		componentNames_cc_.insert(componentNames_cc_.end(), scalarNames.begin(), scalarNames.end());
	}
	// add radiation state variables
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		std::vector<std::string> radNames = {};
		for (int i = 0; i < Physics_Traits<problem_t>::nGroups; ++i) {
			radNames.push_back("radEnergy-Group" + std::to_string(i));
			radNames.push_back("x-RadFlux-Group" + std::to_string(i));
			radNames.push_back("y-RadFlux-Group" + std::to_string(i));
			radNames.push_back("z-RadFlux-Group" + std::to_string(i));
		}
		componentNames_cc_.insert(componentNames_cc_.end(), radNames.begin(), radNames.end());
	}

	// face-centred

	// add face-centered velocities
	for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
		componentNames_fc_.push_back({quokka::face_dir_str[idim] + "-RiemannSolverVelocity"});
	}
	// add mhd state variables
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
			componentNames_fc_.push_back({quokka::face_dir_str[idim] + "-BField"});
		}
	}
}

// initialize metadata
template <typename problem_t> void AMRSimulation<problem_t>::initializeSimulationMetadata()
{
	if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CONSTANTS) {
		// if unit system is CONSTANTS, the units are not well defined unless all four constants, G, k_B, c, and a_rad, are defined. However, in a hydro
		// simulation, only k_B is defined. In a radiation-hydrodynamics simulation, only k_B, c, and a_rad are defined. Besides, CONSTANTS is only used
		// for testing purposes, so we don't care about the units in that case.
		simulationMetadata_["unit_length"] = NAN;
		simulationMetadata_["unit_mass"] = NAN;
		simulationMetadata_["unit_time"] = NAN;
		simulationMetadata_["unit_temperature"] = NAN;

		// constants
		simulationMetadata_["k_B"] = Physics_Traits<problem_t>::boltzmann_constant;
		simulationMetadata_["G"] = Physics_Traits<problem_t>::gravitational_constant;
		if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
			simulationMetadata_["c"] = Physics_Traits<problem_t>::c_light;
			// Commented out because chat_over_c_ is not accessible here.
			// simulationMetadata_["c_hat"] = Physics_Traits<problem_t>::c_light * RadSystem_Traits<problem_t>::c_hat_over_c;
			simulationMetadata_["a_rad"] = Physics_Traits<problem_t>::radiation_constant;
		}
	} else {
		// units
		simulationMetadata_["unit_length"] = unit_length;
		simulationMetadata_["unit_mass"] = unit_mass;
		simulationMetadata_["unit_time"] = unit_time;
		simulationMetadata_["unit_temperature"] = unit_temperature;

		// constants
		double k_B = NAN;
		if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CGS) {
			k_B = C::k_B;
		} else if constexpr (Physics_Traits<problem_t>::unit_system == UnitSystem::CUSTOM) {
			// Have to do a conversion because EOS class is not accessible here
			k_B = C::k_B /
			      (Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_length * Physics_Traits<problem_t>::unit_mass /
			       (Physics_Traits<problem_t>::unit_time * Physics_Traits<problem_t>::unit_time) / Physics_Traits<problem_t>::unit_temperature);
		}
		simulationMetadata_["k_B"] = k_B;
		simulationMetadata_["G"] = Gconst_;
		if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
			simulationMetadata_["c"] = RadSystem<problem_t>::c_light_;
			// Commented out because chat_over_c_ is not accessible here.
			// simulationMetadata_["c_hat"] = RadSystem<problem_t>::c_light_ * chat_over_c_;
			simulationMetadata_["a_rad"] = RadSystem<problem_t>::radiation_constant_;
		}
	}
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::getScalarVariableNames() -> std::vector<std::string>
{
	// return vector of names for the passive scalars
	// this can be specialized by the user to provide more descriptive names
	// (these names are used to label the variables in the plotfiles)

	std::vector<std::string> names;
	int nscalars = HydroSystem<problem_t>::nscalars_;
	names.reserve(nscalars);
	for (int n = 0; n < nscalars; ++n) {
		// write string 'scalar_1', etc.
		names.push_back(fmt::format("scalar_{}", n));
	}
	return names;
}

template <typename problem_t> void QuokkaSimulation<problem_t>::readParmParse()
{
	// set hydro runtime parameters
	{
		amrex::ParmParse hpp("hydro");
		hpp.query("low_level_debugging_output", lowLevelDebuggingOutput_);
		hpp.query("rk_integrator_order", integratorOrder_);
		hpp.query("reconstruction_order", reconstructionOrder_);
		hpp.query("use_dual_energy", useDualEnergy_);
		hpp.query("abort_on_fofc_failure", abortOnFofcFailure_);
		hpp.query("artificial_viscosity_coefficient", artificialViscosityK_);
	}

	// set cooling runtime parameters
	{
		amrex::ParmParse hpp("cooling");
		int alwaysReadTables = 0;
		hpp.query("enabled", enableCooling_);
		hpp.query("cooling_table_type", coolingTableType_);
		hpp.query("read_tables_even_if_disabled", alwaysReadTables);
		hpp.query("hdf5_data_file", coolingTableFilename_);

		if ((enableCooling_ == 1) || (alwaysReadTables == 1)) {
			if (coolingTableType_ == "grackle") {
				// read Grackle tables
				amrex::Print() << "Reading Grackle tables...\n";
				quokka::GrackleLikeCooling::readGrackleData(coolingTableFilename_, grackleTables_);
			} else if (coolingTableType_ == "cloudy_cooling_tools") {
				// read cloudy_cooling_tools tables
				amrex::Print() << "Reading cloudy-cooling-tools tables...\n";
				quokka::TabulatedCooling::readCloudyData(coolingTableFilename_, cloudyTables_);
			} else {
				amrex::Abort("Invalid cooling table type!");
			}
		}
	}

#ifdef CHEMISTRY
	// set chemistry runtime parameters
	{
		amrex::ParmParse hpp("chemistry");
		hpp.query("enabled", enableChemistry_);
		hpp.query("max_density_allowed", max_density_allowed); // chemistry is not accurate for densities > 3e-6
		hpp.query("min_density_allowed", min_density_allowed); // don't do chemistry in cells with densities below the minimum density specified
	}
#endif

	// set radiation runtime parameters
	{
		amrex::ParmParse rpp("radiation");
		rpp.query("reconstruction_order", radiationReconstructionOrder_);
		rpp.query("cfl", radiationCflNumber_);
		rpp.query("c_hat_over_c", chat_over_c_);
		rpp.query("variable_chat_param1", variable_chat_param1_);
		rpp.query("variable_chat_param2", variable_chat_param2_);
		rpp.query("dust_gas_interaction_coeff", dustGasInteractionCoeff_);
		rpp.query("print_iteration_counts", print_rad_counter_);
	}
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::computeNumberOfRadiationSubsteps(int lev, amrex::Real dt_lev_hydro) -> int
{
	// compute radiation timestep
	auto const &dx = geom[lev].CellSizeArray();
	amrex::Real c_hat = RadSystem<problem_t>::c_light_ * chat_over_c_;
	amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	amrex::Real dtrad_tmp = radiationCflNumber_ * (dx_min / c_hat);
	int nsubSteps = std::ceil(dt_lev_hydro / dtrad_tmp);
	return nsubSteps;
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	BL_PROFILE("QuokkaSimulation::computeMaxSignalLocal()");

	// hydro: loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_cc_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateNew = state_new_cc_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);

		if constexpr (Physics_Traits<problem_t>::is_hydro_enabled && !(Physics_Traits<problem_t>::is_radiation_enabled)) {
			// hydro only
			HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateNew, maxSignal, indexRange);
		} else if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
			// radiation hydro, or radiation only
			RadSystem<problem_t>::ComputeMaxSignalSpeed(chat_over_c_, stateNew, maxSignal, indexRange);
			if constexpr (Physics_Traits<problem_t>::is_hydro_enabled) {
				auto maxSignalHydroFAB = amrex::FArrayBox(indexRange);
				auto const &maxSignalHydro = maxSignalHydroFAB.array();
				HydroSystem<problem_t>::ComputeMaxSignalSpeed(stateNew, maxSignalHydro, indexRange);
				const int maxSubsteps = maxSubsteps_;
				// ensure that we use the smaller of the two timesteps
				amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					amrex::Real const maxSignalRadiation = maxSignal(i, j, k) / static_cast<double>(maxSubsteps);
					maxSignal(i, j, k) = std::max(maxSignalRadiation, maxSignalHydro(i, j, k));
				});
			}
		} else {
			// no physics modules enabled, why are we running?
			amrex::Abort("At least one of hydro or radiation must be enabled! Cannot "
				     "compute a time step.");
		}
	}
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::computeExtraPhysicsTimestep(int const /*level*/) -> amrex::Real
{
	BL_PROFILE("QuokkaSimulation::computeExtraPhysicsTimestep()");
	// users can override this to enforce additional timestep constraints
	return std::numeric_limits<amrex::Real>::max();
}

#if !defined(NDEBUG)
#define CHECK_HYDRO_STATES(mf) checkHydroStates(mf, __FILE__, __LINE__)
#else
#define CHECK_HYDRO_STATES(mf)
#endif

template <typename problem_t> void QuokkaSimulation<problem_t>::checkHydroStates(amrex::MultiFab &mf, char const *file, int line)
{
	BL_PROFILE("QuokkaSimulation::checkHydroStates()");

	bool validStates = HydroSystem<problem_t>::CheckStatesValid(mf);
	amrex::ParallelDescriptor::ReduceBoolAnd(validStates);

	if (!validStates) {
		amrex::Print() << "Hydro states invalid (" + std::string(file) + ":" + std::to_string(line) + ")\n";
		amrex::Print() << "Writing checkpoint for debugging...\n";
		amrex::MFIter::allowMultipleMFIters(1);
		WriteCheckpointFile();
		amrex::Abort("Hydro states invalid (" + std::string(file) + ":" + std::to_string(line) + ")");
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::preCalculateInitialConditions()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void QuokkaSimulation<problem_t>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void QuokkaSimulation<problem_t>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if face-centered vars are used
}

template <typename problem_t> void QuokkaSimulation<problem_t>::createInitialParticles()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if particles are used
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeBeforeTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement if desired
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterLevelAdvance(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	// user should implement if desired
}

template <typename problem_t> void QuokkaSimulation<problem_t>::addStrangSplitSources(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt)
{
	// user should implement
	// (when Strang splitting is enabled, dt is actually 0.5*dt_lev)
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::addStrangSplitSourcesWithBuiltin(amrex::MultiFab &state, int lev, amrex::Real time, amrex::Real dt) -> bool
{
	// start by assuming cooling integrator is successful.
	bool cool_success = true;
	if (enableCooling_ == 1) {
		// compute cooling
		if (coolingTableType_ == "grackle") {
			cool_success = quokka::GrackleLikeCooling::computeCooling<problem_t>(state, dt, grackleTables_, tempFloor_);
		} else if (coolingTableType_ == "cloudy_cooling_tools") {
			cool_success = quokka::TabulatedCooling::computeCooling<problem_t>(state, dt, cloudyTables_, tempFloor_);
		} else {
			amrex::Abort("Invalid cooling table type!");
		}
	}

	// start by assuming chemistry burn is successful.
	bool burn_success = true; // NOLINT
#ifdef CHEMISTRY
	if (enableChemistry_ == 1) {
		// compute chemistry
		burn_success = quokka::chemistry::computeChemistry<problem_t>(state, dt, max_density_allowed, min_density_allowed);
	}
#endif

	// compute user-specified sources
	addStrangSplitSources(state, lev, time, dt);

	return (burn_success && cool_success);
}

template <typename problem_t> void QuokkaSimulation<problem_t>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, const int ncomp) const
{
	// compute derived variables and save in 'mf' -- user should implement
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::ComputeProjections(const amrex::Direction /*dir*/) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	// compute projections and return as unordered_map -- user should implement
	return std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>{};
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	// compute statistics and return a std::map<std::string, amrex::Real> -- user should implement
	// IMPORTANT: the user is responsible for performing any necessary MPI reductions before returning
	return std::map<std::string, amrex::Real>{};
}

template <typename problem_t> void QuokkaSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement -- user should implement
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo)
{
	// user should implement
}

template <typename problem_t> void QuokkaSimulation<problem_t>::computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons)
{
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx0 = geom[0].CellSizeArray();
	amrex::Real const vol = AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2]);

	// check conservation of total energy
	amrex::Real const Egas0 = initSumCons[RadSystem<problem_t>::gasEnergy_index];
	amrex::Real const Egas = state_new_cc_[0].sum(RadSystem<problem_t>::gasEnergy_index) * vol;

	amrex::Real Etot0 = NAN;
	amrex::Real Etot = NAN;
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		amrex::Real Erad0 = 0.;
		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
			// We don't multiply by c_over_chat here because we're using variable c_hat and E_gas + E_rad * (chat / c) is no longer conserved
			Erad0 += initSumCons[RadSystem<problem_t>::radEnergy_index + Physics_NumVars::numRadVars * g];
		}
		Etot0 = Egas0 + Erad0;
		amrex::Real Erad = 0.;
		for (int g = 0; g < Physics_Traits<problem_t>::nGroups; ++g) {
			// We don't multiply by c_over_chat here because we're using variable c_hat and E_gas + E_rad * (chat / c) is no longer conserved
			Erad += state_new_cc_[0].sum(RadSystem<problem_t>::radEnergy_index + Physics_NumVars::numRadVars * g) * vol;
		}
		Etot = Egas + Erad;
	} else {
		Etot0 = Egas0;
		Etot = Egas;
	}

	amrex::Real const abs_err = (Etot - Etot0);
	amrex::Real rel_err = NAN;
	if (Etot0 != 0) {
		rel_err = abs_err / Etot0;
	}

	amrex::Print() << "\nInitial gas+radiation energy = " << Etot0 << '\n';
	amrex::Print() << "Final gas+radiation energy = " << Etot << '\n';
	amrex::Print() << "\tabsolute conservation error = " << abs_err << '\n';
	amrex::Print() << "\trelative conservation error = " << rel_err << '\n';
	amrex::Print() << '\n';

	if (computeReferenceSolution_) {
		// compute reference solution
		const int ncomp = state_new_cc_[0].nComp();
		amrex::MultiFab state_ref_level0(boxArray(0), DistributionMap(0), ncomp, 0);
		computeReferenceSolution(state_ref_level0, geom[0].CellSizeArray(), geom[0].ProbLoArray());

		// compute error norm
		amrex::MultiFab residual(boxArray(0), DistributionMap(0), ncomp, 0);
		amrex::MultiFab::Copy(residual, state_ref_level0, 0, 0, ncomp, 0);
		amrex::MultiFab::Saxpy(residual, -1., state_new_cc_[0], 0, 0, ncomp, 0);

		amrex::Real sol_norm = 0.;
		amrex::Real err_norm = 0.;
		// compute rms of each component
		for (int n = 0; n < ncomp; ++n) {
			sol_norm += std::pow(state_ref_level0.norm1(n), 2);
			err_norm += std::pow(residual.norm1(n), 2);
		}
		sol_norm = std::sqrt(sol_norm);
		err_norm = std::sqrt(err_norm);

		const double rel_error = err_norm / sol_norm;
		errorNorm_ = rel_error;
		amrex::Print() << "Relative rms L1 error norm = " << rel_error << '\n';
	}
	amrex::Print() << '\n';

	// compute average number of radiation subcycles per timestep
	if (cellUpdates_ > 0) {
		double const avg_rad_subcycles = static_cast<double>(radiationCellUpdates_) / static_cast<double>(cellUpdates_);
		amrex::Print() << "avg. num. of radiation subcycles = " << avg_rad_subcycles << '\n';
		amrex::Print() << '\n';
	} else {
		amrex::Print() << "No cell updates performed!\n";
		amrex::Print() << '\n';
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int ncycle)
{
	BL_PROFILE("QuokkaSimulation::advanceSingleTimestepAtLevel()");

	// get flux registers
	amrex::YAFluxRegister *fr_as_crse = nullptr;
	amrex::YAFluxRegister *fr_as_fine = nullptr;
	if (do_reflux != 0) {
		if (lev < finestLevel()) {
			fr_as_crse = flux_reg_[lev + 1].get();
			fr_as_crse->reset();
		}
		if (lev > 0) {
			fr_as_fine = flux_reg_[lev].get();
		}
	}

	// since we are starting a new timestep, need to swap old and new state vectors
	std::swap(state_old_cc_[lev], state_new_cc_[lev]);
	if (Physics_Indices<problem_t>::nvarTotal_fc > 0) {
		std::swap(state_old_fc_[lev], state_new_fc_[lev]);
	}

	// check hydro states before update (this can be caused by the flux register!)
	CHECK_HYDRO_STATES(state_old_cc_[lev]);

	// advance hydro
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled) {
		advanceHydroAtLevelWithRetries(lev, time, dt_lev, fr_as_crse, fr_as_fine);
	} else {
		// copy hydro vars from state_old_cc_ to state_new_cc_
		// (otherwise radiation update will be wrong!)
		amrex::MultiFab::Copy(state_new_cc_[lev], state_old_cc_[lev], 0, 0, ncompHydro_, 0);
	}

	// check hydro states after hydro update
	CHECK_HYDRO_STATES(state_new_cc_[lev]);

	// subcycle radiation
	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
		subcycleRadiationAtLevel(lev, time, dt_lev, fr_as_crse, fr_as_fine);
	}

	// check hydro states after radiation update
	CHECK_HYDRO_STATES(state_new_cc_[lev]);

	// compute any operator-split terms here (user-defined)
	computeAfterLevelAdvance(lev, time, dt_lev, ncycle);

	// check hydro states after user work
	CHECK_HYDRO_STATES(state_new_cc_[lev]);

	// check state validity
	AMREX_ASSERT(!state_new_cc_[lev].contains_nan(0, state_new_cc_[lev].nComp()));
}

template <typename problem_t> void QuokkaSimulation<problem_t>::fillPoissonRhsAtLevel(amrex::MultiFab &rhs_mf, const int lev)
{
	// add hydro density to Poisson rhs
	auto const &state = state_new_cc_[lev].const_arrays();
	auto rhs = rhs_mf.arrays();
	const Real G = Gconst_;

	amrex::ParallelFor(rhs_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// *add* density to rhs_mf
		// (N.B. particles **will not work** if you overwrite the density here!)
		rhs[bx](i, j, k) += 4.0 * M_PI * G * state[bx](i, j, k, HydroSystem<problem_t>::density_index);
	});
	amrex::Gpu::streamSynchronizeAll();
}

template <typename problem_t> void QuokkaSimulation<problem_t>::applyPoissonGravityAtLevel(amrex::MultiFab const &phi_mf, const int lev, const amrex::Real dt)
{
	if constexpr (AMREX_SPACEDIM == 3) {
		// apply Poisson gravity operator on level 'lev'
		auto const &dx = geom[lev].CellSizeArray();
		auto const &phi = phi_mf.const_arrays();
		auto state = state_new_cc_[lev].arrays();

		amrex::ParallelFor(phi_mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			// add operator-split gravitational acceleration
			const amrex::Real rho = state[bx](i, j, k, HydroSystem<problem_t>::density_index);
			amrex::Real px = state[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			amrex::Real py = state[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			amrex::Real pz = state[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const amrex::Real KE_old = 0.5 * (px * px + py * py + pz * pz) / rho;

			// g = -grad \phi
			amrex::Real gx = -0.5 * (phi[bx](i + 1, j, k) - phi[bx](i - 1, j, k)) / dx[0];
			amrex::Real gy = -0.5 * (phi[bx](i, j + 1, k) - phi[bx](i, j - 1, k)) / dx[1];
			amrex::Real gz = -0.5 * (phi[bx](i, j, k + 1) - phi[bx](i, j, k - 1)) / dx[2];

			px += dt * rho * gx;
			py += dt * rho * gy;
			pz += dt * rho * gz;
			const amrex::Real KE_new = 0.5 * (px * px + py * py + pz * pz) / rho;
			const amrex::Real dKE = KE_new - KE_old;

			state[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index) = px;
			state[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index) = py;
			state[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index) = pz;
			state[bx](i, j, k, HydroSystem<problem_t>::energy_index) += dKE;
		});
	}
}

// fix-up any unphysical states created by AMR operations
// (e.g., caused by the flux register or from interpolation)
template <typename problem_t> void QuokkaSimulation<problem_t>::FixupState(int lev)
{
	BL_PROFILE("QuokkaSimulation::FixupState()");

	// fix hydro state
	HydroSystem<problem_t>::EnforceLimits(densityFloor_, tempFloor_, state_new_cc_[lev]);

	// sync internal energy and total energy
	HydroSystem<problem_t>::SyncDualEnergy(state_new_cc_[lev]);
}

// Compute a new multifab 'mf' by copying in state from valid region and filling
// ghost cells
// NOTE: This has to be implemented here because PreInterpState and PostInterpState
// are implemented in this class (and must be *static* functions).
template <typename problem_t>
void QuokkaSimulation<problem_t>::FillPatch(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp, int ncomp, quokka::centering cen, quokka::direction dir,
					    FillPatchType fptype)
{
	BL_PROFILE("AMRSimulation::FillPatch()");

	amrex::Vector<amrex::MultiFab *> cmf;
	amrex::Vector<amrex::MultiFab *> fmf;
	amrex::Vector<amrex::Real> ctime;
	amrex::Vector<amrex::Real> ftime;

	if (lev == 0) {
		// in this case, should return either state_new_[lev] or state_old_[lev]
		GetData(lev, time, fmf, ftime, cen, dir);
	} else {
		// in this case, should return either state_new_[lev] or state_old_[lev]
		// returns old state, new state, or both depending on 'time'
		GetData(lev, time, fmf, ftime, cen, dir);
		GetData(lev - 1, time, cmf, ctime, cen, dir);
	}

	if (cen == quokka::centering::cc) {
		FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, BCs_cc_, cen, fptype, PreInterpState, PostInterpState);
	} else if (cen == quokka::centering::fc) {
		FillPatchWithData(lev, time, mf, cmf, ctime, fmf, ftime, icomp, ncomp, BCs_fc_, cen, fptype, PreInterpState, PostInterpState);
	}
}

template <typename problem_t> void QuokkaSimulation<problem_t>::PreInterpState(amrex::MultiFab &mf, int /*scomp*/, int /*ncomp*/)
{
	BL_PROFILE("QuokkaSimulation::PreInterpState()");

	auto const &cons = mf.arrays();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const auto rho = cons[bx](i, j, k, HydroSystem<problem_t>::density_index);
		const auto px = cons[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		const auto py = cons[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		const auto pz = cons[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
		const auto Etot = cons[bx](i, j, k, HydroSystem<problem_t>::energy_index);
		const auto kinetic_energy = (px * px + py * py + pz * pz) / (2.0 * rho);

		// replace hydro total energy with specific internal energy (SIE)
		const auto e = (Etot - kinetic_energy) / rho;
		cons[bx](i, j, k, HydroSystem<problem_t>::energy_index) = e;
	});
}

template <typename problem_t> void QuokkaSimulation<problem_t>::PostInterpState(amrex::MultiFab &mf, int /*scomp*/, int /*ncomp*/)
{
	BL_PROFILE("QuokkaSimulation::PostInterpState()");

	auto const &cons = mf.arrays();
	amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) {
		const auto rho = cons[bx](i, j, k, HydroSystem<problem_t>::density_index);
		const auto px = cons[bx](i, j, k, HydroSystem<problem_t>::x1Momentum_index);
		const auto py = cons[bx](i, j, k, HydroSystem<problem_t>::x2Momentum_index);
		const auto pz = cons[bx](i, j, k, HydroSystem<problem_t>::x3Momentum_index);
		const auto e = cons[bx](i, j, k, HydroSystem<problem_t>::energy_index);
		const auto Eint = rho * e;
		const auto kinetic_energy = (px * px + py * py + pz * pz) / (2.0 * rho);

		// recompute hydro total energy from Eint + KE
		const auto Etot = Eint + kinetic_energy;
		cons[bx](i, j, k, HydroSystem<problem_t>::energy_index) = Etot;
	});
}

template <typename problem_t>
template <typename F>
auto QuokkaSimulation<problem_t>::computeAxisAlignedProfile(const int axis, F const &user_f) -> amrex::Gpu::HostVector<amrex::Real>
{
	// compute a 1D profile of user_f(i, j, k, state) along the given axis.
	BL_PROFILE("QuokkaSimulation::computeAxisAlignedProfile()");

	// allocate temporary multifabs
	amrex::Vector<amrex::MultiFab> q;
	q.resize(finest_level + 1);
	for (int lev = 0; lev <= finest_level; ++lev) {
		q[lev].define(boxArray(lev), DistributionMap(lev), 1, 0);
	}

	// evaluate user_f on all levels
	for (int lev = 0; lev <= finest_level; ++lev) {
		for (amrex::MFIter iter(q[lev]); iter.isValid(); ++iter) {
			auto const &box = iter.validbox();
			auto const &state = state_new_cc_[lev].const_array(iter);
			auto const &result = q[lev].array(iter);
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) { result(i, j, k) = user_f(i, j, k, state); });
		}
	}

	// average down
	for (int crse_lev = finest_level - 1; crse_lev >= 0; --crse_lev) {
		amrex::average_down(q[crse_lev + 1], q[crse_lev], geom[crse_lev + 1], geom[crse_lev], 0, q[crse_lev].nComp(), refRatio(crse_lev));
	}

	// compute 1D profile from level 0 multifab
	amrex::Box domain = geom[0].Domain();
	auto profile = amrex::sumToLine(q[0], 0, q[0].nComp(), domain, axis);

	// normalize profile
	amrex::Long numCells = domain.numPts() / domain.length(axis);
	for (double &bin : profile) {
		bin /= static_cast<amrex::Real>(numCells);
	}

	return profile;
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::advanceHydroAtLevelWithRetries(int lev, amrex::Real time, amrex::Real dt_lev, amrex::YAFluxRegister *fr_as_crse,
								 amrex::YAFluxRegister *fr_as_fine)
{
	BL_PROFILE_REGION("HydroSolver");
	// timestep retries
	const int max_retries = 6;
	bool success = false;

	// save the pre-advance fine flux register state in originalFineData
	amrex::MultiFab originalFineData;
	if (fr_as_fine != nullptr) {
		amrex::MultiFab const &fineData = fr_as_fine->getFineData();
		originalFineData.define(fineData.boxArray(), fineData.DistributionMap(), fineData.nComp(), 0);
		amrex::Copy(originalFineData, fineData, 0, 0, fineData.nComp(), 0);
	}

#ifdef AMREX_PARTICLES
	amrex::AmrTracerParticleContainer::ContainerLike<amrex::DefaultAllocator> originalTracerPC;
	if (do_tracers != 0) {
		// save the pre-advance tracer particles
		originalTracerPC = TracerPC->make_alike();	 // create empty particle container
		originalTracerPC.copyParticles(*TracerPC, true); // do local copy of particles
	}
#endif

	for (int retry_count = 0; retry_count <= max_retries; ++retry_count) {
		// reduce timestep by a factor of 2^retry_count
		const int nsubsteps = static_cast<int>(std::pow(2, retry_count));
		const amrex::Real dt_step = dt_lev / nsubsteps;

		if (retry_count > 0 && Verbose()) {
			amrex::Print() << "\t>> Re-trying hydro advance at level " << lev << " with reduced timestep (nsubsteps = " << nsubsteps
				       << ", dt_new = " << dt_step << ")\n";
		}

		if (retry_count > 0) {
			// reset the flux registers to their pre-advance state
			if (fr_as_crse != nullptr) {
				fr_as_crse->reset();
			}
			if (fr_as_fine != nullptr) {
				amrex::Copy(fr_as_fine->getFineData(), originalFineData, 0, 0, originalFineData.nComp(), 0);
			}

#ifdef AMREX_PARTICLES
			if (do_tracers != 0) {
				// reset the tracer particles to their pre-advance state
				TracerPC->copyParticles(originalTracerPC, true);
			}
#endif
		}

		// create temporary multifab for old state
		amrex::MultiFab state_old_cc_tmp(grids[lev], dmap[lev], Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);
		amrex::Copy(state_old_cc_tmp, state_old_cc_[lev], 0, 0, Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);

		// subcycle advanceHydroAtLevel, checking return value
		for (int substep = 0; substep < nsubsteps; ++substep) {
			if (substep > 0) {
				// since we are starting a new substep, we need to copy hydro state from
				//  the new state vector to old state vector
				amrex::Copy(state_old_cc_tmp, state_new_cc_[lev], 0, 0, ncompHydro_, nghost_cc_);
			}

			success = advanceHydroAtLevel(state_old_cc_tmp, fr_as_crse, fr_as_fine, lev, time, dt_step);

			if (!success) {
				if (Verbose()) {
					amrex::Print() << "\t>> WARNING: Hydro advance failed on level " << lev << "\n";
				}
				break;
			}
		}

		if (success) {
			// we are done, do not attempt more retries
			break;
		}
	}

	if (!success) {
		// crash, we have exceeded max_retries
		amrex::Print() << "\nQUOKKA FATAL ERROR\n"
			       << "Hydro update exceeded max_retries on level " << lev << ". Cannot continue, crashing...\n"
			       << std::endl; // NOLINT(performance-avoid-endl)

		// write plotfile or Ascent Blueprint file
		amrex::ParallelDescriptor::Barrier();
#ifdef AMREX_USE_ASCENT
		conduit::Node mesh;
		amrex::SingleLevelToBlueprint(state_new_cc_[lev], componentNames_cc_, geom[lev], time, istep[lev] + 1, mesh);
		conduit::Node bpMeshHost;
		bpMeshHost.set(mesh); // copy to host mem (needed for Blueprint HDF5 output)
		amrex::WriteBlueprintFiles(bpMeshHost, "debug_hydro_state_fatal", istep[lev] + 1, "hdf5");
#else
		WriteSingleLevelPlotfile(CustomPlotFileName("debug_hydro_state_fatal", istep[lev] + 1), state_new_cc_[lev], componentNames_cc_, geom[lev], time,
					 istep[lev] + 1);
#endif
		amrex::ParallelDescriptor::Barrier();

		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::ParallelDescriptor::Abort();
		}
	}
}

template <typename problem_t> auto QuokkaSimulation<problem_t>::isCflViolated(int lev, amrex::Real /*time*/, amrex::Real dt_actual) -> bool
{
	// check whether dt_actual would violate CFL condition using the post-update hydro state

	// compute max signal speed
	amrex::Real max_signal = HydroSystem<problem_t>::maxSignalSpeedLocal(state_new_cc_[lev]);
	amrex::ParallelDescriptor::ReduceRealMax(max_signal);

	// compute dt_cfl
	auto dx = geom[lev].CellSizeArray();
	const amrex::Real dx_min = std::min({AMREX_D_DECL(dx[0], dx[1], dx[2])});
	const amrex::Real dt_cfl = cflNumber_ * (dx_min / max_signal);

	// check whether dt_actual > dt_cfl (CFL violation)
	const amrex::Real max_factor = 1.1;
	const bool cflViolation = dt_actual > (max_factor * dt_cfl);
	if (cflViolation && Verbose()) {
		amrex::Print() << "\t>> CFL violation detected on level " << lev << " with dt_lev = " << dt_actual << " and dt_cfl = " << dt_cfl << "\n"
			       << "\t   max_signal = " << max_signal << "\n";
	}
	return cflViolation;
}

template <typename problem_t> void QuokkaSimulation<problem_t>::printCoordinates(int lev, const amrex::IntVect &cell_idx)
{

	amrex::Real x_coord = geom[lev].ProbLo(0) + (cell_idx[0] + 0.5) * geom[lev].CellSize(0);
	amrex::Real y_coord = NAN;
	amrex::Real z_coord = NAN;

	if (AMREX_SPACEDIM > 1) {
		y_coord = geom[lev].ProbLo(1) + (cell_idx[1] + 0.5) * geom[lev].CellSize(1);

		if (AMREX_SPACEDIM > 2) {
			z_coord = geom[lev].ProbLo(2) + (cell_idx[2] + 0.5) * geom[lev].CellSize(2);
		}
	}
	amrex::Print() << "Coordinates: (" << x_coord << ", " << y_coord << ", " << z_coord << "): ";
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::advanceHydroAtLevel(amrex::MultiFab &state_old_cc_tmp, amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
						      int lev, amrex::Real time, amrex::Real dt_lev) -> bool
{
	BL_PROFILE("QuokkaSimulation::advanceHydroAtLevel()");

	amrex::Real fluxScaleFactor = NAN;
	if (integratorOrder_ == 2) {
		fluxScaleFactor = 0.5;
	} else if (integratorOrder_ == 1) {
		fluxScaleFactor = 1.0;
	}

	auto dx = geom[lev].CellSizeArray();

	// do Strang split source terms (first half-step)
	auto burn_success_first = addStrangSplitSourcesWithBuiltin(state_old_cc_tmp, lev, time, 0.5 * dt_lev);

	// check if reactions failed for source terms. If it failed, return false.
	if (!burn_success_first) {
		return burn_success_first;
	}

	// create temporary multifab for intermediate state
	amrex::MultiFab state_inter_cc_(grids[lev], dmap[lev], Physics_Indices<problem_t>::nvarTotal_cc, nghost_cc_);
	state_inter_cc_.setVal(0); // prevent assert in fillBoundaryConditions when radiation is enabled

	// create temporary multifabs for combined RK2 flux and time-average face velocity
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux_rk2;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> avgFaceVel;
	const int nghost_vel = 2; // 2 ghost faces are needed for tracer particles
	auto ba = grids[lev];
	auto dm = dmap[lev];
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		// initialize flux MultiFab
		flux_rk2[idim] = amrex::MultiFab(ba_face, dm, ncompHydro_, 0);
		flux_rk2[idim].setVal(0);
		// initialize velocity MultiFab
		avgFaceVel[idim] = amrex::MultiFab(ba_face, dm, 1, nghost_vel);
		avgFaceVel[idim].setVal(0);
	}

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_cc_tmp, state_old_cc_tmp, lev, time, quokka::centering::cc, quokka::direction::na, PreInterpState, PostInterpState);

	// LOW LEVEL DEBUGGING: output state_old_cc_tmp (with ghost cells)
	if (lowLevelDebuggingOutput_ == 1) {
#ifdef AMREX_USE_ASCENT
		// write Blueprint HDF5 files
		conduit::Node mesh;
		amrex::SingleLevelToBlueprint(state_old_cc_tmp, componentNames_cc_, geom[lev], time, istep[lev] + 1, mesh);
		amrex::WriteBlueprintFiles(mesh, "debug_stage1_filled_state_old", istep[lev] + 1, "hdf5");
#else
		// write AMReX plotfile
		// WriteSingleLevelPlotfile(CustomPlotFileName("debug_stage1_filled_state_old", istep[lev]+1),
		//    state_old_cc_tmp, componentNames_cc_, geom[lev], time, istep[lev]+1);
#endif
	}

	// check state validity
	AMREX_ASSERT(!state_old_cc_tmp.contains_nan(0, state_old_cc_tmp.nComp()));
	AMREX_ASSERT(!state_old_cc_tmp.contains_nan()); // check ghost cells

	auto [FOfluxArrays, FOfaceVel] = computeFOHydroFluxes(state_old_cc_tmp, ncompHydro_, lev);

	// Stage 1 of RK2-SSP
	{
		// advance all grids on local processor (Stage 1 of integrator)
		auto const &stateOld = state_old_cc_tmp;
		auto &stateNew = state_inter_cc_;
		auto [fluxArrays, faceVel] = computeHydroFluxes(stateOld, ncompHydro_, lev);

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::MultiFab::Saxpy(flux_rk2[idim], 0.5, fluxArrays[idim], 0, 0, ncompHydro_, 0);
			amrex::MultiFab::Saxpy(avgFaceVel[idim], 0.5, faceVel[idim], 0, 0, 1, 0);
		}

		amrex::MultiFab rhs(grids[lev], dmap[lev], ncompHydro_, 0);
		amrex::iMultiFab redoFlag(grids[lev], dmap[lev], 1, 1);
		redoFlag.setVal(quokka::redoFlag::none);

		HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, fluxArrays, dx, ncompHydro_);
		HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld, dx, faceVel, redoFlag);
		HydroSystem<problem_t>::PredictStep(stateOld, stateNew, rhs, dt_lev, ncompHydro_, redoFlag);

		// LOW LEVEL DEBUGGING: output rhs
		if (lowLevelDebuggingOutput_ == 1) {
			// write rhs
			std::string plotfile_name = CustomPlotFileName("debug_stage1_rhs_fluxes", istep[lev] + 1);
			WriteSingleLevelPlotfile(plotfile_name, rhs, componentNames_cc_, geom[lev], time, istep[lev] + 1);

			// write fluxes
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (amrex::ParallelDescriptor::IOProcessor()) {
					std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
				}
				std::string fullprefix =
				    amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_", std::string("Flux_") + quokka::face_dir_str[idim]);
				amrex::VisMF::Write(fluxArrays[idim], fullprefix);
			}
			// write face velocities
			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
				if (amrex::ParallelDescriptor::IOProcessor()) {
					std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
				}
				std::string fullprefix = amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_",
										       std::string("FaceVel_") + quokka::face_dir_str[idim]);
				amrex::VisMF::Write(faceVel[idim], fullprefix);
			}
		}

		// do first-order flux correction (FOFC)
		amrex::Gpu::streamSynchronizeAll(); // just in case
		amrex::Long const ncells_bad = redoFlag.sum(0);
		if (ncells_bad > 0) {
			if (Verbose()) {
				amrex::Print() << "[FOFC-1] flux correcting " << ncells_bad << " cells on level " << lev << "\n";
				const amrex::IntVect cell_idx = redoFlag.maxIndex(0);
				// Calculate the coordinates based on the cell index and cell size
				printCoordinates(lev, cell_idx);
				amrex::print_state(stateNew, cell_idx);
			}

			// synchronize redoFlag across ranks
			redoFlag.FillBoundary(geom[lev].periodicity());

			// replace fluxes around troubled cells with Godunov fluxes
			replaceFluxes(fluxArrays, FOfluxArrays, redoFlag);
			replaceFluxes(faceVel, FOfaceVel, redoFlag); // needed for dual energy

			// re-do RK update
			HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, fluxArrays, dx, ncompHydro_);
			HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld, dx, faceVel, redoFlag);
			HydroSystem<problem_t>::PredictStep(stateOld, stateNew, rhs, dt_lev, ncompHydro_, redoFlag);

			amrex::Gpu::streamSynchronizeAll(); // just in case
			amrex::Long const ncells_bad = static_cast<int>(redoFlag.sum(0));
			if (ncells_bad > 0) {
				// FOFC failed
				if (Verbose()) {
					const amrex::IntVect cell_idx = redoFlag.maxIndex(0);
					// print cell state
					amrex::Print() << "[FOFC-1] Flux correction failed:\n";
					printCoordinates(lev, cell_idx);
					amrex::print_state(stateNew, cell_idx);
					amrex::Print() << "[FOFC-1] failed for " << ncells_bad << " cells on level " << lev << "\n";
				}
				if (abortOnFofcFailure_ != 0) {
					return false;
				}
			}
		}

		// prevent vacuum
		HydroSystem<problem_t>::EnforceLimits(densityFloor_, tempFloor_, stateNew);

		if (useDualEnergy_ == 1) {
			// sync internal energy (requires positive density)
			HydroSystem<problem_t>::SyncDualEnergy(stateNew);
		}

		if (do_reflux == 1) {
			// increment flux registers
			incrementFluxRegisters(fr_as_crse, fr_as_fine, fluxArrays, lev, fluxScaleFactor * dt_lev);
		}
	}
	amrex::Gpu::streamSynchronizeAll();

	// Stage 2 of RK2-SSP
	if (integratorOrder_ == 2) {
		// update ghost zones [intermediate stage stored in state_inter_cc_]
		fillBoundaryConditions(state_inter_cc_, state_inter_cc_, lev, time + dt_lev, quokka::centering::cc, quokka::direction::na, PreInterpState,
				       PostInterpState);

		// check intermediate state validity
		AMREX_ASSERT(!state_inter_cc_.contains_nan(0, state_inter_cc_.nComp()));
		AMREX_ASSERT(!state_inter_cc_.contains_nan()); // check ghost zones

		// write out FABs with ghost zones
		// amrex::writeFabs(state_inter_cc_, "state_inter_cc_" + std::to_string(istep[lev]));

		auto const &stateOld = state_old_cc_tmp;
		auto const &stateInter = state_inter_cc_;
		auto &stateFinal = state_new_cc_[lev];
		auto [fluxArrays, faceVel] = computeHydroFluxes(stateInter, ncompHydro_, lev);

		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::MultiFab::Saxpy(flux_rk2[idim], 0.5, fluxArrays[idim], 0, 0, ncompHydro_, 0);
			amrex::MultiFab::Saxpy(avgFaceVel[idim], 0.5, faceVel[idim], 0, 0, 1, 0);
		}

		amrex::MultiFab rhs(grids[lev], dmap[lev], ncompHydro_, 0);
		amrex::iMultiFab redoFlag(grids[lev], dmap[lev], 1, 1);
		redoFlag.setVal(quokka::redoFlag::none);

		HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, flux_rk2, dx, ncompHydro_);
		HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld, dx, avgFaceVel, redoFlag);
		HydroSystem<problem_t>::PredictStep(stateOld, stateFinal, rhs, dt_lev, ncompHydro_, redoFlag);

		// do first-order flux correction (FOFC)
		amrex::Gpu::streamSynchronizeAll(); // just in case
		amrex::Long const ncells_bad = redoFlag.sum(0);
		if (ncells_bad > 0) {
			if (Verbose()) {
				amrex::Print() << "[FOFC-2] flux correcting " << ncells_bad << " cells on level " << lev << "\n";
				const amrex::IntVect cell_idx = redoFlag.maxIndex(0);
				printCoordinates(lev, cell_idx);
				amrex::print_state(stateFinal, cell_idx);
			}

			// synchronize redoFlag across ranks
			redoFlag.FillBoundary(geom[lev].periodicity());

			// replace fluxes around troubled cells with Godunov fluxes
			replaceFluxes(flux_rk2, FOfluxArrays, redoFlag);
			replaceFluxes(avgFaceVel, FOfaceVel, redoFlag); // needed for dual energy

			// re-do RK update
			HydroSystem<problem_t>::ComputeRhsFromFluxes(rhs, flux_rk2, dx, ncompHydro_);
			HydroSystem<problem_t>::AddInternalEnergyPdV(rhs, stateOld, dx, avgFaceVel, redoFlag);
			HydroSystem<problem_t>::PredictStep(stateOld, stateFinal, rhs, dt_lev, ncompHydro_, redoFlag);

			amrex::Gpu::streamSynchronizeAll(); // just in case
			amrex::Long ncells_bad = redoFlag.sum(0);
			if (ncells_bad > 0) {
				// FOFC failed
				if (Verbose()) {
					const amrex::IntVect cell_idx = redoFlag.maxIndex(0);
					// print cell state
					amrex::Print() << "[FOFC-2] Flux correction failed:\n";
					printCoordinates(lev, cell_idx);
					amrex::print_state(stateFinal, cell_idx);
					amrex::Print() << "[FOFC-2] failed for " << ncells_bad << " cells on level " << lev << "\n";
				}
				if (abortOnFofcFailure_ != 0) {
					return false;
				}
			}
		}

		// prevent vacuum
		HydroSystem<problem_t>::EnforceLimits(densityFloor_, tempFloor_, stateFinal);

		if (useDualEnergy_ == 1) {
			// sync internal energy (requires positive density)
			HydroSystem<problem_t>::SyncDualEnergy(stateFinal);
		}

		if (do_reflux == 1) {
			// increment flux registers
			incrementFluxRegisters(fr_as_crse, fr_as_fine, fluxArrays, lev, fluxScaleFactor * dt_lev);
		}
	} else { // we are only doing forward Euler
		amrex::Copy(state_new_cc_[lev], state_inter_cc_, 0, 0, ncompHydro_, 0);
	}
	amrex::Gpu::streamSynchronizeAll();

	// advect tracer particles using avgFaceVel
#ifdef AMREX_PARTICLES
	if (do_tracers != 0) {
		// copy avgFaceVel to state_new_fc_[lev]
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::Copy(state_new_fc_[lev][idim], avgFaceVel[idim], Physics_Indices<problem_t>::velFirstIndex, 0,
				    Physics_NumVars::numVelVars_per_dim, nghost_vel);
		}

		// fill ghost faces
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			fillBoundaryConditions(state_new_fc_[lev][idim], state_new_fc_[lev][idim], lev, time + 0.5 * dt_lev, quokka::centering::fc,
					       quokka::direction{idim}, AMRSimulation<problem_t>::InterpHookNone, AMRSimulation<problem_t>::InterpHookNone,
					       FillPatchType::fillpatch_function);
		}

		// copy state_new_fc_[lev] to avgFaceVel
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			amrex::Copy(avgFaceVel[idim], state_new_fc_[lev][idim], 0, Physics_Indices<problem_t>::velFirstIndex,
				    Physics_NumVars::numVelVars_per_dim, nghost_vel);
		}

		// advect particles
		TracerPC->AdvectWithUmac(avgFaceVel.data(), lev, dt_lev);
	}
#endif

	// do Strang split source terms (second half-step)
	auto burn_success_second = addStrangSplitSourcesWithBuiltin(state_new_cc_[lev], lev, time + dt_lev, 0.5 * dt_lev);

	// check if we have violated the CFL timestep or reactions failed for source terms
	return (!isCflViolated(lev, time, dt_lev) && burn_success_second);
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::replaceFluxes(std::array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &FOfluxes,
						amrex::iMultiFab &redoFlag)
{
	BL_PROFILE("QuokkaSimulation::replaceFluxes()");

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) { // loop over dimension
		// ensure that flux arrays have the same number of components
		AMREX_ASSERT(fluxes[idim].nComp() == FOfluxes[idim].nComp());
		int ncomp = fluxes[idim].nComp();

		auto const &FOflux_arrs = FOfluxes[idim].const_arrays();
		auto const &redoFlag_arrs = redoFlag.const_arrays();
		auto flux_arrs = fluxes[idim].arrays();

		// By convention, the fluxes are defined on the left edge of each zone,
		// i.e. flux_(i) is the flux *into* zone i through the interface on the
		// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
		// the interface on the right of zone i.

		amrex::IntVect ng{AMREX_D_DECL(1, 1, 1)}; // must include 1 ghost zone

		amrex::ParallelFor(redoFlag, ng, ncomp, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k, int n) noexcept {
			if (redoFlag_arrs[bx](i, j, k) == quokka::redoFlag::redo) {
				// replace fluxes with first-order ones at the faces of cell (i,j,k)
				if (flux_arrs[bx].contains(i, j, k)) {
					flux_arrs[bx](i, j, k, n) = FOflux_arrs[bx](i, j, k, n);
				}
				if (idim == 0) { // x-dir fluxes
					if (flux_arrs[bx].contains(i + 1, j, k)) {
						flux_arrs[bx](i + 1, j, k, n) = FOflux_arrs[bx](i + 1, j, k, n);
					}
				} else if (idim == 1) { // y-dir fluxes
					if (flux_arrs[bx].contains(i, j + 1, k)) {
						flux_arrs[bx](i, j + 1, k, n) = FOflux_arrs[bx](i, j + 1, k, n);
					}
				} else if (idim == 2) { // z-dir fluxes
					if (flux_arrs[bx].contains(i, j, k + 1)) {
						flux_arrs[bx](i, j, k + 1, n) = FOflux_arrs[bx](i, j, k + 1, n);
					}
				}
			}
		});
	}
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::addFluxArrays(std::array<amrex::MultiFab, AMREX_SPACEDIM> &dstfluxes, std::array<amrex::MultiFab, AMREX_SPACEDIM> &srcfluxes,
						const int srccomp, const int dstcomp)
{
	BL_PROFILE("QuokkaSimulation::addFluxArrays()");

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto const &srcflux = srcfluxes[idim];
		auto &dstflux = dstfluxes[idim];
		amrex::Add(dstflux, srcflux, srccomp, dstcomp, srcflux.nComp(), 0);
	}
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::expandFluxArrays(std::array<amrex::FArrayBox, AMREX_SPACEDIM> &fluxes, const int nstartNew, const int ncompNew)
    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>
{
	BL_PROFILE("QuokkaSimulation::expandFluxArrays()");

	// This is needed because reflux arrays must have the same number of components as
	// state_new_cc_[lev]
	auto copyFlux = [nstartNew, ncompNew](amrex::FArrayBox const &oldFlux) {
		amrex::Box const &fluxRange = oldFlux.box();
		amrex::FArrayBox newFlux(fluxRange, ncompNew, amrex::The_Async_Arena());
		newFlux.setVal<amrex::RunOn::Device>(0.);
		// copy oldFlux (starting at 0) to newFlux (starting at nstart)
		AMREX_ASSERT(ncompNew >= oldFlux.nComp());
		newFlux.copy<amrex::RunOn::Device>(oldFlux, 0, nstartNew, oldFlux.nComp());
		return newFlux;
	};
	return {AMREX_D_DECL(copyFlux(fluxes[0]), copyFlux(fluxes[1]), copyFlux(fluxes[2]))};
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::computeHydroFluxes(amrex::MultiFab const &consVar, const int nvars, const int lev)
    -> std::pair<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>
{
	BL_PROFILE("QuokkaSimulation::computeHydroFluxes()");

	auto ba = grids[lev];
	auto dm = dmap[lev];
	const int flatteningGhost = 2;
	const int reconstructGhost = 1;

	// allocate temporary MultiFabs
	amrex::MultiFab primVar(ba, dm, nvars, nghost_cc_);
	std::array<amrex::MultiFab, 3> flatCoefs;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;

	for (int idim = 0; idim < 3; ++idim) {
		flatCoefs[idim] = amrex::MultiFab(ba, dm, 1, flatteningGhost);
	}

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost);
		rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructGhost);
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, 0);
		facevel[idim] = amrex::MultiFab(ba_face, dm, 1, 0);
	}

	// conserved to primitive variables
	HydroSystem<problem_t>::ConservedToPrimitive(consVar, primVar, nghost_cc_);

	// compute flattening coefficients
	AMREX_D_TERM(HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X1>(primVar, flatCoefs[0], flatteningGhost);
		     , HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X2>(primVar, flatCoefs[1], flatteningGhost);
		     , HydroSystem<problem_t>::template ComputeFlatteningCoefficients<FluxDir::X3>(primVar, flatCoefs[2], flatteningGhost);)

	// compute flux functions
	AMREX_D_TERM(hydroFluxFunction<FluxDir::X1>(primVar, leftState[0], rightState[0], flux[0], facevel[0], flatCoefs[0], flatCoefs[1], flatCoefs[2],
						    reconstructGhost, nvars);
		     , hydroFluxFunction<FluxDir::X2>(primVar, leftState[1], rightState[1], flux[1], facevel[1], flatCoefs[0], flatCoefs[1], flatCoefs[2],
						      reconstructGhost, nvars);
		     , hydroFluxFunction<FluxDir::X3>(primVar, leftState[2], rightState[2], flux[2], facevel[2], flatCoefs[0], flatCoefs[1], flatCoefs[2],
						      reconstructGhost, nvars);)

	// synchronization point to prevent MultiFabs from going out of scope
	amrex::Gpu::streamSynchronizeAll();

	// LOW LEVEL DEBUGGING: output all of the temporary MultiFabs
	if (lowLevelDebuggingOutput_ == 1) {
		// write primitive cell-centered state
		std::string plotfile_name = CustomPlotFileName("debug_reconstruction", istep[lev] + 1);
		WriteSingleLevelPlotfile(plotfile_name, primVar, componentNames_cc_, geom[lev], 0.0, istep[lev] + 1);

		// write flattening coefficients
		std::string flatx_filename = CustomPlotFileName("debug_flattening_x", istep[lev] + 1);
		std::string flaty_filename = CustomPlotFileName("debug_flattening_y", istep[lev] + 1);
		std::string flatz_filename = CustomPlotFileName("debug_flattening_z", istep[lev] + 1);
		amrex::Vector<std::string> flatCompNames{"chi"};
		WriteSingleLevelPlotfile(flatx_filename, flatCoefs[0], flatCompNames, geom[lev], 0.0, istep[lev] + 1);
		WriteSingleLevelPlotfile(flaty_filename, flatCoefs[1], flatCompNames, geom[lev], 0.0, istep[lev] + 1);
		WriteSingleLevelPlotfile(flatz_filename, flatCoefs[2], flatCompNames, geom[lev], 0.0, istep[lev] + 1);

		// write L interface states
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (amrex::ParallelDescriptor::IOProcessor()) {
				std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
			}
			std::string const fullprefix =
			    amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_", std::string("StateL_") + quokka::face_dir_str[idim]);
			amrex::VisMF::Write(leftState[idim], fullprefix);
		}
		// write R interface states
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			if (amrex::ParallelDescriptor::IOProcessor()) {
				std::filesystem::create_directories(plotfile_name + "/raw_fields/Level_" + std::to_string(lev));
			}
			std::string const fullprefix =
			    amrex::MultiFabFileFullPrefix(lev, plotfile_name, "raw_fields/Level_", std::string("StateR_") + quokka::face_dir_str[idim]);
			amrex::VisMF::Write(rightState[idim], fullprefix);
		}
	}

	// return flux and face-centered velocities
	return std::make_pair(std::move(flux), std::move(facevel));
}

template <typename problem_t>
template <FluxDir DIR>
void QuokkaSimulation<problem_t>::hydroFluxFunction(amrex::MultiFab const &primVar, amrex::MultiFab &leftState, amrex::MultiFab &rightState,
						    amrex::MultiFab &flux, amrex::MultiFab &faceVel, amrex::MultiFab const &x1Flat,
						    amrex::MultiFab const &x2Flat, amrex::MultiFab const &x3Flat, const int ng_reconstruct, const int nvars)
{
	if (reconstructionOrder_ == 3) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
	} else if (reconstructionOrder_ == 2) {
		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, SlopeLimiter::minmod>(primVar, leftState, rightState, ng_reconstruct, nvars);
	} else if (reconstructionOrder_ == 1) {
		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
	} else {
		amrex::Abort("Invalid reconstruction order specified!");
	}

	// cell-centered kernel
	HydroSystem<problem_t>::template FlattenShocks<DIR>(primVar, x1Flat, x2Flat, x3Flat, leftState, rightState, ng_reconstruct, nvars);

	// interface-centered kernel
	if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
		HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::HLLD, DIR>(flux, faceVel, leftState, rightState, primVar, artificialViscosityK_);
	} else {
		HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::HLLC, DIR>(flux, faceVel, leftState, rightState, primVar, artificialViscosityK_);
	}
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::computeFOHydroFluxes(amrex::MultiFab const &consVar, const int nvars, const int lev)
    -> std::pair<std::array<amrex::MultiFab, AMREX_SPACEDIM>, std::array<amrex::MultiFab, AMREX_SPACEDIM>>
{
	BL_PROFILE("QuokkaSimulation::computeFOHydroFluxes()");

	auto ba = grids[lev];
	auto dm = dmap[lev];
	const int reconstructRange = 1;

	// allocate temporary MultiFabs
	amrex::MultiFab primVar(ba, dm, nvars, nghost_cc_);
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> facevel;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, 0);
		facevel[idim] = amrex::MultiFab(ba_face, dm, 1, 0);
	}

	// conserved to primitive variables
	HydroSystem<problem_t>::ConservedToPrimitive(consVar, primVar, nghost_cc_);

	// compute flux functions
	AMREX_D_TERM(hydroFOFluxFunction<FluxDir::X1>(primVar, leftState[0], rightState[0], flux[0], facevel[0], reconstructRange, nvars);
		     , hydroFOFluxFunction<FluxDir::X2>(primVar, leftState[1], rightState[1], flux[1], facevel[1], reconstructRange, nvars);
		     , hydroFOFluxFunction<FluxDir::X3>(primVar, leftState[2], rightState[2], flux[2], facevel[2], reconstructRange, nvars);)

	// synchronization point to prevent MultiFabs from going out of scope
	amrex::Gpu::streamSynchronizeAll();

	// return flux and face-centered velocities
	return std::make_pair(std::move(flux), std::move(facevel));
}

template <typename problem_t>
template <FluxDir DIR>
void QuokkaSimulation<problem_t>::hydroFOFluxFunction(amrex::MultiFab const &primVar, amrex::MultiFab &leftState, amrex::MultiFab &rightState,
						      amrex::MultiFab &flux, amrex::MultiFab &faceVel, const int ng_reconstruct, const int nvars)
{
	// donor-cell reconstruction
	HydroSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar, leftState, rightState, ng_reconstruct, nvars);
	// LLF solver
	HydroSystem<problem_t>::template ComputeFluxes<RiemannSolver::LLF, DIR>(flux, faceVel, leftState, rightState, primVar, artificialViscosityK_);
}

template <typename problem_t> void QuokkaSimulation<problem_t>::swapRadiationState(amrex::MultiFab &stateOld, amrex::MultiFab const &stateNew)
{
	// copy radiation state variables from stateNew to stateOld
	amrex::MultiFab::Copy(stateOld, stateNew, nstartHyperbolic_, nstartHyperbolic_, ncompHyperbolic_, 0);
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::subcycleRadiationAtLevel(int lev, amrex::Real time, amrex::Real dt_lev_hydro, amrex::YAFluxRegister *fr_as_crse,
							   amrex::YAFluxRegister *fr_as_fine)
{
	// compute radiation timestep
	int nsubSteps = 0;
	amrex::Real dt_radiation = NAN;

	if (Physics_Traits<problem_t>::is_hydro_enabled && !(constantDt_ > 0.)) {
		// adjust to get integer number of substeps
		nsubSteps = computeNumberOfRadiationSubsteps(lev, dt_lev_hydro);
		dt_radiation = dt_lev_hydro / static_cast<double>(nsubSteps);
	} else { // no hydro, or using constant dt (this is necessary for radiation test problems)
		dt_radiation = dt_lev_hydro;
		nsubSteps = 1;
	}

	if (Verbose() != 0) {
		amrex::Print() << "\tRadiation substeps: " << nsubSteps << "\tdt: " << dt_radiation << "\n";
	}
	AMREX_ALWAYS_ASSERT(nsubSteps >= 1);
	AMREX_ALWAYS_ASSERT(nsubSteps <= (maxSubsteps_ + 1));
	AMREX_ALWAYS_ASSERT(dt_radiation > 0.0);

	// perform subcycle
	auto const &dx = geom[lev].CellSizeArray();
	amrex::Real time_subcycle = time;
	for (int i = 0; i < nsubSteps; ++i) {
		if (i > 0) {
			// since we are starting a new substep, we need to copy radiation state from
			//     new state vector to old state vector
			// (this is not necessary for the i=0 substep because we have already swapped
			//  the full hydro+radiation state vectors at the beginning of the level advance)
			swapRadiationState(state_old_cc_[lev], state_new_cc_[lev]);
		}

		// define a MultiFab to store the reduced speed of light
		amrex::MultiFab reducedSpeedOfLightFactor(grids[lev], dmap[lev], RadSystem<problem_t>::nGroups_, 2);

		if (variable_chat_param1_ < 0.0) {
			reducedSpeedOfLightFactor.setVal(chat_over_c_);
		} else {
			for (amrex::MFIter iter(reducedSpeedOfLightFactor); iter.isValid(); ++iter) {
				auto const &stateNew = state_new_cc_[lev].array(iter);
				const amrex::Box &indexRange = amrex::grow(iter.validbox(), 2);
				RadSystem<problem_t>::ComputeReducedSpeedOfLightFactor(stateNew, chat_over_c_, variable_chat_param1_, variable_chat_param2_,
										       reducedSpeedOfLightFactor.array(iter), indexRange, dx);
			}
		}

		// Debug on CPU
		// if (time_subcycle == 0.0) {
		// 	// print reducedSpeedOfLightFactor
		// 	for (amrex::MFIter iter(reducedSpeedOfLightFactor); iter.isValid(); ++iter) {
		// 		const amrex::Box &indexRange = iter.validbox();
		// 		auto const &reducedSpeedOfLightFactor_ = reducedSpeedOfLightFactor.const_array(iter);
		// 		amrex::Print() << "\nreducedSpeedOfLightFactor (" << RadSystem<problem_t>::nGroups_ << " groups): \n";
		// 		for (int g = 0; g < RadSystem<problem_t>::nGroups_; ++g) {
		// 			amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// 				amrex::Print() << reducedSpeedOfLightFactor_(i, j, k, g) << " ";
		// 			});
		// 			amrex::Print() << "\n";
		// 		}
		// 	}
		// }

		// We use the IMEX PD-ARS scheme to evolve the radiation subsystem and radiation-matter coupling.

		// Stage 1: advance hyperbolic radiation subsystem using Forward Euler method, starting from state_old_cc_ to state_new_cc_
		advanceRadiationForwardEuler(lev, time_subcycle, dt_radiation, i, nsubSteps, fr_as_crse, fr_as_fine, reducedSpeedOfLightFactor);

		// new radiation state is stored in state_new_cc_
		// new hydro state is stored in state_new_cc_ (always the case during radiation update)

		// failure counter for: matter-radiation coupling, dust temperature, outer iteration
		amrex::Gpu::Buffer<int> iteration_failure_counter({0, 0, 0});
		// iteration counter for: radiation update, Newton-Raphson iterations, max Newton-Raphson iterations, decoupled gas-dust update
		amrex::Gpu::Buffer<int> iteration_counter({0, 0, 0, 0});
		int *p_iteration_failure_counter = iteration_failure_counter.data();
		int *p_iteration_counter = iteration_counter.data();

		if constexpr (IMEX_a22 > 0.0) {
			// matter-radiation exchange source terms of stage 1

			for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
				const amrex::Box &indexRange = iter.validbox();
				auto const &stateNew = state_new_cc_[lev].array(iter);
				auto const &prob_lo = geom[lev].ProbLoArray();
				auto const &prob_hi = geom[lev].ProbHiArray();
				// update state_new_cc_[lev] in place (updates both radiation and hydro vars)
				// Note that only a fraction (IMEX_a32) of the matter-radiation exchange source terms are added to hydro. This ensures that the
				// hydro properties get to t + IMEX_a32 dt in terms of matter-radiation exchange.
				operatorSplitSourceTerms(stateNew, indexRange, time_subcycle, dt_radiation, 1, dx, prob_lo, prob_hi,
							 reducedSpeedOfLightFactor.array(iter), p_iteration_counter, p_iteration_failure_counter);
			}
		}

		// Stage 2: advance hyperbolic radiation subsystem using midpoint RK2 method, starting from state_old_cc_ to state_new_cc_
		advanceRadiationMidpointRK2(lev, time_subcycle, dt_radiation, i, nsubSteps, fr_as_crse, fr_as_fine, reducedSpeedOfLightFactor);

		// new radiation state is stored in state_new_cc_
		// new hydro state is stored in state_new_cc_ (always the case during radiation update)

		// Add the matter-radiation exchange source terms to the radiation subsystem and evolve by (1 - IMEX_a32) * dt
		for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &stateNew = state_new_cc_[lev].array(iter);
			auto const &prob_lo = geom[lev].ProbLoArray();
			auto const &prob_hi = geom[lev].ProbHiArray();
			// update state_new_cc_[lev] in place (updates both radiation and hydro vars)
			operatorSplitSourceTerms(stateNew, indexRange, time_subcycle, dt_radiation, 2, dx, prob_lo, prob_hi,
						 reducedSpeedOfLightFactor.array(iter), p_iteration_counter, p_iteration_failure_counter);
		}

		if (print_rad_counter_) {
			auto h_iteration_counter = iteration_counter.copyToHost();
			long global_solver_count = h_iteration_counter[0];	      // number of Newton-Raphson solvings
			long global_iteration_sum = h_iteration_counter[1];	      // sum of Newton-Raphson iterations
			int global_iteration_max = h_iteration_counter[2];	      // max number of Newton-Raphson iterations
			long global_decoupled_iteration_sum = h_iteration_counter[3]; // sum of decoupled gas-dust Newton-Raphson iterations

			amrex::ParallelDescriptor::ReduceLongSum(global_solver_count);
			amrex::ParallelDescriptor::ReduceLongSum(global_iteration_sum);
			amrex::ParallelDescriptor::ReduceIntMax(global_iteration_max);
			amrex::ParallelDescriptor::ReduceLongSum(global_decoupled_iteration_sum);

			if (amrex::ParallelDescriptor::IOProcessor()) {
				const auto n_cells = CountCells(lev);
				amrex::Print() << "time_subcycle = " << time_subcycle << ", total number of cells updated is " << n_cells << "\n";
				if (n_cells > 0 && global_solver_count > 0) {
					const double global_iteration_mean =
					    static_cast<double>(global_iteration_sum) / static_cast<double>(global_solver_count);
					const double global_solving_mean =
					    static_cast<double>(global_solver_count) / static_cast<double>(n_cells) / 2.0; // 2 stages
					const double global_decoupled_iteration_mean =
					    static_cast<double>(global_decoupled_iteration_sum) / static_cast<double>(global_solver_count);
					amrex::Print() << "The average number of Newton-Raphson solvings per IMEX stage is " << global_solving_mean
						       << ", (mean, max) number of Newton-Raphson iterations are " << global_iteration_mean << ", "
						       << global_iteration_max << ".\n";
					if constexpr (ISM_Traits<problem_t>::enable_dust_gas_thermal_coupling_model) {
						amrex::Print() << "The fraction of gas-dust interactions that are decoupled is "
							       << global_decoupled_iteration_mean << "\n";
					}
				}
			}
		}

		auto h_iteration_failure_counter = iteration_failure_counter.copyToHost();
		long nf_coupling = h_iteration_failure_counter[0]; // number of matter-radiation coupling failures
		long nf_dust = h_iteration_failure_counter[1];	   // number of dust temperature failures
		long nf_outer = h_iteration_failure_counter[2];	   // number of outer iterations failures

		amrex::ParallelDescriptor::ReduceLongSum(nf_coupling);
		amrex::ParallelDescriptor::ReduceLongSum(nf_dust);
		amrex::ParallelDescriptor::ReduceLongSum(nf_outer);

		// Note that the nf_dust has to abort BEFORE nf_coupling, because the dust temperature is used in the matter-radiation coupling and if dust
		// temperature is negative, the matter-radiation coupling will fail to converge.
		if (nf_dust > 0) {
			amrex::Abort("Newton-Raphson iteration for dust temperature failed to converge or dust temperature is negative!");
		}
		if (nf_coupling > 0) {
			amrex::Abort("Newton-Raphson iteration for matter-radiation coupling failed to converge!");
		}
		if (nf_outer > 0) {
			amrex::Abort("Outer iteration for matter-radiation coupling failed to converge!");
		}

		// new hydro+radiation state is stored in state_new_cc_

		// update 'time_subcycle'
		time_subcycle += dt_radiation;

		// update cell update counter
		radiationCellUpdates_ += CountCells(lev); // keep track of number of cell updates
	}
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::advanceRadiationForwardEuler(int lev, amrex::Real time, amrex::Real dt_radiation, int const /*iter_count*/,
							       int const /*nsubsteps*/, amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
							       amrex::MultiFab const &reducedSpeedOfLightFactor)
{
	// get cell sizes
	auto const &dx = geom[lev].CellSizeArray();

	// update ghost zones [old timestep]
	fillBoundaryConditions(state_old_cc_[lev], state_old_cc_[lev], lev, time, quokka::centering::cc, quokka::direction::na, PreInterpState,
			       PostInterpState);

	// advance all grids on local processor (Stage 1 of integrator)
	for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &reducedSpeedOfLightFactor_ = reducedSpeedOfLightFactor.const_array(iter);
		auto const &stateOld = state_old_cc_[lev].const_array(iter);
		auto const &stateNew = state_new_cc_[lev].array(iter);
		auto [fluxArrays, fluxDiffusiveArrays] = computeRadiationFluxes(stateOld, indexRange, ncompHyperbolic_, dx, reducedSpeedOfLightFactor_);

		// Stage 1 of RK2-SSP
		RadSystem<problem_t>::PredictStep(
		    stateOld, stateNew, {AMREX_D_DECL(fluxArrays[0].array(), fluxArrays[1].array(), fluxArrays[2].array())},
		    {AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(), fluxDiffusiveArrays[1].const_array(), fluxDiffusiveArrays[2].const_array())},
		    dt_radiation, dx, indexRange, ncompHyperbolic_);

		if (do_reflux) {
			// increment flux registers
			// WARNING: as written, diffusive flux correction is not compatible with reflux!!
			auto expandedFluxes = expandFluxArrays(fluxArrays, nstartHyperbolic_, state_new_cc_[lev].nComp());
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, expandedFluxes, lev, 0.5 * dt_radiation);
		}
	}
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::advanceRadiationMidpointRK2(int lev, amrex::Real time, amrex::Real dt_radiation, int const /*iter_count*/,
							      int const /*nsubsteps*/, amrex::YAFluxRegister *fr_as_crse, amrex::YAFluxRegister *fr_as_fine,
							      amrex::MultiFab const &reducedSpeedOfLightFactor)
{
	auto const &dx = geom[lev].CellSizeArray();

	// update ghost zones [intermediate stage stored in state_new_cc_]
	fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], lev, (time + dt_radiation), quokka::centering::cc, quokka::direction::na, PreInterpState,
			       PostInterpState);

	// advance all grids on local processor (Stage 2 of integrator)
	for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &reducedSpeedOfLightFactor_ = reducedSpeedOfLightFactor.const_array(iter);
		auto const &stateOld = state_old_cc_[lev].const_array(iter);
		auto const &stateInter = state_new_cc_[lev].const_array(iter);
		auto const &stateNew = state_new_cc_[lev].array(iter);
		auto [fluxArraysOld, fluxDiffusiveArraysOld] = computeRadiationFluxes(stateOld, indexRange, ncompHyperbolic_, dx, reducedSpeedOfLightFactor_);
		auto [fluxArrays, fluxDiffusiveArrays] = computeRadiationFluxes(stateInter, indexRange, ncompHyperbolic_, dx, reducedSpeedOfLightFactor_);

		// Stage 2 of RK2-SSP
		RadSystem<problem_t>::AddFluxesRK2(
		    stateNew, stateOld, stateInter, {AMREX_D_DECL(fluxArraysOld[0].array(), fluxArraysOld[1].array(), fluxArraysOld[2].array())},
		    {AMREX_D_DECL(fluxArrays[0].array(), fluxArrays[1].array(), fluxArrays[2].array())},
		    {AMREX_D_DECL(fluxDiffusiveArraysOld[0].const_array(), fluxDiffusiveArraysOld[1].const_array(), fluxDiffusiveArraysOld[2].const_array())},
		    {AMREX_D_DECL(fluxDiffusiveArrays[0].const_array(), fluxDiffusiveArrays[1].const_array(), fluxDiffusiveArrays[2].const_array())},
		    dt_radiation, dx, indexRange, ncompHyperbolic_);

		if (do_reflux) {
			// increment flux registers
			// WARNING: as written, diffusive flux correction is not compatible with reflux!!
			auto expandedFluxes = expandFluxArrays(fluxArrays, nstartHyperbolic_, state_new_cc_[lev].nComp());
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, expandedFluxes, lev, 0.5 * dt_radiation);
		}
	}
}

template <typename problem_t>
void QuokkaSimulation<problem_t>::operatorSplitSourceTerms(amrex::Array4<amrex::Real> const &stateNew, const amrex::Box &indexRange, const amrex::Real time,
							   const double dt, const int stage, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
							   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
							   amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor, int *p_iteration_counter,
							   int *p_iteration_failure_counter)
{
	amrex::FArrayBox radEnergySource(indexRange, Physics_Traits<problem_t>::nGroups,
					 amrex::The_Async_Arena()); // cell-centered scalar

	radEnergySource.setVal<amrex::RunOn::Device>(0.);

	// cell-centered radiation energy source
	RadSystem<problem_t>::SetRadEnergySource(radEnergySource.array(), indexRange, dx, prob_lo, prob_hi, time + dt);

	// cell-centered source terms
	if constexpr (Physics_Traits<problem_t>::nGroups <= 1) {
		RadSystem<problem_t>::AddSourceTermsSingleGroup(stateNew, radEnergySource.const_array(), indexRange, dt, stage, dustGasInteractionCoeff_,
								reducedSpeedOfLightFactor, p_iteration_counter, p_iteration_failure_counter);
	} else {
		RadSystem<problem_t>::AddSourceTermsMultiGroup(stateNew, radEnergySource.const_array(), indexRange, dt, stage, dustGasInteractionCoeff_,
							       reducedSpeedOfLightFactor, p_iteration_counter, p_iteration_failure_counter);
	}
}

template <typename problem_t>
auto QuokkaSimulation<problem_t>::computeRadiationFluxes(amrex::Array4<const amrex::Real> const &consVar, const amrex::Box &indexRange, const int nvars,
							 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
							 amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor)
    -> std::tuple<std::array<amrex::FArrayBox, AMREX_SPACEDIM>, std::array<amrex::FArrayBox, AMREX_SPACEDIM>>
{
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in x
	amrex::FArrayBox x1FluxDiffusive(x1FluxRange, nvars, amrex::The_Async_Arena());
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in y
	amrex::FArrayBox x2FluxDiffusive(x2FluxRange, nvars, amrex::The_Async_Arena());
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars, amrex::The_Async_Arena()); // node-centered in z
	amrex::FArrayBox x3FluxDiffusive(x3FluxRange, nvars, amrex::The_Async_Arena());
#endif

	AMREX_D_TERM(fluxFunction<FluxDir::X1>(consVar, x1Flux, x1FluxDiffusive, indexRange, nvars, dx, reducedSpeedOfLightFactor);
		     , fluxFunction<FluxDir::X2>(consVar, x2Flux, x2FluxDiffusive, indexRange, nvars, dx, reducedSpeedOfLightFactor);
		     , fluxFunction<FluxDir::X3>(consVar, x3Flux, x3FluxDiffusive, indexRange, nvars, dx, reducedSpeedOfLightFactor);)

	std::array<amrex::FArrayBox, AMREX_SPACEDIM> fluxArrays = {AMREX_D_DECL(std::move(x1Flux), std::move(x2Flux), std::move(x3Flux))};
	std::array<amrex::FArrayBox, AMREX_SPACEDIM> fluxDiffusiveArrays{
	    AMREX_D_DECL(std::move(x1FluxDiffusive), std::move(x2FluxDiffusive), std::move(x3FluxDiffusive))};

	return std::make_tuple(std::move(fluxArrays), std::move(fluxDiffusiveArrays));
}

template <typename problem_t>
template <FluxDir DIR>
void QuokkaSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState, amrex::FArrayBox &x1Flux, amrex::FArrayBox &x1FluxDiffusive,
					       const amrex::Box &indexRange, const int nvars, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
					       amrex::Array4<const amrex::Real> const &reducedSpeedOfLightFactor)
{
	int dir = 0;
	if constexpr (DIR == FluxDir::X1) {
		dir = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		dir = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		dir = 2;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_cc_);
	// N.B.: A one-zone layer around the cells must be fully reconstructed in order for PPM to
	// work.
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dir);

	amrex::FArrayBox primVar(ghostRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	RadSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(), ghostRange);

	if (radiationReconstructionOrder_ == 3) {
		// mixed interface/cell-centered kernel
		HyperbolicSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
										x1ReconstructRange, nvars);
	} else if (radiationReconstructionOrder_ == 2) {
		// PLM and donor cell are interface-centered kernels
		HyperbolicSystem<problem_t>::template ReconstructStatesPLM<DIR, SlopeLimiter::MC>(primVar.array(), x1LeftState.array(), x1RightState.array(),
												  x1ReconstructRange, nvars);
	} else if (radiationReconstructionOrder_ == 1) {
		HyperbolicSystem<problem_t>::template ReconstructStatesConstant<DIR>(primVar.array(), x1LeftState.array(), x1RightState.array(),
										     x1ReconstructRange, nvars);
	} else {
		amrex::Abort("Invalid reconstruction order for radiation variables! Aborting...");
	}

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dir);
	RadSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux.array(), x1FluxDiffusive.array(), x1LeftState.array(), x1RightState.array(), x1FluxRange,
							  consState, dx, use_wavespeed_correction_,
							  reducedSpeedOfLightFactor); // watch out for argument order!!
}

#endif // RADIATION_SIMULATION_HPP_
