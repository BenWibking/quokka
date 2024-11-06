#ifndef ADVECTION_SIMULATION_HPP_ // NOLINT
#define ADVECTION_SIMULATION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file AdvectionSimulation.hpp
/// \brief Implements classes and functions to organise the overall setup,
/// timestepping, solving, and I/O of a simulation for linear advection.

#include <array>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TagBox.H"
#include "AMReX_YAFluxRegister.H"
#include <AMReX_FluxRegister.H>

#include "linear_advection/linear_advection.hpp"
#include "simulation.hpp"
#include "util/ArrayView.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class AdvectionSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_cc_;
	using AMRSimulation<problem_t>::state_new_cc_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::dt_;
	using AMRSimulation<problem_t>::BCs_cc_;
	using AMRSimulation<problem_t>::nghost_cc_;
	using AMRSimulation<problem_t>::cycleCount_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::componentNames_cc_;

	using AMRSimulation<problem_t>::fillBoundaryConditions;
	using AMRSimulation<problem_t>::geom;
	using AMRSimulation<problem_t>::grids;
	using AMRSimulation<problem_t>::dmap;
	using AMRSimulation<problem_t>::refRatio;
	using AMRSimulation<problem_t>::flux_reg_;
	using AMRSimulation<problem_t>::do_reflux;
	using AMRSimulation<problem_t>::incrementFluxRegisters;
	using AMRSimulation<problem_t>::finest_level;
	using AMRSimulation<problem_t>::finestLevel;
	using AMRSimulation<problem_t>::tOld_;
	using AMRSimulation<problem_t>::tNew_;
	using AMRSimulation<problem_t>::boxArray;
	using AMRSimulation<problem_t>::DistributionMap;

	explicit AdvectionSimulation(amrex::Vector<amrex::BCRec> &BCs_cc) : AMRSimulation<problem_t>(BCs_cc) { componentNames_cc_.push_back({"density"}); }

	void computeMaxSignalLocal(int level) override;
	auto computeExtraPhysicsTimestep(int level) -> amrex::Real override;
	void preCalculateInitialConditions() override;
	void setInitialConditionsOnGrid(quokka::grid const &grid_elem) override;
	void setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem) override;
	void createInitialParticles() override;
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int /*ncycle*/) override;
	void computeBeforeTimestep() override;
	void computeAfterTimestep() override;
	void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;
	void computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
				      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi);
	void fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev) override;
	void applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt) override;

	// compute derived variables
	void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;
	// compute projected vars
	[[nodiscard]] auto ComputeProjections(int dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> override;

	// compute statistics
	auto ComputeStatistics() -> std::map<std::string, amrex::Real> override;

	void FixupState(int lev) override;

	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	auto computeFluxes(amrex::MultiFab const &consVar, int nvars, int lev) -> std::array<amrex::MultiFab, AMREX_SPACEDIM>;

	template <FluxDir DIR>
	void fluxFunction(amrex::MultiFab const &consState, amrex::MultiFab &primVar, amrex::MultiFab &x1Flux, amrex::MultiFab &x1LeftState,
			  amrex::MultiFab &x1RightState, int ng_reconstruct, int nvars);

	double advectionVx_ = 1.0; // default
	double advectionVy_ = 0.0; // default
	double advectionVz_ = 0.0; // default

	amrex::Real errorNorm_ = NAN;

	static constexpr int reconstructOrder_ = 3; // PPM = 3 ['third order'], piecewise constant == 1
	static constexpr int integratorOrder_ = 2;  // RK2-SSP = 2, forward Euler = 1
};

template <typename problem_t> void AdvectionSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	// loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_cc_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_cc_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);
		LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(stateOld, maxSignal, advectionVx_, advectionVy_, advectionVz_, indexRange);
	}
}

template <typename problem_t> void AdvectionSimulation<problem_t>::fillPoissonRhsAtLevel(amrex::MultiFab &rhs, int lev)
{
	// deliberately empty
}

template <typename problem_t> void AdvectionSimulation<problem_t>::applyPoissonGravityAtLevel(amrex::MultiFab const &phi, int lev, amrex::Real dt)
{
	// deliberately empty
}

template <typename problem_t> auto AdvectionSimulation<problem_t>::computeExtraPhysicsTimestep(int const /*level*/) -> amrex::Real
{
	// user can override this
	return std::numeric_limits<amrex::Real>::max();
}

template <typename problem_t> void AdvectionSimulation<problem_t>::preCalculateInitialConditions()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if face-centered vars are used
}

template <typename problem_t> void AdvectionSimulation<problem_t>::createInitialParticles()
{
	// default empty implementation
	// user should implement using problem-specific template specialization
	// note: an implementation is only required if particles are used
}

template <typename problem_t> void AdvectionSimulation<problem_t>::computeBeforeTimestep()
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::computeAfterTimestep()
{
	// do nothing -- user should implement using problem-specific template specialization
}

template <typename problem_t> void AdvectionSimulation<problem_t>::ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const
{
	// user should implement
}

template <typename problem_t>
auto AdvectionSimulation<problem_t>::ComputeProjections(int /*dir*/) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
{
	// user should implement
	return std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>{};
}

template <typename problem_t> auto AdvectionSimulation<problem_t>::ComputeStatistics() -> std::map<std::string, amrex::Real>
{
	// user should implement
	return std::map<std::string, amrex::Real>{};
}

template <typename problem_t> void AdvectionSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement -- implement in problem generator
}

template <typename problem_t> void AdvectionSimulation<problem_t>::FixupState(int lev)
{
	// fix negative states
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::computeReferenceSolution(amrex::MultiFab &ref, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
							      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
							      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
{
	// user implemented
}

template <typename problem_t> void AdvectionSimulation<problem_t>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	// compute reference solution
	const int ncomp = state_new_cc_[0].nComp();
	amrex::MultiFab state_ref_level0(boxArray(0), DistributionMap(0), ncomp, 0);
	computeReferenceSolution(state_ref_level0, geom[0].CellSizeArray(), geom[0].ProbLoArray(), geom[0].ProbHiArray());

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

	amrex::Print() << "\nRelative rms L1 error norm = " << rel_error << "\n\n";
}

template <typename problem_t> void AdvectionSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev, int /*ncycle*/)
{
	// based on amrex/Tests/EB/CNS/Source/CNS_advance.cpp

	// since we are starting a new timestep, need to swap old and new states on this
	// level
	std::swap(state_old_cc_[lev], state_new_cc_[lev]);

	// check state validity
	AMREX_ASSERT(!state_old_cc_[lev].contains_nan(0, state_old_cc_[lev].nComp()));
	AMREX_ASSERT(!state_old_cc_[lev].contains_nan()); // check ghost cells

	// get geometry (used only for cell sizes)
	auto const &geomLevel = geom[lev];

#ifdef USE_YAFLUXREGISTER
	// get flux registers
	amrex::YAFluxRegister *fr_as_crse = nullptr;
	amrex::YAFluxRegister *fr_as_fine = nullptr;

	if (do_reflux) {
		if (lev < finestLevel()) {
			fr_as_crse = flux_reg_[lev + 1].get();
			fr_as_crse->reset();
		}
		if (lev > 0) {
			fr_as_fine = flux_reg_[lev].get();
		}
	}
#else
	amrex::FluxRegister *fine = nullptr;
	amrex::FluxRegister *current = nullptr;

	if (do_reflux && lev < finest_level) {
		fine = flux_reg_[lev + 1].get();
		fine->setVal(0.0);
	}

	if (do_reflux && lev > 0) {
		current = flux_reg_[lev].get();
	}

	// create temporary MultiFab to store the fluxes from each grid on this level
	std::array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

	if (do_reflux) {
		for (int j = 0; j < AMREX_SPACEDIM; j++) {
			amrex::BoxArray ba = state_new_cc_[lev].boxArray();
			ba.surroundingNodes(j);
			fluxes[j].define(ba, dmap[lev], Physics_Indices<problem_t>::nvarTotal_cc, 0);
			fluxes[j].setVal(0.);
		}
	}
#endif // USE_YAFLUXREGISTER

	// We use the RK2-SSP integrator in a method-of-lines framework. It needs 2
	// registers: one to store the old timestep, and one to store the intermediate stage
	// and final stage. The intermediate stage and final stage reuse the same register.

	// update ghost zones [w/ old timestep]
	// (N.B. the input and output multifabs are allowed to be the same, as done here)
	fillBoundaryConditions(state_old_cc_[lev], state_old_cc_[lev], lev, time, quokka::centering::cc, quokka::direction::na,
			       AMRSimulation<problem_t>::InterpHookNone, AMRSimulation<problem_t>::InterpHookNone);

	amrex::Real fluxScaleFactor = NAN;
	if constexpr (integratorOrder_ == 2) {
		fluxScaleFactor = 0.5;
	} else if constexpr (integratorOrder_ == 1) {
		fluxScaleFactor = 1.0;
	}

	// advance all grids on local processor (Stage 1 of integrator)
	{
		auto const &stateOld = state_old_cc_[lev];
		auto &stateNew = state_new_cc_[lev];
		auto fluxArrays = computeFluxes(stateOld, Physics_Indices<problem_t>::nvarTotal_cc, lev);

		// Stage 1 of RK2-SSP
		LinearAdvectionSystem<problem_t>::PredictStep(stateOld, stateNew, fluxArrays, dt_lev, geomLevel.CellSizeArray(),
							      Physics_Indices<problem_t>::nvarTotal_cc);

		if (do_reflux) {
#ifdef USE_YAFLUXREGISTER
			// increment flux registers
			incrementFluxRegisters(fr_as_crse, fr_as_fine, fluxArrays, lev, fluxScaleFactor * dt_lev);
#else
			for (int i = 0; i < AMREX_SPACEDIM; i++) {
				fluxes[i][iter].plus<amrex::RunOn::Gpu>(fluxArrays[i]);
			}
#endif // USE_YAFLUXREGISTER
		}
	}

	if constexpr (integratorOrder_ == 2) {
		// update ghost zones [w/ intermediate stage stored in state_new_cc_]
		fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], lev, (time + dt_lev), quokka::centering::cc, quokka::direction::na,
				       AMRSimulation<problem_t>::InterpHookNone, AMRSimulation<problem_t>::InterpHookNone);

		// advance all grids on local processor (Stage 2 of integrator)
		{
			auto const &stateInOld = state_old_cc_[lev];
			auto const &stateInStar = state_new_cc_[lev];
			auto &stateOut = state_new_cc_[lev];
			auto fluxArrays = computeFluxes(stateInStar, Physics_Indices<problem_t>::nvarTotal_cc, lev);

			// Stage 2 of RK2-SSP
			LinearAdvectionSystem<problem_t>::AddFluxesRK2(stateOut, stateInOld, stateInStar, fluxArrays, dt_lev, geomLevel.CellSizeArray(),
								       Physics_Indices<problem_t>::nvarTotal_cc);

			if (do_reflux) {
#ifdef USE_YAFLUXREGISTER
				// increment flux registers
				incrementFluxRegisters(fr_as_crse, fr_as_fine, fluxArrays, lev, fluxScaleFactor * dt_lev);
#else
				for (int i = 0; i < AMREX_SPACEDIM; i++) {
					fluxes[i][iter].plus<amrex::RunOn::Gpu>(fluxArrays[i]);
				}
#endif // USE_YAFLUXREGISTER
			}
		}
	}

#ifndef USE_YAFLUXREGISTER
	if (do_reflux) {
		// rescale by face area
		auto dx = geomLevel.CellSizeArray();
		amrex::Real const cell_vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

		for (int i = 0; i < AMREX_SPACEDIM; i++) {
			amrex::Real const face_area = cell_vol / dx[i];
			amrex::Real const rescaleFactor = fluxScaleFactor * dt_lev * face_area;
			fluxes[i].mult(rescaleFactor);
		}

		if (current != nullptr) {
			for (int i = 0; i < AMREX_SPACEDIM; i++) {
				current->FineAdd(fluxes[i], i, 0, 0, Physics_Indices<problem_t>::nvarTotal_cc, 1.);
			}
		}

		if (fine != nullptr) {
			for (int i = 0; i < AMREX_SPACEDIM; i++) {
				fine->CrseInit(fluxes[i], i, 0, 0, Physics_Indices<problem_t>::nvarTotal_cc, -1.);
			}
		}
	}
#endif
}

template <typename problem_t>
auto AdvectionSimulation<problem_t>::computeFluxes(amrex::MultiFab const &consVar, const int nvars, const int lev)
    -> std::array<amrex::MultiFab, AMREX_SPACEDIM>
{
	auto ba = grids[lev];
	auto dm = dmap[lev];
	const int reconstructRange = 1;

	// allocate temporary MultiFabs
	amrex::MultiFab primVar(ba, dm, nvars, nghost_cc_);
	std::array<amrex::MultiFab, AMREX_SPACEDIM> flux;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> leftState;
	std::array<amrex::MultiFab, AMREX_SPACEDIM> rightState;

	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		auto ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
		leftState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		rightState[idim] = amrex::MultiFab(ba_face, dm, nvars, reconstructRange);
		flux[idim] = amrex::MultiFab(ba_face, dm, nvars, 0);
	}

	AMREX_D_TERM(fluxFunction<FluxDir::X1>(consVar, primVar, flux[0], leftState[0], rightState[0], reconstructRange, nvars);
		     , fluxFunction<FluxDir::X2>(consVar, primVar, flux[1], leftState[1], rightState[1], reconstructRange, nvars);
		     , fluxFunction<FluxDir::X3>(consVar, primVar, flux[2], leftState[2], rightState[2], reconstructRange, nvars);)

	// synchronization point to prevent MultiFabs from going out of scope
	amrex::Gpu::streamSynchronizeAll();
	return flux;
}

template <typename problem_t>
template <FluxDir DIR>
void AdvectionSimulation<problem_t>::fluxFunction(amrex::MultiFab const &consState, amrex::MultiFab &primVar, amrex::MultiFab &x1Flux,
						  amrex::MultiFab &x1LeftState, amrex::MultiFab &x1RightState, const int ng_reconstruct, const int nvars)
{
	amrex::Real advectionVel = NAN;
	if constexpr (DIR == FluxDir::X1) {
		advectionVel = advectionVx_;
	} else if constexpr (DIR == FluxDir::X2) {
		advectionVel = advectionVy_;
	} else if constexpr (DIR == FluxDir::X3) {
		advectionVel = advectionVz_;
	}

	// amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	// amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dim);

	LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consState, primVar, nghost_cc_, nvars);

	LinearAdvectionSystem<problem_t>::template ReconstructStatesPPM<DIR>(primVar, x1LeftState, x1RightState, ng_reconstruct, nvars);

	LinearAdvectionSystem<problem_t>::template ComputeFluxes<DIR>(x1Flux, x1LeftState, x1RightState, advectionVel, nvars);
}

#endif // ADVECTION_SIMULATION_HPP_
