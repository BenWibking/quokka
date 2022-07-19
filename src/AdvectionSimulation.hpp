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

#include "AMReX.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_Config.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_TagBox.H"
#include "AMReX_Utility.H"
#include "AMReX_YAFluxRegister.H"
#include <AMReX_FluxRegister.H>

#include "ArrayView.hpp"
#include "linear_advection.hpp"
#include "simulation.hpp"

// Simulation class should be initialized only once per program (i.e., is a singleton)
template <typename problem_t> class AdvectionSimulation : public AMRSimulation<problem_t>
{
      public:
	using AMRSimulation<problem_t>::state_old_;
	using AMRSimulation<problem_t>::state_new_;
	using AMRSimulation<problem_t>::max_signal_speed_;

	using AMRSimulation<problem_t>::cflNumber_;
	using AMRSimulation<problem_t>::dt_;
	using AMRSimulation<problem_t>::ncomp_;
	using AMRSimulation<problem_t>::nghost_;
	using AMRSimulation<problem_t>::cycleCount_;
	using AMRSimulation<problem_t>::areInitialConditionsDefined_;
	using AMRSimulation<problem_t>::componentNames_;

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

	AdvectionSimulation(amrex::IntVect & /*gridDims*/, amrex::RealBox & /*boxSize*/,
			    amrex::Vector<amrex::BCRec> &boundaryConditions, const int ncomp = 1)
	    : AMRSimulation<problem_t>(boundaryConditions, ncomp)
	{
		componentNames_ = {"density"};
	}

	explicit AdvectionSimulation(amrex::Vector<amrex::BCRec> &boundaryConditions, const int ncomp = 1)
	    : AMRSimulation<problem_t>(boundaryConditions, ncomp)
	{
		componentNames_ = {"density"};
	}

	void computeMaxSignalLocal(int level) override;
	void setInitialConditionsAtLevel(int level) override;
	void advanceSingleTimestepAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
					  				  int /*ncycle*/) override;
	void computeAfterTimestep() override;
	void computeAfterEvolve(amrex::Vector<amrex::Real> &initSumCons) override;
	void computeReferenceSolution(
	    amrex::MultiFab &ref,
    	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi);

	// compute derived variables
	void ComputeDerivedVar(int lev, std::string const &dname, amrex::MultiFab &mf, int ncomp) const override;

	void FixupState(int lev) override;

	// tag cells for refinement
	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;

	auto computeFluxes(amrex::Array4<const amrex::Real> const &consVar,
			   const amrex::Box &indexRange, int nvars)
	    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>;

	template <FluxDir DIR>
	void fluxFunction(amrex::Array4<const amrex::Real> const &consState,
			  amrex::Array4<amrex::Real> const &x1Flux, const amrex::Box &indexRange,
			  int nvars);

	double advectionVx_ = 1.0; // default
	double advectionVy_ = 0.0; // default
	double advectionVz_ = 0.0; // default

	amrex::Real errorNorm_ = NAN;

	static constexpr int reconstructOrder_ =
	    3; // PPM = 3 ['third order'], piecewise constant == 1
	static constexpr int integratorOrder_ = 3; // RK3-SSP = 3, RK2-SSP = 2, forward Euler = 1
};

template <typename problem_t>
void AdvectionSimulation<problem_t>::computeMaxSignalLocal(int const level)
{
	// loop over local grids, compute CFL timestep
	for (amrex::MFIter iter(state_new_[level]); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_[level].const_array(iter);
		auto const &maxSignal = max_signal_speed_[level].array(iter);
		LinearAdvectionSystem<problem_t>::ComputeMaxSignalSpeed(
		    stateOld, maxSignal, advectionVx_, advectionVy_, advectionVz_, indexRange);
	}
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::setInitialConditionsAtLevel(int level)
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
void AdvectionSimulation<problem_t>::ErrorEst(int lev, amrex::TagBoxArray &tags,
					      amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement -- implement in problem generator
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::FixupState(int lev)
{
	// fix negative states
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::computeReferenceSolution(
    amrex::MultiFab &ref,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi)
{
	// user implemented
}

template <typename problem_t>
void AdvectionSimulation<problem_t>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	// compute reference solution
	const int ncomp = state_new_[0].nComp();
	const int nghost = state_new_[0].nGrow();
	amrex::MultiFab state_ref_level0(boxArray(0), DistributionMap(0), ncomp, nghost);
	computeReferenceSolution(state_ref_level0, geom[0].CellSizeArray(),
							 geom[0].ProbLoArray(), geom[0].ProbHiArray());

	// compute error norm
	amrex::MultiFab residual(boxArray(0), DistributionMap(0), ncomp, nghost);
	amrex::MultiFab::Copy(residual, state_ref_level0, 0, 0, ncomp, nghost);
	amrex::MultiFab::Saxpy(residual, -1., state_new_[0], 0, 0, ncomp, nghost);

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

template <typename problem_t>
void AdvectionSimulation<problem_t>::advanceSingleTimestepAtLevel(int lev, amrex::Real time,
								  amrex::Real dt_lev, int /*ncycle*/)
{
	// based on amrex/Tests/EB/CNS/Source/CNS_advance.cpp

	// since we are starting a new timestep, need to swap old and new states on this
	// level
	std::swap(state_old_[lev], state_new_[lev]);

	// get geometry (used only for cell sizes)
	auto const &geomLevel = geom[lev];

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

	// set flux register scale factor for each stage
	amrex::Vector<amrex::Real> fluxScaleFactor{};

	if (integratorOrder_ == 3) {
		fluxScaleFactor = {1./6., 1./6., 2./3.};
	} else if (integratorOrder_ == 2) {
		fluxScaleFactor = {0.5, 0.5};
	}

	// We use the RK3-SSP integrator in a method-of-lines framework.
	// Note that we cannot re-use multifabs for intermediate and final stages
	// if first-order flux correction or inner-outer updates are used.

	// intermediate stage multifabs
	amrex::MultiFab state_U1(grids[lev], dmap[lev], ncomp_, nghost_);
	amrex::MultiFab state_U2(grids[lev], dmap[lev], ncomp_, nghost_);

	// RK2/3-SSP stage 1

	// update ghost zones [w/ old timestep]
	fillBoundaryConditions(state_old_[lev], state_old_[lev], lev, time);

	// check state validity
	AMREX_ASSERT(!state_old_[lev].contains_nan(0, state_old_[lev].nComp()));
	AMREX_ASSERT(!state_old_[lev].contains_nan()); // check ghost cells

	for (amrex::MFIter iter(state_U1); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateOld = state_old_[lev].const_array(iter);
		auto const &stateNew = state_U1.array(iter);
		auto fluxArrays = computeFluxes(stateOld, indexRange, ncomp_);

		LinearAdvectionSystem<problem_t>::PredictStep(
		    stateOld, stateNew,
		    {AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
				  fluxArrays[2].const_array())},
		    dt_lev, geomLevel.CellSizeArray(), indexRange, ncomp_);

		if (do_reflux) {
			incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, fluxArrays, lev,
					       fluxScaleFactor[0] * dt_lev);
		}
	}

	// update ghost zones [w/ intermediate stage stored in state_U1]
	fillBoundaryConditions(state_U1, state_U1, lev, time + dt_lev);

	// check state validity
	AMREX_ASSERT(!state_U1.contains_nan(0, state_U1.nComp()));
	AMREX_ASSERT(!state_U1.contains_nan()); // check ghost cells

	if (integratorOrder_ == 2) {
		// RK2-SSP stage 2

		for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &stateInOld = state_old_[lev].const_array(iter);
			auto const &stateInStar = state_U1.const_array(iter);
			auto const &stateOut = state_new_[lev].array(iter);
			auto fluxArrays = computeFluxes(stateInStar, indexRange, ncomp_);

			LinearAdvectionSystem<problem_t>::AddFluxesRK2(
				stateOut, stateInOld, stateInStar,
				{AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
					fluxArrays[2].const_array())},
				dt_lev, geomLevel.CellSizeArray(), indexRange, ncomp_);

			if (do_reflux) {
				incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, fluxArrays,
							lev, fluxScaleFactor[1] * dt_lev);
			}
		}
	} else if (integratorOrder_ == 3) {
		// RK3-SSP stage 2

		for (amrex::MFIter iter(state_U2); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &stateInOld = state_old_[lev].const_array(iter);
			auto const &stateInStar = state_U1.const_array(iter);
			auto const &stateOut = state_U2.array(iter);
			auto fluxArrays = computeFluxes(stateInStar, indexRange, ncomp_);

			LinearAdvectionSystem<problem_t>::RK3_Stage2(
				stateOut, stateInOld, stateInStar,
				{AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
					fluxArrays[2].const_array())},
				dt_lev, geomLevel.CellSizeArray(), indexRange, ncomp_);

			if (do_reflux) {
				incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, fluxArrays,
							lev, fluxScaleFactor[1] * dt_lev);
			}
		}

		// RK3-SSP stage 3

		// update ghost zones [w/ intermediate stage stored in state_U2]
		fillBoundaryConditions(state_U2, state_U2, lev, time + 0.5 * dt_lev);

		// check state validity
		AMREX_ASSERT(!state_U2.contains_nan(0, state_U2.nComp()));
		AMREX_ASSERT(!state_U2.contains_nan()); // check ghost cells

		for (amrex::MFIter iter(state_new_[lev]); iter.isValid(); ++iter) {
			const amrex::Box &indexRange = iter.validbox();
			auto const &stateOld = state_old_[lev].const_array(iter);
			auto const &stateInU1 = state_U1.const_array(iter);
			auto const &stateInStar = state_U2.const_array(iter);
			auto const &stateOut = state_new_[lev].array(iter);
			auto fluxArrays = computeFluxes(stateInStar, indexRange, ncomp_);

			LinearAdvectionSystem<problem_t>::RK3_Stage3(
				stateOut, stateOld, stateInU1, stateInStar,
				{AMREX_D_DECL(fluxArrays[0].const_array(), fluxArrays[1].const_array(),
					fluxArrays[2].const_array())},
				dt_lev, geomLevel.CellSizeArray(), indexRange, ncomp_);

			if (do_reflux) {
				incrementFluxRegisters(iter, fr_as_crse, fr_as_fine, fluxArrays,
							lev, fluxScaleFactor[2] * dt_lev);
			}
		}
	}
}

template <typename problem_t>
auto AdvectionSimulation<problem_t>::computeFluxes(amrex::Array4<const amrex::Real> const &consVar,
						   const amrex::Box &indexRange, const int nvars)
    -> std::array<amrex::FArrayBox, AMREX_SPACEDIM>
{
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, 0);
	amrex::FArrayBox x1Flux(x1FluxRange, nvars,
				amrex::The_Async_Arena()); // node-centered in x
#if (AMREX_SPACEDIM >= 2)
	amrex::Box const &x2FluxRange = amrex::surroundingNodes(indexRange, 1);
	amrex::FArrayBox x2Flux(x2FluxRange, nvars,
				amrex::The_Async_Arena()); // node-centered in y
#endif
#if (AMREX_SPACEDIM == 3)
	amrex::Box const &x3FluxRange = amrex::surroundingNodes(indexRange, 2);
	amrex::FArrayBox x3Flux(x3FluxRange, nvars,
				amrex::The_Async_Arena()); // node-centered in z
#endif

	AMREX_D_TERM(fluxFunction<FluxDir::X1>(consVar, x1Flux.array(), indexRange, nvars);
		     , fluxFunction<FluxDir::X2>(consVar, x2Flux.array(), indexRange, nvars);
		     , fluxFunction<FluxDir::X3>(consVar, x3Flux.array(), indexRange, nvars);)

	return {AMREX_D_DECL(std::move(x1Flux), std::move(x2Flux), std::move(x3Flux))};
}

template <typename problem_t>
template <FluxDir DIR>
void AdvectionSimulation<problem_t>::fluxFunction(amrex::Array4<const amrex::Real> const &consState,
						  amrex::Array4<amrex::Real> const &x1Flux,
						  const amrex::Box &indexRange, const int nvars)
{
	amrex::Real advectionVel = NAN;
	int dim = 0;
	if constexpr (DIR == FluxDir::X1) {
		advectionVel = advectionVx_;
		// [0 == x1 direction]
		dim = 0;
	} else if constexpr (DIR == FluxDir::X2) {
		advectionVel = advectionVy_;
		// [1 == x2 direction]
		dim = 1;
	} else if constexpr (DIR == FluxDir::X3) {
		advectionVel = advectionVz_;
		// [2 == x3 direction]
		dim = 2;
	}

	// extend box to include ghost zones
	amrex::Box const &ghostRange = amrex::grow(indexRange, nghost_);
	amrex::Box const &reconstructRange = amrex::grow(indexRange, 1);
	amrex::Box const &x1ReconstructRange = amrex::surroundingNodes(reconstructRange, dim);
	amrex::FArrayBox primVar(ghostRange, nvars,
				 amrex::The_Async_Arena()); // cell-centered
	amrex::FArrayBox x1LeftState(x1ReconstructRange, nvars, amrex::The_Async_Arena());
	amrex::FArrayBox x1RightState(x1ReconstructRange, nvars, amrex::The_Async_Arena());

	// cell-centered kernel
	LinearAdvectionSystem<problem_t>::ConservedToPrimitive(consState, primVar.array(),
							       ghostRange, nvars);

	if constexpr (reconstructOrder_ == 3) {
		// mixed interface/cell-centered kernel
		LinearAdvectionSystem<problem_t>::template ReconstructStatesPPM<DIR>(
		    primVar.array(), x1LeftState.array(), x1RightState.array(), reconstructRange,
		    x1ReconstructRange, nvars);
	} else if constexpr (reconstructOrder_ == 1) {
		// interface-centered kernel
		LinearAdvectionSystem<problem_t>::template ReconstructStatesConstant<DIR>(
		    primVar.array(), x1LeftState.array(), x1RightState.array(), x1ReconstructRange,
		    nvars);
	}

	// interface-centered kernel
	amrex::Box const &x1FluxRange = amrex::surroundingNodes(indexRange, dim);

	LinearAdvectionSystem<problem_t>::template ComputeFluxes<DIR>(
	    x1Flux, x1LeftState.array(), x1RightState.array(), advectionVel, x1FluxRange, nvars);
}

#endif // ADVECTION_SIMULATION_HPP_