#ifndef PROJECTION_HPP_ // NOLINT
#define PROJECTION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
//! \file projection.hpp
///  \brief AMReX I/O for 2D projections

#include <string>

// AMReX headers
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Orientation.H"
#include <AMReX.H>

namespace quokka::diagnostics
{

namespace detail
{
auto direction_to_string(const amrex::Direction dir) -> std::string;
} // namespace detail

template <typename ReduceOp, typename F>
auto ComputePlaneProjection(amrex::Vector<amrex::MultiFab> const &state_new, const int finest_level, amrex::Vector<amrex::Geometry> const &geom,
			    amrex::Vector<amrex::IntVect> const &ref_ratio, const amrex::Direction dir, F const &user_f) -> amrex::BaseFab<amrex::Real>
{
	// compute plane-parallel projection of user_f(i, j, k, state) along the given axis.
	BL_PROFILE("quokka::DiagProjection::computePlaneProjection()");

	// allocate temporary multifabs
	amrex::Vector<amrex::MultiFab> q;
	q.resize(finest_level + 1);

	for (int lev = 0; lev <= finest_level; ++lev) {
		q[lev].define(state_new[lev].boxArray(), state_new[lev].DistributionMap(), 1, 0);
	}

	// evaluate user_f on all levels
	for (int lev = 0; lev <= finest_level; ++lev) {
		auto const &state = state_new[lev].const_arrays();
		auto const &result = q[lev].arrays();
		amrex::ParallelFor(q[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) { result[bx](i, j, k) = user_f(i, j, k, state[bx]); });
	}
	amrex::Gpu::streamSynchronize();

	// average down
	for (int lev = finest_level; lev < 0; --lev) {
		amrex::average_down(q[lev], q[lev - 1], geom[lev], geom[lev - 1], 0, 1, ref_ratio[lev - 1]);
	}

	auto const &domain_box = geom[0].Domain();
	auto const &dx = geom[0].CellSizeArray();
	auto const &arr = q[0].const_arrays();
	amrex::BaseFab<amrex::Real> proj =
	    amrex::ReduceToPlane<ReduceOp, amrex::Real>(int(dir), domain_box, q[0], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) -> amrex::Real {
		    return dx[int(dir)] * arr[box_no](i, j, k); // data at (i,j,k) of Box box_no
	    });
	amrex::Gpu::streamSynchronize();

	// copy to host pinned memory to work around AMReX bug
	amrex::BaseFab<amrex::Real> proj_host(proj.box(), 1, amrex::The_Pinned_Arena());
	proj_host.copy<amrex::RunOn::Device>(proj);
	amrex::Gpu::streamSynchronize();

	if constexpr (std::is_same<ReduceOp, amrex::ReduceOpSum>::value) {
		amrex::ParallelReduce::Sum(proj_host.dataPtr(), static_cast<int>(proj_host.size()), amrex::ParallelDescriptor::ioProcessor,
					   amrex::ParallelDescriptor::Communicator());
	} else if constexpr (std::is_same<ReduceOp, amrex::ReduceOpMin>::value) {
		amrex::ParallelReduce::Min(proj_host.dataPtr(), static_cast<int>(proj_host.size()), amrex::ParallelDescriptor::ioProcessor,
					   amrex::ParallelDescriptor::Communicator());
	} else {
		amrex::Abort("invalid reduce op!");
	}

	// return BaseFab in host memory
	return proj_host;
}

void WriteProjection(const amrex::Direction dir, std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> const &proj, amrex::Real time, int istep);

} // namespace quokka::diagnostics

#endif // PROJECTION_HPP_