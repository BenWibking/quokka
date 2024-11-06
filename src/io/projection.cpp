//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
//! \file projection.cpp
///  \brief AMReX I/O for 2D projections

#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_Orientation.H"
#include "AMReX_PlotFileUtil.H"

#include "projection.hpp"

namespace quokka::diagnostics
{

namespace detail
{
auto direction_to_string(const amrex::Direction dir) -> std::string
{
	if (dir == amrex::Direction::x) {
		return std::string("x");
	}
#if AMREX_SPACEDIM >= 2
	if (dir == amrex::Direction::y) {
		return std::string("y");
	}
#endif
#if AMREX_SPACEDIM == 3
	if (dir == amrex::Direction::z) {
		return std::string("z");
	}
#endif

	amrex::Error("invalid direction in quokka::diagnostics::direction_to_string!");
	return std::string("");
}
} // namespace detail

void WriteProjection(const amrex::Direction dir, std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> const &proj, amrex::Real time, int istep)
{
	// write projections to plotfile

	auto const &firstFab = proj.begin()->second;
	const amrex::BoxArray ba(firstFab.box());
	const amrex::DistributionMapping dm(amrex::Vector<int>{0});
	amrex::MultiFab mf_all(ba, dm, static_cast<int>(proj.size()), 0);
	amrex::Vector<std::string> varnames;

	auto iter = proj.begin();
	for (int icomp = 0; icomp < static_cast<int>(proj.size()); ++icomp) {
		const std::string &varname = iter->first;
		const amrex::BaseFab<amrex::Real> &baseFab = iter->second;

		const amrex::BoxArray ba(baseFab.box());
		const amrex::DistributionMapping dm(amrex::Vector<int>{0});
		amrex::MultiFab mf(ba, dm, 1, 0, amrex::MFInfo().SetAlloc(false));
		if (amrex::ParallelDescriptor::IOProcessor()) {
			mf.setFab(0, amrex::FArrayBox(baseFab.array()));
		}
		amrex::MultiFab::Copy(mf_all, mf, 0, icomp, 1, 0);
		varnames.push_back(varname);
		++iter;
	}

	const std::string basename = "proj_" + detail::direction_to_string(dir) + "_plt";
	const std::string filename = amrex::Concatenate(basename, istep, 5);
	amrex::Print() << "Writing projection " << filename << "\n";
	const amrex::Geometry mygeom(firstFab.box());
	amrex::WriteSingleLevelPlotfile(filename, mf_all, varnames, mygeom, time, istep);
}

} // namespace quokka::diagnostics
