#ifndef CHECKNAN_HPP_ // NOLINT
#define CHECKNAN_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file CheckNaN.hpp
/// \brief Implements functions to check NaN values in arrays on the GPU.

#include "AMReX_Array4.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_GpuQualifiers.H"

namespace quokka
{
template <typename T>
AMREX_GPU_HOST_DEVICE auto CheckSymmetryArray(amrex::Array4<const amrex::Real> const & /*arr*/, amrex::Box const & /*indexRange*/, const int /*ncomp*/,
					      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/) -> bool
{
	return true; // problem-specific implementation for test problems
}

template <typename T>
AMREX_GPU_HOST_DEVICE auto CheckSymmetryFluxes(amrex::Array4<const amrex::Real> const & /*arr1*/, amrex::Array4<const amrex::Real> const & /*arr2*/,
					       amrex::Box const & /*indexRange*/, const int /*ncomp*/, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/)
    -> bool
{
	return true; // problem-specific implementation for test problems
}

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CheckNaN(amrex::FArrayBox const &arr, amrex::Box const & /*symmetryRange*/, amrex::Box const &nanRange,
						       const int ncomp, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/)
{
	AMREX_ASSERT(!arr.template contains_nan<amrex::RunOn::Gpu>(nanRange, 0, ncomp));
}
} // namespace quokka

#endif // CHECKNAN_HPP_
