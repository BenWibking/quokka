#ifndef FEXTRACT_HPP_
#define FEXTRACT_HPP_

#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

auto fextract(amrex::MultiFab &mf, amrex::Geometry &geom, int idir, amrex::Real slice_coord, bool center = false)
    -> std::tuple<amrex::Vector<amrex::Real>, amrex::Vector<amrex::Gpu::HostVector<amrex::Real>>>;

#endif // FEXTRACT_HPP_