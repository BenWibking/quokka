//==============================================================================
// Poisson gravity solver, adapted from Castro's gravity module:
//   Commit history:
//   https://github.com/AMReX-Astro/Castro/commits/main/Source/gravity/Gravity.cpp
// Used under the terms of the open-source license (BSD 3-clause) given here:
//   https://github.com/AMReX-Astro/Castro/blob/main/license.txt
//==============================================================================
/// \file Gravity_residual_impl.hpp
/// \brief Implements a class for solving the Poisson equation.
///

#include "Gravity.hpp"

template <typename T>
void Gravity<T>::test_residual(const Box &bx, amrex::Array4<Real> const &rhs, amrex::Array4<Real> const &ecx, amrex::Array4<Real> const &ecy, amrex::Array4<Real> const &ecz,
			       amrex::GpuArray<Real, AMREX_SPACEDIM> dx)
{
	// Test whether using the edge-based gradients
	// to compute Div(Grad(Phi)) satisfies Lap(phi) = RHS
	// Fill the RHS array with the residual

	amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real lapphi = AMREX_D_TERM((ecx(i + 1, j, k) - ecx(i, j, k)) / dx[0], +(ecy(i, j + 1, k) - ecy(i, j, k)) / dx[1],
					   +(ecz(i, j, k + 1) - ecz(i, j, k)) / dx[2]);
		rhs(i, j, k) -= lapphi;
	});
}

template <typename T> void Gravity<T>::test_level_grad_phi_prev(int level)
{
	BL_PROFILE("Gravity::test_level_grad_phi_prev()");

	// Fill the RHS for the solve
	MultiFab &S_old = *(sim->getStateOld(level));
	MultiFab Rhs(sim->boxArray(level), sim->DistributionMap(level), 1, 0);
	MultiFab::Copy(Rhs, S_old, Density, 0, 1, 0);

	const Geometry &geom_lev = sim->Geom(level);

	// This is a correction for fully periodic domains only
	if (geom_lev.isAllPeriodic()) {
		if (gravity::verbose > 1 && mass_offset != 0.0) {
			amrex::Print() << " ... subtracting average density from RHS at level ... " << level << " " << mass_offset << std::endl;
		}
		Rhs.plus(-mass_offset, 0, 1, 0);
	}

	AMREX_ALWAYS_ASSERT(Ggravity != 0.);
	Rhs.mult(Ggravity);

	if (gravity::verbose > 1) {
		Real rhsnorm = Rhs.norm0();
		amrex::Print() << "... test_level_grad_phi_prev at level " << level << std::endl;
		amrex::Print() << "       norm of RHS             " << rhsnorm << std::endl;
	}

	auto dx = geom_lev.CellSizeArray();

	for (MFIter mfi(Rhs); mfi.isValid(); ++mfi) {
		const Box &bx = mfi.tilebox();
		// does this make sense if the box is covered by a fine grid?

		test_residual(bx, Rhs.array(mfi), (*grad_phi_prev[level][0]).array(mfi), (*grad_phi_prev[level][1]).array(mfi),
			      (*grad_phi_prev[level][2]).array(mfi), dx);
	}

	if (gravity::verbose > 1) {
		Real resnorm = Rhs.norm0();
		amrex::Print() << "       norm of residual        " << resnorm << std::endl;
	}
}

template <typename T> void Gravity<T>::test_level_grad_phi_curr(int level)
{
	BL_PROFILE("Gravity::test_level_grad_phi_curr()");

	// Fill the RHS for the solve
	MultiFab &S_new = *(sim->getStateNew(level));
	MultiFab Rhs(sim->boxArray(level), sim->DistributionMap(level), 1, 0);
	MultiFab::Copy(Rhs, S_new, Density, 0, 1, 0);

	const Geometry &geom_lev = sim->Geom(level);

	// This is a correction for fully periodic domains only
	if (geom_lev.isAllPeriodic()) {
		if (gravity::verbose > 1 && mass_offset != 0.0) {
			amrex::Print() << " ... subtracting average density from RHS in solve ... " << mass_offset << std::endl;
		}
		Rhs.plus(-mass_offset, 0, 1, 0);
	}

	AMREX_ALWAYS_ASSERT(Ggravity != 0.);
	Rhs.mult(Ggravity);

	if (gravity::verbose > 1) {
		Real rhsnorm = Rhs.norm0();
		amrex::Print() << "... test_level_grad_phi_curr at level " << level << std::endl;
		amrex::Print() << "       norm of RHS             " << rhsnorm << std::endl;
	}

	auto dx = geom_lev.CellSizeArray();

	for (MFIter mfi(Rhs); mfi.isValid(); ++mfi) {
		const Box &bx = mfi.tilebox();

		test_residual(bx, Rhs.array(mfi), (*grad_phi_curr[level][0]).array(mfi), (*grad_phi_curr[level][1]).array(mfi),
			      (*grad_phi_curr[level][2]).array(mfi), dx);
	}

	if (gravity::verbose > 1) {
		Real resnorm = Rhs.norm0();
		amrex::Print() << "       norm of residual        " << resnorm << std::endl;
	}
}

template <typename T> void Gravity<T>::test_composite_phi(int crse_level)
{
	BL_PROFILE("Gravity::test_composite_phi()");

	if (gravity::verbose > 1) {
		amrex::Print() << "   " << '\n';
		amrex::Print() << "... test_composite_phi at base level " << crse_level << '\n';
	}

	int finest_level_local = sim->finestLevel();
	int nlevels = finest_level_local - crse_level + 1;

	amrex::Vector<std::unique_ptr<MultiFab>> phi(nlevels);
	amrex::Vector<std::unique_ptr<MultiFab>> rhs(nlevels);
	amrex::Vector<std::unique_ptr<MultiFab>> res(nlevels);

	for (int ilev = 0; ilev < nlevels; ++ilev) {
		int amr_lev = crse_level + ilev;

		phi[ilev] = std::make_unique<MultiFab>(sim->boxArray(amr_lev), sim->DistributionMap(amr_lev), 1, 1);
		MultiFab::Copy(*phi[ilev], phi_new_[amr_lev], 0, 0, 1, 1);

		rhs[ilev] = std::make_unique<MultiFab>(sim->boxArray(amr_lev), sim->DistributionMap(amr_lev), 1, 1);
		MultiFab::Copy(*rhs[ilev], *(sim->getStateNew(amr_lev)), Density, 0, 1, 0);

		res[ilev] = std::make_unique<MultiFab>(sim->boxArray(amr_lev), sim->DistributionMap(amr_lev), 1, 0);
		res[ilev]->setVal(0.);
	}

	Real time = sim->tNew_[crse_level];

	amrex::Vector<Vector<MultiFab *>> grad_phi_null;
	solve_phi_with_mlmg(crse_level, finest_level_local, amrex::GetVecOfPtrs(phi), amrex::GetVecOfPtrs(rhs), grad_phi_null, amrex::GetVecOfPtrs(res), time);

	// Average residual from fine to coarse level before printing the norm
	for (int amr_lev = finest_level_local - 1; amr_lev >= 0; --amr_lev) {
		int ilev = amr_lev - crse_level;
		const IntVect &ratio = sim->refRatio(ilev);
		amrex::average_down(*res[ilev + 1], *res[ilev], 0, 1, ratio);
	}

	for (int amr_lev = crse_level; amr_lev <= finest_level_local; ++amr_lev) {
		Real resnorm = res[amr_lev]->norm0();
		amrex::Print() << "      ... norm of composite residual at level " << amr_lev << "  " << resnorm << '\n';
	}
	amrex::Print() << std::endl;
}
