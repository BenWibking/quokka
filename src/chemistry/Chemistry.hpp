#ifndef CHEMISTRY_HPP_ // NOLINT
#define CHEMISTRY_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Chemistry.hpp
/// \brief Defines methods for integrating primordial chemical network using Microphysics
///

#include <array>

#include "AMReX.H"
#include "AMReX_GpuQualifiers.H"

#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"

#ifdef CHEMISTRY
#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"

namespace quokka::chemistry
{

AMREX_GPU_DEVICE void chemburner(burn_t &chemstate, Real dt);

template <typename problem_t> auto computeChemistry(amrex::MultiFab &mf, const Real dt, const Real max_density_allowed, const Real min_density_allowed) -> bool
{

	// Start off by assuming a successful burn.
	int burn_success = 1;

	amrex::Gpu::Buffer<int> d_num_failed({0});
	auto *p_num_failed = d_num_failed.data();

	int num_failed = 0;

	const BL_PROFILE("Chemistry::computeChemistry()");
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &state = mf.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = state(i, j, k, HydroSystem<problem_t>::density_index);
			const Real xmom = state(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const Real ymom = state(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const Real zmom = state(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			const Real Ener = state(i, j, k, HydroSystem<problem_t>::energy_index);
			const Real Eint = RadSystem<problem_t>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener);

			std::array<Real, NumSpec> chem = {-1.0};
			std::array<Real, NumSpec> inmfracs = {-1.0};
			Real insum = 0.0_rt;

			for (int nn = 0; nn < NumSpec; ++nn) {
				chem[nn] = state(i, j, k, HydroSystem<problem_t>::scalar0_index + nn) /
					   rho; // state has partial densities, so divide by rho to get mass fractions
			}

			// do chemistry using microphysics

			burn_t chemstate;
			chemstate.success = true;
			int burn_failed = 0;

			for (int nn = 0; nn < NumSpec; ++nn) {
				inmfracs[nn] = chem[nn] * rho / spmasses[nn];
				chemstate.xn[nn] = inmfracs[nn];
			}

			// dont do chemistry in cells with densities below the minimum density specified
			if (rho < min_density_allowed) {
				return;
			}

			// stop the test if we have reached very high densities
			if (rho > max_density_allowed) {
				amrex::Abort("Density exceeded max_density_allowed!");
			}

			// input density and eint in burn state
			// Microphysics needs specific eint
			chemstate.rho = rho;
			chemstate.e = Eint / rho;

			// call the EOS to set initial internal energy e
			eos(eos_input_re, chemstate);

			// do the actual integration
			// do it in .cpp so that it is not built at compile time for all tests
			// which would otherwise slow down compilation due to the large RHS file
			chemburner(chemstate, dt);

			if (std::isnan(chemstate.xn[0]) || std::isnan(chemstate.rho)) {
				amrex::Abort("Burner returned NAN");
			}

			if (!chemstate.success) {
				burn_failed = 1;
			}

			if (burn_failed) {
				amrex::Gpu::Atomic::Add(p_num_failed, burn_failed);
			}

			// ensure positivity and normalize
			for (int nn = 0; nn < NumSpec; ++nn) {
				chemstate.xn[nn] = amrex::max(chemstate.xn[nn], small_x);
				inmfracs[nn] = spmasses[nn] * chemstate.xn[nn] / chemstate.rho;
				insum += inmfracs[nn];
			}

			for (int nn = 0; nn < NumSpec; ++nn) {
				inmfracs[nn] /= insum;
				// update the number densities with conserved mass fractions
				chemstate.xn[nn] = inmfracs[nn] * chemstate.rho / spmasses[nn];
			}

			// update the number density of electrons due to charge conservation
			// TODO(psharda): generalize this to other chem networks
			chemstate.xn[0] = -chemstate.xn[3] - chemstate.xn[7] + chemstate.xn[1] + chemstate.xn[12] + chemstate.xn[6] + chemstate.xn[4] +
					  chemstate.xn[9] + 2.0 * chemstate.xn[11];

			// reconserve mass fractions post charge conservation
			insum = 0;
			for (int nn = 0; nn < NumSpec; ++nn) {
				chemstate.xn[nn] = amrex::max(chemstate.xn[nn], small_x);
				inmfracs[nn] = spmasses[nn] * chemstate.xn[nn] / chemstate.rho;
				insum += inmfracs[nn];
			}

			for (int nn = 0; nn < NumSpec; ++nn) {
				inmfracs[nn] /= insum;
				// update the number densities with conserved mass fractions
				chemstate.xn[nn] = inmfracs[nn] * chemstate.rho / spmasses[nn];
			}

			// get the updated specific eint
			eos(eos_input_rt, chemstate);

			// get dEint
			// Quokka uses rho*eint
			const Real dEint = (chemstate.e * chemstate.rho) - Eint;
			state(i, j, k, HydroSystem<problem_t>::internalEnergy_index) += dEint;
			state(i, j, k, HydroSystem<problem_t>::energy_index) += dEint;

			for (int nn = 0; nn < NumSpec; ++nn) {
				state(i, j, k, HydroSystem<problem_t>::scalar0_index + nn) = inmfracs[nn] * rho; // scale by rho to return partial densities
			}
		});

#if defined(AMREX_USE_HIP)
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif
	}

	num_failed = *(d_num_failed.copyToHost());

	burn_success = !num_failed;
	amrex::ParallelDescriptor::ReduceIntMin(burn_success);

	if (!burn_success) {
		// amrex::Abort("Burn failed in VODE. Aborting.");
		amrex::Print() << "\t>> WARNING: Unsuccessful burn. Retrying hydro step."
			       << "\n";
	}

	return burn_success;
}

} // namespace quokka::chemistry
#endif
#endif // CHEMISTRY_HPP_
