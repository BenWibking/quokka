#ifndef CICPARTICLES_HPP_ // NOLINT
#define CICPARTICLES_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file CICParticles.hpp
/// \brief Implements the particle container for gravitationally-interacting particles
///

#include "AMReX.H"
#include "AMReX_AmrParticles.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParIter.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_Particles.H"

template <int T_NArrayReal, int T_NArrayInt=0,
          template<class> class Allocator=amrex::DefaultAllocator, class CellAssignor=amrex::DefaultAssignor>
using AmrParticleContainerPureSoA = amrex::AmrParticleContainer_impl<amrex::SoAParticle<T_NArrayReal, T_NArrayInt>, T_NArrayReal, T_NArrayInt, Allocator, CellAssignor>;

namespace quokka
{

enum ParticleDataIdx { ParticleMassIdx = 0, ParticleVxIdx, ParticleVyIdx, ParticleVzIdx };
constexpr int CICParticleRealComps = 4; // mass vx vy vz

using CICParticleContainer = amrex::AmrParticleContainer<CICParticleRealComps>;
using CICParticleIterator = amrex::ParIter<CICParticleRealComps>;

struct CICDeposition {
	amrex::Real Gconst{};
	int start_part_comp{};
	int start_mesh_comp{};
	int num_comp{};

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const CICParticleContainer::ParticleType &p, amrex::Array4<amrex::Real> const &rho,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp,
				      [=] AMREX_GPU_DEVICE(const CICParticleContainer::ParticleType &part, int comp) {
					      return 4.0 * M_PI * Gconst * part.rdata(comp); // weight by 4 pi G
				      });
	}
};

} // namespace quokka

#endif // CICPARTICLES_HPP_