#ifndef RADPARTICLES_HPP_ // NOLINT
#define RADPARTICLES_HPP_
/// \file RadParticles.hpp
/// \brief Implements the particle container for radiating particles.
///

#include "AMReX_AmrParticles.H"
#include "AMReX_ParIter.H"
#include "AMReX_ParticleInterpolators.H"

namespace quokka
{

enum RadParticleDataIdx { RadParticleMassIdx = 0, RadParticleBirthTimeIdx, RadParticleDeathTimeIdx };
constexpr int RadParticleRealComps = 3;

using RadParticleContainer = amrex::AmrParticleContainer<RadParticleRealComps>;
using RadParticleIterator = amrex::ParIter<RadParticleRealComps>;

struct RadDeposition {
	double current_time{};
	int start_part_comp{};
	int start_mesh_comp{};
	int num_comp{};

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const RadParticleContainer::ParticleType &p, amrex::Array4<amrex::Real> const &radEnergySource,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp,
				      [=] AMREX_GPU_DEVICE(const RadParticleContainer::ParticleType &part, int comp) {
					      if (current_time < part.rdata(RadParticleBirthTimeIdx) || current_time >= part.rdata(RadParticleDeathTimeIdx)) {
						      return 0.0;
					      }
					      return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
				      });
	}
};

} // namespace quokka

#endif // RADPARTICLES_HPP_