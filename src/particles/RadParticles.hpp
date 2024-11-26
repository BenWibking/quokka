#ifndef RADPARTICLES_HPP_ // NOLINT
#define RADPARTICLES_HPP_
/// \file RadParticles.hpp
/// \brief Implements the particle container for radiating particles.
///

#include "AMReX_AmrParticles.H"
#include "AMReX_ParIter.H"

namespace quokka
{

enum RadParticleDataIdx { RadParticleMassIdx = 0, RadParticleBirthTimeIdx, RadParticleDeathTimeIdx };
constexpr int RadParticleRealComps = 3;

using RadParticleContainer = amrex::AmrParticleContainer<RadParticleRealComps>;
using RadParticleIterator = amrex::ParIter<RadParticleRealComps>;

} // namespace quokka

#endif // RADPARTICLES_HPP_