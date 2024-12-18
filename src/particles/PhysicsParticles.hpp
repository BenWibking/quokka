#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include "AMReX_AmrParticles.H"
#include "AMReX_Array.H"
#include "AMReX_Extension.H"
#include "AMReX_ParIter.H"
#include "AMReX_ParticleContainerBase.H"
#include "AMReX_ParticleInterpolators.H"
#include "physics_info.hpp"

namespace quokka
{

// CIC particles
enum CICParticleDataIdx { CICParticleMassIdx = 0, CICParticleVxIdx, CICParticleVyIdx, CICParticleVzIdx };
constexpr int CICParticleRealComps = 4; // mass vx vy vz
using CICParticleContainer = amrex::AmrParticleContainer<CICParticleRealComps>;
using CICParticleIterator = amrex::ParIter<CICParticleRealComps>;

// Radiation particles
enum RadParticleDataIdx { RadParticleBirthTimeIdx = 0, RadParticleDeathTimeIdx, RadParticleLumIdx};
template <typename problem_t>
constexpr int RadParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 2 + Physics_Traits<problem_t>::nGroups; // birth_time death_time lum1 ... lumN
	} else {
		return 2; // birth_time death_time
	}
}();
template <typename problem_t> using RadParticleContainer = amrex::AmrParticleContainer<RadParticleRealComps<problem_t>>;
template <typename problem_t> using RadParticleIterator = amrex::ParIter<RadParticleRealComps<problem_t>>;

// CICRad particles
enum CICRadParticleDataIdx { CICRadParticleMassIdx = 0, CICRadParticleVxIdx, CICRadParticleVyIdx, CICRadParticleVzIdx, CICRadParticleBirthTimeIdx, CICRadParticleDeathTimeIdx, CICRadParticleLumIdx };
template <typename problem_t>
constexpr int CICRadParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
		return 6 + Physics_Traits<problem_t>::nGroups; // mass vx vy vz birth_time death_time lum1 ... lumN
	} else {
		return 6; // mass vx vy vz birth_time death_time
	}
}();
template <typename problem_t> using CICRadParticleContainer = amrex::AmrParticleContainer<CICRadParticleRealComps<problem_t>>;
template <typename problem_t> using CICRadParticleIterator = amrex::ParIter<CICRadParticleRealComps<problem_t>>;

struct MassDeposition {
	amrex::Real Gconst{};
	int start_part_comp{};
	int start_mesh_comp{};
	int num_comp{};

	template <typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ParticleType &p, amrex::Array4<amrex::Real> const &rho,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, rho, start_part_comp, start_mesh_comp, num_comp, [this] AMREX_GPU_DEVICE(const ParticleType &part, int comp) {
			return 4.0 * M_PI * Gconst * part.rdata(comp); // weight by 4 pi G
		});
	}
};

struct RadDeposition {
	double current_time{};
	int start_part_comp{};
	int start_mesh_comp{};
	int num_comp{};

	template <typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ParticleType &p, amrex::Array4<amrex::Real> const &radEnergySource,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp,
				      [this, dxi] AMREX_GPU_DEVICE(const ParticleType &part, int comp) {
					      if (current_time < part.rdata(RadParticleBirthTimeIdx) || current_time >= part.rdata(RadParticleDeathTimeIdx)) {
						      return 0.0;
					      }
					      return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
				      });
	}
};

// Forward declarations
template <typename problem_t> class PhysicsParticleRegister;

// Base class for physics particle descriptors
class PhysicsParticleDescriptor
{
      protected:
	int massIndex_{-1};		 // index for gravity mass, -1 if not used
	int lumIndex_{-1};		 // index for radiation luminosity, -1 if not used
	bool interactsWithHydro_{false}; // whether particles interact with hydro

      public:
	PhysicsParticleDescriptor(int mass_idx, int lum_idx, bool hydro_interact)
	    : massIndex_(mass_idx), lumIndex_(lum_idx), interactsWithHydro_(hydro_interact)
	{
	}
	~PhysicsParticleDescriptor() = default;
	amrex::ParticleContainerBase *neighborParticleContainer_{nullptr}; // non-owning pointer to particle container

	// Getters
	[[nodiscard]] auto getMassIndex() const -> int { return massIndex_; }
	[[nodiscard]] auto getLumIndex() const -> int { return lumIndex_; }
	[[nodiscard]] auto getInteractsWithHydro() const -> bool { return interactsWithHydro_; }

	void hydroInteract() {} // Default no-op

	// Delete copy/move constructors/assignments
	PhysicsParticleDescriptor(const PhysicsParticleDescriptor &) = delete;
	PhysicsParticleDescriptor &operator=(const PhysicsParticleDescriptor &) = delete;
	PhysicsParticleDescriptor(PhysicsParticleDescriptor &&) = delete;
	PhysicsParticleDescriptor &operator=(PhysicsParticleDescriptor &&) = delete;
};

// Registry for physics particles
template <typename problem_t> class PhysicsParticleRegister
{
      private:
	std::map<std::string, std::unique_ptr<PhysicsParticleDescriptor>> particleRegistry_;

      public:
	PhysicsParticleRegister() = default;
	~PhysicsParticleRegister() = default;

	// Register a new particle type
	template <typename ParticleDescriptor> void registerParticleType(const std::string &name, std::unique_ptr<ParticleDescriptor> descriptor)
	{
		particleRegistry_[name] = std::move(descriptor);
	}

	// Get a particle descriptor
	[[nodiscard]] auto getParticleDescriptor(const std::string &name) const -> const PhysicsParticleDescriptor *
	{
		auto it = particleRegistry_.find(name);
		if (it != particleRegistry_.end()) {
			return it->second.get();
		}
		return nullptr;
	}

	// Deposit radiation from all particles that have luminosity
	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getLumIndex() >= 0) {
				// Get particle container and deposit luminosity
				auto *container = dynamic_cast<RadParticleContainer<problem_t> *>(descriptor->neighborParticleContainer_);
				if (container != nullptr) {
					amrex::ParticleToMesh(*container, radEnergySource, lev,
							      RadDeposition{current_time, descriptor->getLumIndex(), 0, Physics_Traits<problem_t>::nGroups},
							      false);
				}
			}
		}
	}

	// Deposit mass from all particles that have mass for gravity calculation
	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				// Get particle container and deposit mass
				auto *container = dynamic_cast<RadParticleContainer<problem_t> *>(descriptor->neighborParticleContainer_);
				if (container != nullptr) {
					amrex::ParticleToMesh(*container, amrex::GetVecOfPtrs(rhs), 0, finest_lev,
							      MassDeposition{Gconst, descriptor->getMassIndex(), 0, 1}, true);
				}
			}
		}
	}

	// Run Redistribute(lev) on all particles in particleRegistry_
	void redistribute(int lev)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			auto *container = dynamic_cast<RadParticleContainer<problem_t> *>(descriptor->neighborParticleContainer_);
			if (container != nullptr) {
				container->Redistribute(lev);
			}
		}
	}

	// Run Redistribute(lev, ngrow) on all particles in particleRegistry_
	void redistribute(int lev, int ngrow)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			auto *container = dynamic_cast<RadParticleContainer<problem_t> *>(descriptor->neighborParticleContainer_);
			if (container != nullptr) {
				container->Redistribute(lev, container->finestLevel(), ngrow);
			}
		}
	}

	// Run WritePlotFile(plotfilename, name) on all particles in particleRegistry_
	void writePlotFile(const std::string &plotfilename)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			auto *container = dynamic_cast<RadParticleContainer<problem_t> *>(descriptor->neighborParticleContainer_);
			if (container != nullptr) {
				container->WritePlotFile(plotfilename, name);
			}
		}
	}

	// Run Checkpoint(checkpointname, name, true) on all particles in particleRegistry_
	void writeCheckpoint(const std::string &checkpointname, bool include_header)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			auto *container = dynamic_cast<RadParticleContainer<problem_t> *>(descriptor->neighborParticleContainer_);
			if (container != nullptr) {
				container->Checkpoint(checkpointname, name, include_header);
			}
		}
	}

	// Delete copy/move constructors/assignments to prevent accidental copying or moving of objects.
	// The class has a raw pointer member, neighborParticleContainer_. Copying would be dangerous as multiple objects could end up pointing to the same
	// memory. The class is meant to be a base class for particle descriptors, and copying base classes can lead to slicing problems
	PhysicsParticleRegister(const PhysicsParticleRegister &) = delete;
	PhysicsParticleRegister &operator=(const PhysicsParticleRegister &) = delete;
	PhysicsParticleRegister(PhysicsParticleRegister &&) = delete;
	PhysicsParticleRegister &operator=(PhysicsParticleRegister &&) = delete;
};

} // namespace quokka

#endif // PHYSICS_PARTICLES_HPP_
