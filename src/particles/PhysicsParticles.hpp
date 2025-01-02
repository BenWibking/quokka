#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "AMReX.H"
#include "AMReX_AmrParticles.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_ParIter.H"
#include "AMReX_ParticleContainerBase.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_Vector.H"
#include "physics_info.hpp"

namespace quokka
{

// CIC particles
enum CICParticleDataIdx { CICParticleMassIdx = 0, CICParticleVxIdx, CICParticleVyIdx, CICParticleVzIdx };
#if AMREX_SPACEDIM == 1
constexpr int CICParticleRealComps = 2; // mass vx
#elif AMREX_SPACEDIM == 2
constexpr int CICParticleRealComps = 3; // mass vx vy
#elif AMREX_SPACEDIM == 3
constexpr int CICParticleRealComps = 4; // mass vx vy vz
#endif
using CICParticleContainer = amrex::AmrParticleContainer<CICParticleRealComps>;
using CICParticleIterator = amrex::ParIter<CICParticleRealComps>;

// Radiation particles
enum RadParticleDataIdx { RadParticleBirthTimeIdx = 0, RadParticleDeathTimeIdx, RadParticleLumIdx };
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
enum CICRadParticleDataIdx {
	CICRadParticleMassIdx = 0,
	CICRadParticleVxIdx,
	CICRadParticleVyIdx,
	CICRadParticleVzIdx,
	CICRadParticleBirthTimeIdx,
	CICRadParticleDeathTimeIdx,
	CICRadParticleLumIdx
};
// CICRadParticleRealComps has to be defined when radiation is disabled to avoid compilation errors
template <typename problem_t>
constexpr int CICRadParticleRealComps = []() constexpr {
	if constexpr (Physics_Traits<problem_t>::is_hydro_enabled || Physics_Traits<problem_t>::is_radiation_enabled) {
#if AMREX_SPACEDIM == 1
		return 4 + Physics_Traits<problem_t>::nGroups; // mass vx birth_time death_time lum1 ... lumN
#elif AMREX_SPACEDIM == 2
		return 5 + Physics_Traits<problem_t>::nGroups; // mass vx vy birth_time death_time lum1 ... lumN
#elif AMREX_SPACEDIM == 3
		return 6 + Physics_Traits<problem_t>::nGroups; // mass vx vy vz birth_time death_time lum1 ... lumN
#endif
	} else {
#if AMREX_SPACEDIM == 1
		return 4; // mass vx birth_time death_time
#elif AMREX_SPACEDIM == 2
		return 5; // mass vx vy birth_time death_time
#elif AMREX_SPACEDIM == 3
		return 6; // mass vx vy vz birth_time death_time
#endif
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
	int birthTimeIndex{};
	// deathTimeIndex is assumed to be birthTimeIndex + 1

	template <typename ParticleType>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const ParticleType &p, amrex::Array4<amrex::Real> const &radEnergySource,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp,
				      [this, dxi] AMREX_GPU_DEVICE(const ParticleType &part, int comp) {
					      if (current_time < part.rdata(birthTimeIndex) || current_time >= part.rdata(birthTimeIndex + 1)) {
						      return 0.0;
					      }
					      return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
				      });
	}
};

// Forward declarations
template <typename problem_t> class PhysicsParticleRegister;

// Add a virtual interface for particle operations
class ParticleOperations
{
      public:
	virtual ~ParticleOperations() = default;
	virtual void redistribute(int lev) = 0;
	virtual void redistribute(int lev, int ngrow) = 0;
	virtual void writePlotFile(const std::string &plotfilename, const std::string &name) = 0;
	virtual void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) = 0;
	virtual void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int lumIndex, int birthTimeIndex, int nGroups) = 0;
	virtual void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst, int massIndex) = 0;
	virtual void driftParticlesAllLevels(amrex::Real dt, int massIndex) = 0;
	virtual void kickParticles(amrex::Real dt, const amrex::MultiFab &accel, const amrex::Vector<amrex::Geometry> &geom, int lev, int massIndex) = 0;
};

// Template wrapper that implements the interface for any particle container type
template <typename ParticleContainerType> class ParticleOperationsImpl : public ParticleOperations
{

	ParticleContainerType *container_;

      public:
	explicit ParticleOperationsImpl(ParticleContainerType *container) : container_(container) {}

	void redistribute(int lev) override
	{
		if (container_) {
			container_->Redistribute(lev);
		}
	}

	void redistribute(int lev, int ngrow) override
	{
		if (container_) {
			container_->Redistribute(lev, container_->finestLevel(), ngrow);
		}
	}

	void writePlotFile(const std::string &plotfilename, const std::string &name) override
	{
		if (container_) {
			container_->WritePlotFile(plotfilename, name);
		}
	}

	void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) override
	{
		if (container_) {
			container_->Checkpoint(checkpointname, name, include_header);
		}
	}

	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int lumIndex, int birthTimeIndex, int nGroups) override
	{
		if (container_) {
			amrex::ParticleToMesh(*container_, radEnergySource, lev, RadDeposition{current_time, lumIndex, 0, nGroups, birthTimeIndex}, false);
		}
	}

	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst, int massIndex) override
	{
		if (container_) {
			amrex::ParticleToMesh(*container_, amrex::GetVecOfPtrs(rhs), 0, finest_lev, MassDeposition{Gconst, massIndex, 0, 1}, true);
		}
	}

	void driftParticlesAllLevels(amrex::Real dt, int massIndex) override
	{
		if (container_) {
			for (int lev = 0; lev <= container_->finestLevel(); ++lev) {
				for (typename ParticleContainerType::ParIterType pti(*container_, lev); pti.isValid(); ++pti) {
					auto &particles = pti.GetArrayOfStructs();
					typename ParticleContainerType::ParticleType *pData = particles().data();
					const amrex::Long np = pti.numParticles();

					amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
						typename ParticleContainerType::ParticleType &p =
						    pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
						// update particle position
						for (int i = 0; i < AMREX_SPACEDIM; ++i) {
							p.pos(i) += dt * p.rdata(massIndex + 1 + i);
						}
					});
				}
			}
		}
	}

	void kickParticles(amrex::Real dt, const amrex::MultiFab &accel, const amrex::Vector<amrex::Geometry> &geom, int lev, int massIndex) override
	{
		if (container_) {
			const auto dx_inv = geom[lev].InvCellSizeArray();
			for (typename ParticleContainerType::ParIterType pti(*container_, lev); pti.isValid(); ++pti) {
				auto &particles = pti.GetArrayOfStructs();
				typename ParticleContainerType::ParticleType *pData = particles().data();
				const amrex::Long np = pti.numParticles();

				amrex::Array4<const amrex::Real> const &accel_arr = accel.array(pti);
				const auto plo = geom[lev].ProbLoArray(); // TODO(cch): later, pass geom[lev] as amrex::Geometry

				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
					typename ParticleContainerType::ParticleType &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					amrex::ParticleInterpolator::Linear interp(p, plo, dx_inv);
					interp.MeshToParticle(
					    p, accel_arr, 0, massIndex + 1, AMREX_SPACEDIM,
					    [=] AMREX_GPU_DEVICE(amrex::Array4<const amrex::Real> const &acc, int i, int j, int k, int comp) {
						    return acc(i, j, k, comp); // no weighting
					    },
					    [=] AMREX_GPU_DEVICE(typename ParticleContainerType::ParticleType & p, int comp, amrex::Real acc_comp) {
						    // kick particle by updating its velocity
								// AMREX_ASSERT(comp >= massIndex + 1 && comp < massIndex + 1 + AMREX_SPACEDIM);
						    p.rdata(comp) += 0.5 * dt * acc_comp;
					    });
				});
			}
		}
	}
};

// Base class for physics particle descriptors
class PhysicsParticleDescriptor
{
      protected:
	int massIndex_{-1};				 // index for gravity mass, -1 if not used
	int lumIndex_{-1};				 // index for radiation luminosity, -1 if not used
	int birthTimeIndex_{-1};			 // index for birth time, -1 if not used
	bool interactsWithHydro_{false};		 // whether particles interact with hydro
	std::unique_ptr<ParticleOperations> operations_; // Add this

      public:
	PhysicsParticleDescriptor(int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact)
	    : massIndex_(mass_idx), lumIndex_(lum_idx), birthTimeIndex_(birth_time_idx), interactsWithHydro_(hydro_interact)
	{
	}
	~PhysicsParticleDescriptor() = default;
	amrex::ParticleContainerBase *neighborParticleContainer_{nullptr}; // non-owning pointer to particle container

	// Getters
	[[nodiscard]] auto getMassIndex() const -> int { return massIndex_; }
	[[nodiscard]] auto getLumIndex() const -> int { return lumIndex_; }
	[[nodiscard]] auto getBirthTimeIndex() const -> int { return birthTimeIndex_; }
	[[nodiscard]] auto getInteractsWithHydro() const -> bool { return interactsWithHydro_; }

	void hydroInteract() {} // Default no-op

	// Delete copy/move constructors/assignments
	PhysicsParticleDescriptor(const PhysicsParticleDescriptor &) = delete;
	PhysicsParticleDescriptor &operator=(const PhysicsParticleDescriptor &) = delete;
	PhysicsParticleDescriptor(PhysicsParticleDescriptor &&) = delete;
	PhysicsParticleDescriptor &operator=(PhysicsParticleDescriptor &&) = delete;

	// Setter for operations
	template <typename ParticleContainerType> void setParticleContainer(ParticleContainerType *container)
	{
		neighborParticleContainer_ = container;
		operations_ = std::make_unique<ParticleOperationsImpl<ParticleContainerType>>(container);
	}

	// Getter for particle container
	template <typename ParticleContainerType> ParticleContainerType *getParticleContainer() const
	{
		return dynamic_cast<ParticleContainerType *>(neighborParticleContainer_);
	}

	// Getter for operations
	ParticleOperations *getOperations() const { return operations_.get(); }
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
			if (auto *ops = descriptor->getOperations()) {
				if (descriptor->getLumIndex() >= 0) {
					ops->depositRadiation(radEnergySource, lev, current_time, descriptor->getLumIndex(), descriptor->getBirthTimeIndex(),
							      Physics_Traits<problem_t>::nGroups);
				}
			}
		}
	}

	// Deposit mass from all particles that have mass for gravity calculation
	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (auto *ops = descriptor->getOperations()) {
				if (descriptor->getMassIndex() >= 0) {
					ops->depositMass(rhs, finest_lev, Gconst, descriptor->getMassIndex());
				}
			}
		}
	}

	// Drift particles
	void driftParticlesAllLevels(amrex::Real dt)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (auto *ops = descriptor->getOperations()) {
				if (descriptor->getMassIndex() >= 0) {
					ops->driftParticlesAllLevels(dt, descriptor->getMassIndex());
				}
			}
		}
	}

	// Kick particles
	void kickParticles(amrex::Real dt, const amrex::MultiFab &accel, const amrex::Vector<amrex::Geometry> &geom, int lev)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (auto *ops = descriptor->getOperations()) {
				if (descriptor->getMassIndex() >= 0) {
					ops->kickParticles(dt, accel, geom, lev, descriptor->getMassIndex());
				}
			}
		}
	}

	// Run Redistribute(lev) on all particles in particleRegistry_
	void redistribute(int lev)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (auto *ops = descriptor->getOperations()) {
				ops->redistribute(lev);
			}
		}
	}

	// Run Redistribute(lev, ngrow) on all particles in particleRegistry_
	void redistribute(int lev, int ngrow)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (auto *ops = descriptor->getOperations()) {
				ops->redistribute(lev, ngrow);
			}
		}
	}

	// Run WritePlotFile(plotfilename, name) on all particles in particleRegistry_
	void writePlotFile(const std::string &plotfilename)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (auto *ops = descriptor->getOperations()) {
				ops->writePlotFile(plotfilename, name);
			}
		}
	}

	// Run Checkpoint(checkpointname, name, true) on all particles in particleRegistry_
	void writeCheckpoint(const std::string &checkpointname, bool include_header) const
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (auto *ops = descriptor->getOperations()) {
				ops->writeCheckpoint(checkpointname, name, include_header);
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
