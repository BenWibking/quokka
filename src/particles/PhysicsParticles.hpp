#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include <map>
#include <string>

#include "AMReX_AmrParticles.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParIter.H"
#include "AMReX_ParticleContainerBase.H"
#include "AMReX_ParticleInterpolators.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "physics_info.hpp"

namespace quokka
{

// CIC particles
enum CICParticleDataIdx { CICParticleMassIdx = 0, CICParticleVxIdx, CICParticleVyIdx, CICParticleVzIdx };
constexpr int CICParticleRealComps = 4; // mass vx vy vz
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

// Base class for physics particle descriptors
class PhysicsParticleDescriptor
{
      protected:
	int massIndex_{-1};						   // index for gravity mass, -1 if not used
	int lumIndex_{-1};						   // index for radiation luminosity, -1 if not used
	int birthTimeIndex_{-1};					   // index for birth time, -1 if not used
	bool interactsWithHydro_{false};				   // whether particles interact with hydro
	amrex::ParticleContainerBase *neighborParticleContainer_{nullptr}; // non-owning pointer to particle container

      public:
	PhysicsParticleDescriptor(int mass_idx, int lum_idx, int birth_time_idx, bool hydro_interact)
	    : massIndex_(mass_idx), lumIndex_(lum_idx), birthTimeIndex_(birth_time_idx), interactsWithHydro_(hydro_interact)
	{
	}
	virtual ~PhysicsParticleDescriptor() = default;

	// Getters
	[[nodiscard]] auto getMassIndex() const -> int { return massIndex_; }
	[[nodiscard]] auto getLumIndex() const -> int { return lumIndex_; }
	[[nodiscard]] auto getBirthTimeIndex() const -> int { return birthTimeIndex_; }
	[[nodiscard]] auto getInteractsWithHydro() const -> bool { return interactsWithHydro_; }

	virtual void hydroInteract() {} // Default no-op

	// Delete copy/move constructors/assignments
	PhysicsParticleDescriptor(const PhysicsParticleDescriptor &) = delete;
	PhysicsParticleDescriptor &operator=(const PhysicsParticleDescriptor &) = delete;
	PhysicsParticleDescriptor(PhysicsParticleDescriptor &&) = delete;
	PhysicsParticleDescriptor &operator=(PhysicsParticleDescriptor &&) = delete;

	// Setter for particle container
	template <typename ParticleContainerType> void setParticleContainer(ParticleContainerType *container) { neighborParticleContainer_ = container; }

	// Getter for particle container
	template <typename ParticleContainerType> auto getParticleContainer() -> ParticleContainerType *
	{
		return dynamic_cast<ParticleContainerType *>(neighborParticleContainer_);
	}

	// Virtual operations that will be called on the particle container
	virtual void redistribute(int lev) = 0;
	virtual void redistribute(int lev, int ngrow) = 0;
	virtual void writePlotFile(const std::string &plotfilename, const std::string &name) = 0;
	virtual void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) = 0;
	virtual void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int lumIndex, int birthTimeIndex, int nGroups) = 0;
	virtual void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst, int massIndex) = 0;
	virtual void driftParticlesAllLevels(amrex::Real dt) = 0;
	virtual void kickParticles(amrex::Real dt, const amrex::MultiFab &accel, const amrex::Vector<amrex::Geometry> &geom, int lev) = 0;
};

// Derived class for Rad particles
template <typename problem_t> class RadParticleDescriptor : public PhysicsParticleDescriptor
{
      public:
	using PhysicsParticleDescriptor::PhysicsParticleDescriptor;

	void redistribute(int lev) override
	{
		if (auto *container = dynamic_cast<quokka::RadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			container->Redistribute(lev);
		}
	}

	void redistribute(int lev, int ngrow) override
	{
		if (auto *container = dynamic_cast<quokka::RadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			container->Redistribute(lev, container->finestLevel(), ngrow);
		}
	}

	void writePlotFile(const std::string &plotfilename, const std::string &name) override
	{
		if (auto *container = dynamic_cast<quokka::RadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			container->WritePlotFile(plotfilename, name);
		}
	}

	void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) override
	{
		if (auto *container = dynamic_cast<quokka::RadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			container->Checkpoint(checkpointname, name, include_header);
		}
	}

	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int lumIndex, int birthTimeIndex, int nGroups) override
	{
		if (auto *container = dynamic_cast<quokka::RadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			amrex::ParticleToMesh(*container, radEnergySource, lev, RadDeposition{current_time, lumIndex, 0, nGroups, birthTimeIndex}, false);
		}
	}

	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst, int massIndex) override
	{
		if (auto *container = dynamic_cast<quokka::RadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			amrex::ParticleToMesh(*container, amrex::GetVecOfPtrs(rhs), 0, finest_lev, MassDeposition{Gconst, massIndex, 0, 1}, true);
		}
	}

	void driftParticlesAllLevels(amrex::Real dt) override
	{
		// Rad particles don't have velocity components, so this is a no-op
	}

	void kickParticles(amrex::Real dt, const amrex::MultiFab &accel, const amrex::Vector<amrex::Geometry> &geom, int lev) override
	{
		// Rad particles don't have velocity components, so this is a no-op
	}
};

// Derived class for CIC particles
template <typename problem_t> class CICParticleDescriptor : public PhysicsParticleDescriptor
{
      public:
	using PhysicsParticleDescriptor::PhysicsParticleDescriptor;

	void redistribute(int lev) override
	{
		if (auto *container = dynamic_cast<quokka::CICParticleContainer *>(this->neighborParticleContainer_)) {
			container->Redistribute(lev);
		}
	}

	void redistribute(int lev, int ngrow) override
	{
		if (auto *container = dynamic_cast<quokka::CICParticleContainer *>(this->neighborParticleContainer_)) {
			container->Redistribute(lev, container->finestLevel(), ngrow);
		}
	}

	void writePlotFile(const std::string &plotfilename, const std::string &name) override
	{
		if (auto *container = dynamic_cast<quokka::CICParticleContainer *>(this->neighborParticleContainer_)) {
			container->WritePlotFile(plotfilename, name);
		}
	}

	void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) override
	{
		if (auto *container = dynamic_cast<quokka::CICParticleContainer *>(this->neighborParticleContainer_)) {
			container->Checkpoint(checkpointname, name, include_header);
		}
	}

	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int lumIndex, int birthTimeIndex, int nGroups) override
	{
		if (auto *container = dynamic_cast<quokka::CICParticleContainer *>(this->neighborParticleContainer_)) {
			amrex::ParticleToMesh(*container, radEnergySource, lev, RadDeposition{current_time, lumIndex, 0, nGroups, birthTimeIndex}, false);
		}
	}

	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst, int massIndex) override
	{
		if (auto *container = dynamic_cast<quokka::CICParticleContainer *>(this->neighborParticleContainer_)) {
			amrex::ParticleToMesh(*container, amrex::GetVecOfPtrs(rhs), 0, finest_lev, MassDeposition{Gconst, massIndex, 0, 1}, true);
		}
	}

	void driftParticlesAllLevels(amrex::Real dt) override
	{
		if (auto *container = dynamic_cast<quokka::CICParticleContainer *>(this->neighborParticleContainer_)) {
			for (int lev = 0; lev <= container->finestLevel(); ++lev) {
				for (quokka::CICParticleIterator pIter(*container, lev); pIter.isValid(); ++pIter) {
					auto &particles = pIter.GetArrayOfStructs();
					quokka::CICParticleContainer::ParticleType *pData = particles().data();
					const amrex::Long np = pIter.numParticles();

					amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
						quokka::CICParticleContainer::ParticleType &p =
						    pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
						// update particle position
						for (int i = 0; i < AMREX_SPACEDIM; ++i) {
							p.pos(i) += dt * p.rdata(quokka::CICParticleVxIdx + i);
						}
					});
				}
			}
		}
	}

	void kickParticles(amrex::Real dt, const amrex::MultiFab &accel, const amrex::Vector<amrex::Geometry> &geom, int lev) override
	{
		if (auto *container = dynamic_cast<quokka::CICParticleContainer *>(this->neighborParticleContainer_)) {
			const auto dx_inv = geom[lev].InvCellSizeArray();
			for (quokka::CICParticleIterator pIter(*container, lev); pIter.isValid(); ++pIter) {
				auto &particles = pIter.GetArrayOfStructs();
				quokka::CICParticleContainer::ParticleType *pData = particles().data();
				const amrex::Long np = pIter.numParticles();

				amrex::Array4<const amrex::Real> const &accel_arr = accel.array(pIter);
				const auto plo = geom[lev].ProbLoArray();

				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
					quokka::CICParticleContainer::ParticleType &p = pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					amrex::ParticleInterpolator::Linear interp(p, plo, dx_inv);
					interp.MeshToParticle(
					    p, accel_arr, 0, quokka::CICParticleVxIdx, AMREX_SPACEDIM,
					    [=] AMREX_GPU_DEVICE(amrex::Array4<const amrex::Real> const &acc, int i, int j, int k, int comp) {
						    return acc(i, j, k, comp); // no weighting
					    },
					    [=] AMREX_GPU_DEVICE(quokka::CICParticleContainer::ParticleType & p, int comp, amrex::Real acc_comp) {
						    // kick particle by updating its velocity
						    p.rdata(comp) += 0.5 * dt * static_cast<amrex::ParticleReal>(acc_comp);
					    });
				});
			}
		}
	}
};

// Derived class for CICRad particles
template <typename problem_t> class CICRadParticleDescriptor : public PhysicsParticleDescriptor
{
      public:
	using PhysicsParticleDescriptor::PhysicsParticleDescriptor;

	void redistribute(int lev) override
	{
		if (auto *container = dynamic_cast<quokka::CICRadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			container->Redistribute(lev);
		}
	}

	void redistribute(int lev, int ngrow) override
	{
		if (auto *container = dynamic_cast<quokka::CICRadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			container->Redistribute(lev, container->finestLevel(), ngrow);
		}
	}

	void writePlotFile(const std::string &plotfilename, const std::string &name) override
	{
		if (auto *container = dynamic_cast<quokka::CICRadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			container->WritePlotFile(plotfilename, name);
		}
	}

	void writeCheckpoint(const std::string &checkpointname, const std::string &name, bool include_header) override
	{
		if (auto *container = dynamic_cast<quokka::CICRadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			container->Checkpoint(checkpointname, name, include_header);
		}
	}

	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time, int lumIndex, int birthTimeIndex, int nGroups) override
	{
		if (auto *container = dynamic_cast<quokka::CICRadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			amrex::ParticleToMesh(*container, radEnergySource, lev, RadDeposition{current_time, lumIndex, 0, nGroups, birthTimeIndex}, false);
		}
	}

	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst, int massIndex) override
	{
		if (auto *container = dynamic_cast<quokka::CICRadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			amrex::ParticleToMesh(*container, amrex::GetVecOfPtrs(rhs), 0, finest_lev, MassDeposition{Gconst, massIndex, 0, 1}, true);
		}
	}

	void driftParticlesAllLevels(amrex::Real dt) override
	{
		if (auto *container = dynamic_cast<quokka::CICRadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			for (int lev = 0; lev <= container->finestLevel(); ++lev) {
				for (quokka::CICRadParticleIterator<problem_t> pIter(*container, lev); pIter.isValid(); ++pIter) {
					auto &particles = pIter.GetArrayOfStructs();
					typename quokka::CICRadParticleContainer<problem_t>::ParticleType *pData = particles().data();
					const amrex::Long np = pIter.numParticles();

					amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
						typename quokka::CICRadParticleContainer<problem_t>::ParticleType &p =
						    pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
						// update particle position
						for (int i = 0; i < AMREX_SPACEDIM; ++i) {
							p.pos(i) += dt * p.rdata(quokka::CICRadParticleVxIdx + i);
						}
					});
				}
			}
		}
	}

	void kickParticles(amrex::Real dt, const amrex::MultiFab &accel, const amrex::Vector<amrex::Geometry> &geom, int lev) override
	{
		if (auto *container = dynamic_cast<quokka::CICRadParticleContainer<problem_t> *>(this->neighborParticleContainer_)) {
			const auto dx_inv = geom[lev].InvCellSizeArray();
			for (quokka::CICRadParticleIterator<problem_t> pIter(*container, lev); pIter.isValid(); ++pIter) {
				auto &particles = pIter.GetArrayOfStructs();
				typename quokka::CICRadParticleContainer<problem_t>::ParticleType *pData = particles().data();
				const amrex::Long np = pIter.numParticles();

				amrex::Array4<const amrex::Real> const &accel_arr = accel.array(pIter);
				const auto plo = geom[lev].ProbLoArray();

				amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int64_t idx) {
					typename quokka::CICRadParticleContainer<problem_t>::ParticleType &p =
					    pData[idx]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
					amrex::ParticleInterpolator::Linear interp(p, plo, dx_inv);
					interp.MeshToParticle(
					    p, accel_arr, 0, quokka::CICRadParticleVxIdx, AMREX_SPACEDIM,
					    [=] AMREX_GPU_DEVICE(amrex::Array4<const amrex::Real> const &acc, int i, int j, int k, int comp) {
						    return acc(i, j, k, comp); // no weighting
					    },
					    [=] AMREX_GPU_DEVICE(typename quokka::CICRadParticleContainer<problem_t>::ParticleType & p, int comp,
								 amrex::Real acc_comp) {
						    // kick particle by updating its velocity
						    p.rdata(comp) += 0.5 * dt * static_cast<amrex::ParticleReal>(acc_comp);
					    });
				});
			}
		}
	}
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
		// Check if name already exists
		if (particleRegistry_.find(name) != particleRegistry_.end()) {
			amrex::Abort("Particle type '" + name + "' already registered!");
		}
		particleRegistry_.emplace(name, std::move(descriptor));
	}

	// Get particle descriptor by name (returns nullptr if not found)
	[[nodiscard]] auto getParticleDescriptor(const std::string &name) const -> PhysicsParticleDescriptor *
	{
		auto it = particleRegistry_.find(name);
		if (it != particleRegistry_.end()) {
			return it->second.get();
		}
		return nullptr;
	}

	// Run WritePlotFile(plotfilename, name) on all particles
	void writePlotFile(const std::string &plotfilename) const
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->writePlotFile(plotfilename, name);
		}
	}

	// Run Checkpoint(checkpointname, name, true) on all particles
	void writeCheckpoint(const std::string &checkpointname, bool include_header) const
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->writeCheckpoint(checkpointname, name, include_header);
		}
	}

	// Deposit radiation from all particles that have luminosity
	void depositRadiation(amrex::MultiFab &radEnergySource, int lev, amrex::Real current_time) const
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getLumIndex() >= 0) {
				descriptor->depositRadiation(radEnergySource, lev, current_time, descriptor->getLumIndex(), descriptor->getBirthTimeIndex(),
							     Physics_Traits<problem_t>::nGroups);
			}
		}
	}

	// Deposit mass from all particles that have mass for gravity calculation
	void depositMass(amrex::Vector<amrex::MultiFab> &rhs, int finest_lev, amrex::Real Gconst) const
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			if (descriptor->getMassIndex() >= 0) {
				descriptor->depositMass(rhs, finest_lev, Gconst, descriptor->getMassIndex());
			}
		}
	}

	// Run Redistribute(lev) on all particles
	void redistribute(int lev)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev);
		}
	}

	// Run Redistribute(lev, ngrow) on all particles
	void redistribute(int lev, int ngrow)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->redistribute(lev, ngrow);
		}
	}

	// Run driftParticles(dt) on all particles
	void driftParticlesAllLevels(amrex::Real dt)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->driftParticlesAllLevels(dt);
		}
	}

	// Run kickParticles(dt) on all particles
	void kickParticles(amrex::Real dt, const amrex::MultiFab &accel, const amrex::Vector<amrex::Geometry> &geom, int lev)
	{
		for (const auto &[name, descriptor] : particleRegistry_) {
			descriptor->kickParticles(dt, accel, geom, lev);
		}
	}

	// Delete copy/move constructors/assignments
	PhysicsParticleRegister(const PhysicsParticleRegister &) = delete;
	PhysicsParticleRegister &operator=(const PhysicsParticleRegister &) = delete;
	PhysicsParticleRegister(PhysicsParticleRegister &&) = delete;
	PhysicsParticleRegister &operator=(PhysicsParticleRegister &&) = delete;
};

} // namespace quokka

#endif // PHYSICS_PARTICLES_HPP_
