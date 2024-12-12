#ifndef PHYSICS_PARTICLES_HPP_
#define PHYSICS_PARTICLES_HPP_

#include <map>
#include <memory>
#include <string>
#include <AMReX_AmrParticles.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParticleInterpolators.H>
#include "physics_info.hpp"

namespace quokka
{

enum RadParticleDataIdx { RadParticleMassIdx = 0, RadParticleBirthTimeIdx, RadParticleDeathTimeIdx, RadParticleLumIdx };
template <typename problem_t> constexpr int RadParticleRealComps = 3 + Physics_Traits<problem_t>::nGroups;

template <typename problem_t> using RadParticleContainer = amrex::AmrParticleContainer<RadParticleRealComps<problem_t>>;
template <typename problem_t> using RadParticleIterator = amrex::ParIter<RadParticleRealComps<problem_t>>;

template <typename problem_t> struct RadDeposition {
	double current_time{};
	int start_part_comp{};
	int start_mesh_comp{};
	int num_comp{};

	AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const typename RadParticleContainer<problem_t>::ParticleType &p,
							    amrex::Array4<amrex::Real> const &radEnergySource,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
							    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
	{
		amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
		interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp,
				      [=] AMREX_GPU_DEVICE(const typename RadParticleContainer<problem_t>::ParticleType &part, int comp) {
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
class PhysicsParticleDescriptor {
protected:
    int massIndex_{-1};              // index for gravity mass, -1 if not used
    int lumIndex_{-1};               // index for radiation luminosity, -1 if not used  
    bool interactsWithHydro_{false}; // whether particles interact with hydro

public:
    PhysicsParticleDescriptor() = default;
    virtual ~PhysicsParticleDescriptor() = default;
    void* neighborParticleContainer_{}; // pointer to particle container, type-erased

    // Getters
    [[nodiscard]] auto getMassIndex() const -> int { return massIndex_; }
    [[nodiscard]] auto getLumIndex() const -> int { return lumIndex_; }
    [[nodiscard]] auto getInteractsWithHydro() const -> bool { return interactsWithHydro_; }
    [[nodiscard]] auto getParticleContainer() const -> void* { return neighborParticleContainer_; }

    // Set the luminosity index for radiation particles
    void setLumIndex(int index) noexcept { lumIndex_ = index; }

    // Virtual methods that can be overridden
    virtual void hydroInteract() {} // Default no-op
    
    // Delete copy/move constructors/assignments
    PhysicsParticleDescriptor(const PhysicsParticleDescriptor&) = delete;
    PhysicsParticleDescriptor& operator=(const PhysicsParticleDescriptor&) = delete;
    PhysicsParticleDescriptor(PhysicsParticleDescriptor&&) = delete;
    PhysicsParticleDescriptor& operator=(PhysicsParticleDescriptor&&) = delete;
};

// Registry for physics particles
template <typename problem_t>
class PhysicsParticleRegister {
private:
    std::map<std::string, std::unique_ptr<PhysicsParticleDescriptor>> particleRegistry_;

public:
    PhysicsParticleRegister() = default;
    ~PhysicsParticleRegister() = default;

    // Register a new particle type
    template <typename ParticleDescriptor>
    void registerParticleType(const std::string& name, std::unique_ptr<ParticleDescriptor> descriptor) {
        particleRegistry_[name] = std::move(descriptor);
    }

    // Get a particle descriptor
    [[nodiscard]] auto getParticleDescriptor(const std::string& name) const -> const PhysicsParticleDescriptor* {
        auto it = particleRegistry_.find(name);
        if (it != particleRegistry_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    // // Deposit gravity from all particles that have mass
    // void depositGravity(amrex::MultiFab& rhs, int lev) {
    //     for (const auto& [name, descriptor] : particleRegistry_) {
    //         if (descriptor->getMassIndex() >= 0) {
    //             // Get particle container and deposit mass
    //             auto* container = static_cast<amrex::AmrParticleContainer<RadParticleRealComps<problem_t>>*>(descriptor->getParticleContainer());
    //             if (container != nullptr) {
    //                 amrex::ParticleToMesh(*container, amrex::GetVecOfPtrs(rhs), 0, lev,
    //                                      [=] AMREX_GPU_DEVICE(const typename amrex::AmrParticleContainer<RadParticleRealComps<problem_t>>::ParticleType& p,
    //                                                         amrex::Array4<amrex::Real> const& rhs_arr,
    //                                                         const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>& plo,
    //                                                         const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>& /*dxi*/) {
    //                                          // Deposit mass to grid
    //                                          return p.rdata(descriptor->getMassIndex());
    //                                      });
    //             }
    //         }
    //     }
    // }

		struct RadDeposition {
			double current_time{};
			int start_part_comp{};
			int start_mesh_comp{};
			int num_comp{};

			AMREX_GPU_DEVICE AMREX_FORCE_INLINE void operator()(const typename RadParticleContainer<problem_t>::ParticleType &p,
											amrex::Array4<amrex::Real> const &radEnergySource,
											amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &plo,
											amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dxi) const noexcept
			{
				amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
				interp.ParticleToMesh(p, radEnergySource, start_part_comp, start_mesh_comp, num_comp,
									[=] AMREX_GPU_DEVICE(const typename RadParticleContainer<problem_t>::ParticleType &part, int comp) {
										if (current_time < part.rdata(RadParticleBirthTimeIdx) || current_time >= part.rdata(RadParticleDeathTimeIdx)) {
											return 0.0;
										}
										return part.rdata(comp) * (AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2]));
									});
			}
		};

    // Deposit radiation from all particles that have luminosity
    void depositRadiation(amrex::MultiFab& radEnergySource, int lev, amrex::Real current_time) {
        for (const auto& [name, descriptor] : particleRegistry_) {
            if (descriptor->getLumIndex() >= 0) {
                // Get particle container and deposit luminosity
                auto* container = static_cast<RadParticleContainer<problem_t>*>(descriptor->getParticleContainer());
                if (container != nullptr) {
                    amrex::ParticleToMesh(*container, radEnergySource, lev,
										 	RadDeposition{current_time, descriptor->getLumIndex(), 0, Physics_Traits<problem_t>::nGroups}, false);
                }
            }
        }
    }

    // Delete copy/move constructors/assignments
    PhysicsParticleRegister(const PhysicsParticleRegister&) = delete;
    PhysicsParticleRegister& operator=(const PhysicsParticleRegister&) = delete; 
    PhysicsParticleRegister(PhysicsParticleRegister&&) = delete;
    PhysicsParticleRegister& operator=(PhysicsParticleRegister&&) = delete;
};

} // namespace quokka

#endif // PHYSICS_PARTICLES_HPP_
