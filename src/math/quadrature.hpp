#ifndef QUADRATURE_HPP_
#define QUADRATURE_HPP_

// system headers
#include <cmath>

// library headers
#include <AMReX.H>
#include <AMReX_REAL.H>

#include "gauss.hpp"

AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto kernel_wendland_c2(const amrex::Real r) -> amrex::Real
{
	amrex::Real val = NAN;
	if (r > 1.0) {
		val = 0;
	} else {
		val = (21. / (2. * M_PI)) * std::pow((1.0 - r), 4) * (4.0 * r + 1.0);
	}
	return val;
}

template <typename F>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_3d(F &&f, amrex::Real x0, amrex::Real x1, amrex::Real y0, amrex::Real y1, amrex::Real z0, amrex::Real z1)
    -> amrex::Real
{
	// integrate F over the rectangular domain [x0, y0, z0] -> [x1, y1, z1].
	auto integrand = [=] AMREX_GPU_DEVICE(amrex::Real z) {
		return quad_2d([=] AMREX_GPU_DEVICE(amrex::Real x, amrex::Real y) { return f(x, y, z); }, x0, x1, y0, y1);
	};
	return quad_1d(integrand, z0, z1);
}

template <typename F> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_2d(F &&f, amrex::Real x0, amrex::Real x1, amrex::Real y0, amrex::Real y1) -> amrex::Real
{
	// integrate F over the rectangular domain [x0, y0] -> [x1, y1].
	auto integrand = [=] AMREX_GPU_DEVICE(amrex::Real y) { return quad_1d([=] AMREX_GPU_DEVICE(amrex::Real x) { return f(x, y); }, x0, x1); };
	return quad_1d(integrand, y0, y1);
}

template <typename F> AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto quad_1d(F &&f, amrex::Real x0, amrex::Real x1) -> amrex::Real
{
	// integrate F over the rectangular domain [x0] -> [x1].
	// use 7-point Gauss-Legendre quadrature
	return quokka::math::quadrature::gauss<double, 7>::integrate(f, x0, x1);
}

#endif // QUADRATURE_HPP_