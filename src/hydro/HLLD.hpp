#ifndef HLLD_HPP_ // NOLINT
#define HLLD_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>

#include "hydro/HydroState.hpp"
#include "util/ArrayView.hpp"
#include "util/valarray.hpp"

namespace quokka::Riemann
{
constexpr double DELTA = 1.0e-4;

template <class T> constexpr auto SQUARE(const T x) -> T { return x * x; }

// density, momentum, total energy, transverse magnetic field
struct ConsHydro1D {
	double rho; // density
	double mx;  // x-momentum
	double my;  // y-momentum
	double mz;  // z-momentum
	double E;   // total energy density
	double by;  // y-magnetic field
	double bz;  // z-magnetic field
};

template <int N_scalars, int N_mscalars>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto FastMagnetoSonicSpeed(double gamma, quokka::HydroState<N_scalars, N_mscalars> const state, const double bx) -> double
{
	double gp = gamma * state.P;
	double bx_sq = bx * bx;
	double byz_sq = state.by * state.by + state.bz * state.bz;
	double b_sq = bx_sq + byz_sq;
	double bgp_p = b_sq + gp;
	double bgp_m = b_sq - gp;
	return std::sqrt(0.5 * (bgp_p + std::sqrt(bgp_m * bgp_m + 4.0 * gp * byz_sq)) / state.rho);
}

// HLLD solver following Miyoshi and Kusano (2005), hereafter MK5.
template <typename problem_t, int N_scalars, int N_mscalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLD(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR,
					      const double gamma, const double bx) -> quokka::valarray<double, fluxdim>
{
	//--- Step 1. Compute L/R states

	// initialize left and right conserved states
	ConsHydro1D u_L{};
	ConsHydro1D u_R{};
	// initialize temporary container to store flux across interface
	ConsHydro1D f_x{};
	// initialize fluxes at left and right side of the interface
	ConsHydro1D f_L{};
	ConsHydro1D f_R{};
	// initialise signal speeds (left to right)
	std::array<double, 5> spds{};
	// initialise four intermediate conserved states
	ConsHydro1D u_star_L{};
	ConsHydro1D u_dstar_L{};
	ConsHydro1D u_dstar_R{};
	ConsHydro1D u_star_R{};

	// frequently used term
	double const bx_sq = SQUARE(bx);

	// compute L/R states for select conserved variables
	// (group transverse vector components for floating-point associativity symmetry)
	// magnetic pressure
	double const pb_L = 0.5 * (bx_sq + (SQUARE(sL.by) + SQUARE(sL.bz)));
	double const pb_R = 0.5 * (bx_sq + (SQUARE(sR.by) + SQUARE(sR.bz)));
	// kinetic energy
	double const ke_L = 0.5 * sL.rho * (SQUARE(sL.u) + (SQUARE(sL.v) + SQUARE(sL.w)));
	double const ke_R = 0.5 * sR.rho * (SQUARE(sR.u) + (SQUARE(sR.v) + SQUARE(sR.w)));
	// set left conserved states
	u_L.rho = sL.rho;
	u_L.mx = sL.u * u_L.rho;
	u_L.my = sL.v * u_L.rho;
	u_L.mz = sL.w * u_L.rho;
	u_L.E = ke_L + pb_L + sL.P / (gamma - 1.0); // TODO(neco): generalise EOS
	u_L.by = sL.by;
	u_L.bz = sL.bz;
	// set right conserved states
	u_R.rho = sR.rho;
	u_R.mx = sR.u * u_R.rho;
	u_R.my = sR.v * u_R.rho;
	u_R.mz = sR.w * u_R.rho;
	u_R.E = ke_R + pb_R + sR.P / (gamma - 1.0);
	u_R.by = sR.by;
	u_R.bz = sR.bz;

	//--- Step 2. Compute L & R wave speeds according to MK5, eqn. (67)

	const double cfs_L = FastMagnetoSonicSpeed(gamma, sL, bx);
	const double cfs_R = FastMagnetoSonicSpeed(gamma, sR, bx);
	spds[0] = std::min(sL.u - cfs_L, sR.u - cfs_R);
	spds[4] = std::max(sL.u + cfs_L, sR.u + cfs_R);

	//--- Step 3. Compute L/R fluxes

	// total pressure
	double ptot_L = sL.P + pb_L;
	double ptot_R = sR.P + pb_R;
	// fluxes on the left side of the interface
	f_L.rho = u_L.mx;
	f_L.mx = u_L.mx * sL.u + ptot_L - bx_sq;
	f_L.my = u_L.my * sL.u + bx * u_L.by;
	f_L.mz = u_L.mz * sL.u + bx * u_L.bz;
	f_L.E = sL.u * (u_L.E + ptot_L - bx_sq) - bx * (sL.v * u_L.by + sL.w * u_L.bz);
	f_L.by = u_L.by * sL.u - bx * sL.v;
	f_L.bz = u_L.bz * sL.u - bx * sL.w;
	// fluxes on the right side of the interface
	f_R.rho = u_R.mx;
	f_R.mx = u_R.mx * sR.u + ptot_R - bx_sq;
	f_R.my = u_R.my * sR.u + bx * u_R.by;
	f_R.mz = u_R.mz * sR.u + bx * u_R.bz;
	f_R.E = sR.u * (u_R.E + ptot_R - bx_sq) - bx * (sR.v * u_R.by + sR.w * u_R.bz);
	f_R.by = u_R.by * sR.u - bx * sR.v;
	f_R.bz = u_R.bz * sR.u - bx * sR.w;

	//--- Step 4. Compute middle and Alfven wave speeds

	// MK5: S_i - u_i (for i=L or R)
	double siui_L = spds[0] - sL.u;
	double siui_R = spds[4] - sR.u;
	// MK5: S_M from eqn (38)
	// group ptot terms for floating-point associativity symmetry
	spds[2] = (siui_R * u_R.mx - siui_L * u_L.mx + (ptot_L - ptot_R)) / (siui_R * u_R.rho - siui_L * u_L.rho);
	// S_i - S_M (for i=L or R)
	double sism_L = spds[0] - spds[2];
	double sism_R = spds[4] - spds[2];
	double sism_inv_L = 1.0 / sism_L;
	double sism_inv_R = 1.0 / sism_R;
	// MK5: rho_i from eqn (43)
	u_star_L.rho = u_L.rho * siui_L * sism_inv_L;
	u_star_R.rho = u_R.rho * siui_R * sism_inv_R;
	double u_star_rho_inv_L = 1.0 / u_star_L.rho;
	double u_star_rho_inv_R = 1.0 / u_star_R.rho;
	double rho_sqrt_L = std::sqrt(u_star_L.rho);
	double rho_sqrt_R = std::sqrt(u_star_R.rho);
	// MK5: eqn (51)
	spds[1] = spds[2] - std::abs(bx) / rho_sqrt_L;
	spds[3] = spds[2] + std::abs(bx) / rho_sqrt_R;

	//--- Step 5. Compute intermediate states

	// compute total pressure
	// MK5: eqn (41) can be calculated (more explicitly) via eqn (23)
	double ptot_star_L = ptot_L - u_L.rho * siui_L * (spds[2] - sL.u);
	double ptot_star_R = ptot_R - u_R.rho * siui_R * (spds[2] - sR.u);
	double ptot_star = 0.5 * (ptot_star_L + ptot_star_R);

	// MK5: u_L^(star, dstar) from, eqn (39)
	u_star_L.mx = u_star_L.rho * spds[2];
	if (std::abs(u_L.rho * siui_L * sism_L - bx_sq) < (DELTA)*ptot_star) {
		// degenerate case
		u_star_L.my = u_star_L.rho * sL.v;
		u_star_L.mz = u_star_L.rho * sL.w;
		u_star_L.by = u_L.by;
		u_star_L.bz = u_L.bz;
	} else {
		// MK5: eqns (44) and (46)
		double tmp = bx * (siui_L - sism_L) / (u_L.rho * siui_L * sism_L - bx_sq);
		u_star_L.my = u_star_L.rho * (sL.v - u_L.by * tmp);
		u_star_L.mz = u_star_L.rho * (sL.w - u_L.bz * tmp);
		// MK5: eqns (45) and (47)
		tmp = (u_L.rho * SQUARE(siui_L) - bx_sq) / (u_L.rho * siui_L * sism_L - bx_sq);
		u_star_L.by = u_L.by * tmp;
		u_star_L.bz = u_L.bz * tmp;
	}
	// vec(v_L^star) dot vec(b_L^star)
	// group transverse momenta-components for floating-point associativity
	double vb_star_L = (u_star_L.mx * bx + (u_star_L.my * u_star_L.by + u_star_L.mz * u_star_L.bz)) * u_star_rho_inv_L;
	// MK5: eqn (48)
	u_star_L.E = (siui_L * u_L.E - ptot_L * sL.u + ptot_star * spds[2] + bx * (sL.u * bx + (sL.v * u_L.by + sL.w * u_L.bz) - vb_star_L)) * sism_inv_L;

	// MK5: u_R^(star, dstar) from, eqn (39)
	u_star_R.mx = u_star_R.rho * spds[2];
	if (std::abs(u_R.rho * siui_R * sism_R - bx_sq) < (DELTA)*ptot_star) {
		// degenerate case
		u_star_R.my = u_star_R.rho * sR.v;
		u_star_R.mz = u_star_R.rho * sR.w;
		u_star_R.by = u_R.by;
		u_star_R.bz = u_R.bz;
	} else {
		// MK5: eqns (44) and (46)
		double tmp = bx * (siui_R - sism_R) / (u_R.rho * siui_R * sism_R - bx_sq);
		u_star_R.my = u_star_R.rho * (sR.v - u_R.by * tmp);
		u_star_R.mz = u_star_R.rho * (sR.w - u_R.bz * tmp);
		// MK5: eqns (45) and (47)
		tmp = (u_R.rho * SQUARE(siui_R) - bx_sq) / (u_R.rho * siui_R * sism_R - bx_sq);
		u_star_R.by = u_R.by * tmp;
		u_star_R.bz = u_R.bz * tmp;
	}
	// vec(v_R^star) dot vec(b_R^star)
	// group transverse momenta-components for floating-point associativity
	double vb_star_R = (u_star_R.mx * bx + (u_star_R.my * u_star_R.by + u_star_R.mz * u_star_R.bz)) * u_star_rho_inv_R;
	// MK5: eqn (48)
	u_star_R.E = (siui_R * u_R.E - ptot_R * sR.u + ptot_star * spds[2] + bx * (sR.u * bx + (sR.v * u_R.by + sR.w * u_R.bz) - vb_star_R)) * sism_inv_R;

	// if Bx is near zero, then u_i^dstar = u_i^star
	if (0.5 * bx_sq < (DELTA)*ptot_star) {
		u_dstar_L = u_star_L;
		u_dstar_R = u_star_R;
	} else {
		double rho_sum_inv = 1.0 / (rho_sqrt_L + rho_sqrt_R);
		double bx_sign = (bx > 0.0 ? 1.0 : -1.0);
		u_dstar_L.rho = u_star_L.rho;
		u_dstar_R.rho = u_star_R.rho;
		u_dstar_L.mx = u_star_L.mx;
		u_dstar_R.mx = u_star_R.mx;
		// MK5: eqn (59)
		double tmp = rho_sum_inv * (rho_sqrt_L * (u_star_L.my * u_star_rho_inv_L) + rho_sqrt_R * (u_star_R.my * u_star_rho_inv_R) +
					    bx_sign * (u_star_R.by - u_star_L.by));
		u_dstar_L.my = u_dstar_L.rho * tmp;
		u_dstar_R.my = u_dstar_R.rho * tmp;
		// MK5: eqn (60)
		tmp = rho_sum_inv *
		      (rho_sqrt_L * (u_star_L.mz * u_star_rho_inv_L) + rho_sqrt_R * (u_star_R.mz * u_star_rho_inv_R) + bx_sign * (u_star_R.bz - u_star_L.bz));
		u_dstar_L.mz = u_dstar_L.rho * tmp;
		u_dstar_R.mz = u_dstar_R.rho * tmp;
		// MK5: eqn (61)
		tmp = rho_sum_inv * (rho_sqrt_L * u_star_R.by + rho_sqrt_R * u_star_L.by +
				     bx_sign * rho_sqrt_L * rho_sqrt_R * ((u_star_R.my * u_star_rho_inv_R) - (u_star_L.my * u_star_rho_inv_L)));
		u_dstar_L.by = tmp;
		u_dstar_R.by = tmp;
		// MK5: eqn (62)
		tmp = rho_sum_inv * (rho_sqrt_L * u_star_R.bz + rho_sqrt_R * u_star_L.bz +
				     bx_sign * rho_sqrt_L * rho_sqrt_R * ((u_star_R.mz * u_star_rho_inv_R) - (u_star_L.mz * u_star_rho_inv_L)));
		u_dstar_L.bz = tmp;
		u_dstar_R.bz = tmp;
		// MK5: eqn (63)
		tmp = spds[2] * bx + (u_dstar_L.my * u_dstar_L.by + u_dstar_L.mz * u_dstar_L.bz) / u_dstar_L.rho;
		u_dstar_L.E = u_star_L.E - rho_sqrt_L * bx_sign * (vb_star_L - tmp);
		u_dstar_R.E = u_star_R.E + rho_sqrt_R * bx_sign * (vb_star_R - tmp);
	}

	//--- Step 6. Compute fluxes

	u_dstar_L.rho = spds[1] * (u_dstar_L.rho - u_star_L.rho);
	u_dstar_L.mx = spds[1] * (u_dstar_L.mx - u_star_L.mx);
	u_dstar_L.my = spds[1] * (u_dstar_L.my - u_star_L.my);
	u_dstar_L.mz = spds[1] * (u_dstar_L.mz - u_star_L.mz);
	u_dstar_L.E = spds[1] * (u_dstar_L.E - u_star_L.E);
	u_dstar_L.by = spds[1] * (u_dstar_L.by - u_star_L.by);
	u_dstar_L.bz = spds[1] * (u_dstar_L.bz - u_star_L.bz);

	u_star_L.rho = spds[0] * (u_star_L.rho - u_L.rho);
	u_star_L.mx = spds[0] * (u_star_L.mx - u_L.mx);
	u_star_L.my = spds[0] * (u_star_L.my - u_L.my);
	u_star_L.mz = spds[0] * (u_star_L.mz - u_L.mz);
	u_star_L.E = spds[0] * (u_star_L.E - u_L.E);
	u_star_L.by = spds[0] * (u_star_L.by - u_L.by);
	u_star_L.bz = spds[0] * (u_star_L.bz - u_L.bz);

	u_dstar_R.rho = spds[3] * (u_dstar_R.rho - u_star_R.rho);
	u_dstar_R.mx = spds[3] * (u_dstar_R.mx - u_star_R.mx);
	u_dstar_R.my = spds[3] * (u_dstar_R.my - u_star_R.my);
	u_dstar_R.mz = spds[3] * (u_dstar_R.mz - u_star_R.mz);
	u_dstar_R.E = spds[3] * (u_dstar_R.E - u_star_R.E);
	u_dstar_R.by = spds[3] * (u_dstar_R.by - u_star_R.by);
	u_dstar_R.bz = spds[3] * (u_dstar_R.bz - u_star_R.bz);

	u_star_R.rho = spds[4] * (u_star_R.rho - u_R.rho);
	u_star_R.mx = spds[4] * (u_star_R.mx - u_R.mx);
	u_star_R.my = spds[4] * (u_star_R.my - u_R.my);
	u_star_R.mz = spds[4] * (u_star_R.mz - u_R.mz);
	u_star_R.E = spds[4] * (u_star_R.E - u_R.E);
	u_star_R.by = spds[4] * (u_star_R.by - u_R.by);
	u_star_R.bz = spds[4] * (u_star_R.bz - u_R.bz);

	if (spds[0] >= 0.0) {
		// return u_L if flow is supersonic
		f_x.rho = f_L.rho;
		f_x.mx = f_L.mx;
		f_x.my = f_L.my;
		f_x.mz = f_L.mz;
		f_x.E = f_L.E;
		f_x.by = f_L.by;
		f_x.bz = f_L.bz;
	} else if (spds[4] <= 0.0) {
		// return u_R if flow is supersonic
		f_x.rho = f_R.rho;
		f_x.mx = f_R.mx;
		f_x.my = f_R.my;
		f_x.mz = f_R.mz;
		f_x.E = f_R.E;
		f_x.by = f_R.by;
		f_x.bz = f_R.bz;
	} else if (spds[1] >= 0.0) {
		// return u_star_L
		f_x.rho = f_L.rho + u_star_L.rho;
		f_x.mx = f_L.mx + u_star_L.mx;
		f_x.my = f_L.my + u_star_L.my;
		f_x.mz = f_L.mz + u_star_L.mz;
		f_x.E = f_L.E + u_star_L.E;
		f_x.by = f_L.by + u_star_L.by;
		f_x.bz = f_L.bz + u_star_L.bz;
	} else if (spds[2] >= 0.0) {
		// return u_dstar_L
		f_x.rho = f_L.rho + u_star_L.rho + u_dstar_L.rho;
		f_x.mx = f_L.mx + u_star_L.mx + u_dstar_L.mx;
		f_x.my = f_L.my + u_star_L.my + u_dstar_L.my;
		f_x.mz = f_L.mz + u_star_L.mz + u_dstar_L.mz;
		f_x.E = f_L.E + u_star_L.E + u_dstar_L.E;
		f_x.by = f_L.by + u_star_L.by + u_dstar_L.by;
		f_x.bz = f_L.bz + u_star_L.bz + u_dstar_L.bz;
	} else if (spds[3] > 0.0) {
		// return u_dstar_R
		f_x.rho = f_R.rho + u_star_R.rho + u_dstar_R.rho;
		f_x.mx = f_R.mx + u_star_R.mx + u_dstar_R.mx;
		f_x.my = f_R.my + u_star_R.my + u_dstar_R.my;
		f_x.mz = f_R.mz + u_star_R.mz + u_dstar_R.mz;
		f_x.E = f_R.E + u_star_R.E + u_dstar_R.E;
		f_x.by = f_R.by + u_star_R.by + u_dstar_R.by;
		f_x.bz = f_R.bz + u_star_R.bz + u_dstar_R.bz;
	} else {
		// return u_star_R
		f_x.rho = f_R.rho + u_star_R.rho;
		f_x.mx = f_R.mx + u_star_R.mx;
		f_x.my = f_R.my + u_star_R.my;
		f_x.mz = f_R.mz + u_star_R.mz;
		f_x.E = f_R.E + u_star_R.E;
		f_x.by = f_R.by + u_star_R.by;
		f_x.bz = f_R.bz + u_star_R.bz;
	}

	// TODO(neco): Eint=0 for now; pscalars will also be needed in the future.
	quokka::valarray<double, fluxdim> F_hydro = {f_x.rho, f_x.mx, f_x.my, f_x.mz, f_x.E, 0.0};
	return F_hydro;
}
} // namespace quokka::Riemann

#endif // HLLD_HPP_