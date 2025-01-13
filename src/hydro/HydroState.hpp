#ifndef HYDROSTATE_HPP_ // NOLINT
#define HYDROSTATE_HPP_

#include "AMReX_Array.H"

namespace quokka
{
template <int Nall, int Nmass> struct HydroState {
	double rho;				   // density
	double u;				   // normal velocity component
	double v;				   // transverse velocity component
	double w;				   // 2nd transverse velocity component
	double P;				   // pressure
	double cs;				   // adiabatic sound speed
	double E;				   // total energy density
	double Eint;				   // internal energy density
	double by;				   // transverse bfield component
	double bz;				   // 2nd transverse bfield density
	amrex::GpuArray<double, Nall> scalar;	   // passive scalars
	amrex::GpuArray<double, Nmass> massScalar; // mass scalars
};

} // namespace quokka

#endif // HYDROSTATE_HPP_
