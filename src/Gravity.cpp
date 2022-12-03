#include "Gravity.hpp"

///
/// Multipole gravity data
///

AMREX_GPU_MANAGED Real multipole::rmax;

AMREX_GPU_MANAGED amrex::Array2D<Real, 0, multipole::lnum_max, 0, multipole::lnum_max> multipole::factArray;
AMREX_GPU_MANAGED amrex::Array1D<Real, 0, multipole::lnum_max> multipole::parity_q0;
AMREX_GPU_MANAGED amrex::Array2D<Real, 0, multipole::lnum_max, 0, multipole::lnum_max> multipole::parity_qC_qS;
