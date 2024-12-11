#ifndef INTERPOLATE2D_HPP_ // NOLINT
#define INTERPOLATE2D_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file Interpolate2D.cpp
/// \brief Defines methods for interpolating 2D table data using bilinear
/// interpolation.
///

#include "AMReX_BLassert.H"
#include "AMReX_TableData.H"

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate2d(double x, double y, amrex::Table1D<const double> const &xv, amrex::Table1D<const double> const &yv,
							    amrex::Table2D<const double> const &table) -> double
{
	// return the bilinearly-interpolated value in table
	// NOTE: table must be uniformly spaced in xv and yv
	double xi = xv(xv.begin);
	double xf = xv(xv.end - 1);
	double yi = yv(yv.begin);
	double yf = yv(yv.end - 1);

	double dx = (xf - xi) / static_cast<double>(xv.end - xv.begin - 1);
	double dy = (yf - yi) / static_cast<double>(yv.end - yv.begin - 1);

	x = amrex::Clamp(x, xi, xf);
	y = amrex::Clamp(y, yi, yf);

	// compute indices
	int ix = amrex::Clamp(static_cast<int>(std::floor((x - xi) / dx)), xv.begin, xv.end - 1);
	int iy = amrex::Clamp(static_cast<int>(std::floor((y - yi) / dy)), yv.begin, yv.end - 1);
	int iix = (ix == xv.end - 1) ? ix : ix + 1;
	int iiy = (iy == yv.end - 1) ? iy : iy + 1;

	// get values
	double x1 = xv(ix);
	double x2 = xv(iix);
	double y1 = yv(iy);
	double y2 = yv(iiy);

	// compute weights
	double w11 = 0;
	double w12 = 0;
	double w21 = 0;
	double w22 = 0;

	if (ix != iix && iy != iiy) {
		const double vol = ((x2 - x1) * (y2 - y1));
		AMREX_ASSERT(vol > 0.);
		w11 = (x2 - x) * (y2 - y) / vol;
		w12 = (x2 - x) * (y - y1) / vol;
		w21 = (x - x1) * (y2 - y) / vol;
		w22 = (x - x1) * (y - y1) / vol;
	} else if (ix == iix && yi != iiy) {
		const double vol = (y2 - y1);
		AMREX_ASSERT(vol > 0.);
		w11 = (y2 - y) / vol;
		w12 = (y - y1) / vol;
	} else if (ix != iix && yi == iiy) {
		const double vol = (x2 - x1);
		AMREX_ASSERT(vol > 0.);
		w11 = (x2 - x) / vol;
		w21 = (x - x1) / vol;
	} else { // ix == iix && yi == iiy
		w11 = 1.0;
	}

	double A = table(ix, iy);
	double B = table(ix, iiy);
	double C = table(iix, iy);
	double D = table(iix, iiy);

	double value = w11 * A + w12 * B + w21 * C + w22 * D;
	AMREX_ASSERT(!std::isnan(value));

	return value;
}

#endif // INTERPOLATE2D_HPP_
