#ifndef VALARRAY_HPP_
#define VALARRAY_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file valarray.hpp
/// \brief A container for a vector with addition, multiplication with expression templates
/// (This is necessary because std::valarray is not defined in CUDA C++!)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>

// library headers
#include "AMReX_Extension.H"
#include <AMReX_GpuQualifiers.H>

namespace quokka
{
template <typename T, int d> class valarray
{
      public:
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray() = default;

	// we *want* implicit construction from initializer lists for valarrays,
	// (although not cppcore-compliant)
	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE valarray(std::initializer_list<T> list) // NOLINT
	{
		const size_t max_count = std::min(list.size(), static_cast<size_t>(d));

		T const *input = std::data(list); // requires nvcc to be in C++17 mode! (if it fails, the
						  // compiler flags are wrong, probably due to a CMake issue.)

		for (size_t i = 0; i < max_count; ++i) {
			values[i] = input[i]; // NOLINT
		}

		// it is undefined behavior to not fully initialize an object!
		// (this does happen in practice with gcc 10+, which optimizes out ctor
		//  calls if an object is unused before a subsequent assignment.)
		for (size_t i = max_count; i < d; ++i) {
			values[i] = default_value;
		}
	}

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) -> T & { return values[i]; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator[](size_t i) const -> T { return values[i]; }

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto size() const -> size_t { return d; }

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void fillin(T const &scalar)
	{
		for (size_t i = 0; i < d; ++i) {
			values[i] = scalar;
		}
	}

	[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto hasnan() const -> bool
	{
		for (size_t i = 0; i < d; ++i) {
			if (std::isnan(values[i])) {
				return true;
			}
		}
		return false;
	}

      private:
	T values[d]; // NOLINT
	static constexpr T default_value = 0;
};
} // namespace quokka

// array + array
template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> sum;
	for (size_t i = 0; i < a.size(); ++i) {
		sum[i] = a[i] + b[i];
	}
	return sum;
}

// array + scalar
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> scalarsum;
	for (size_t i = 0; i < v.size(); ++i) {
		scalarsum[i] = v[i] + scalar;
	}
	return scalarsum;
}

// scalar + array
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator+(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> scalarsum;
	for (size_t i = 0; i < v.size(); ++i) {
		scalarsum[i] = scalar + v[i];
	}
	return scalarsum;
}

// array - array
template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator-(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> diff;
	for (size_t i = 0; i < a.size(); ++i) {
		diff[i] = a[i] - b[i];
	}
	return diff;
}

// array * array
template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> prod;
	for (size_t i = 0; i < a.size(); ++i) {
		prod[i] = a[i] * b[i];
	}
	return prod;
}

// array / array
template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> div;
	for (size_t i = 0; i < a.size(); ++i) {
		div[i] = a[i] / b[i];
	}
	return div;
}

// scalar * array
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> scalarprod;
	for (size_t i = 0; i < v.size(); ++i) {
		scalarprod[i] = scalar * v[i];
	}
	return scalarprod;
}

// array * scalar
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator*(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> scalarprod;
	for (size_t i = 0; i < v.size(); ++i) {
		scalarprod[i] = scalar * v[i];
	}
	return scalarprod;
}

// array *= scalar
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator*=(quokka::valarray<T, d> &v, T const &scalar)
{
	for (size_t i = 0; i < v.size(); ++i) {
		v[i] *= scalar;
	}
}

// array += array
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator+=(quokka::valarray<T, d> &a, quokka::valarray<T, d> const &b)
{
	for (size_t i = 0; i < a.size(); ++i) {
		a[i] += b[i];
	}
}

// array / scalar
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(quokka::valarray<T, d> const &v, T const &scalar) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> scalardiv;
	for (size_t i = 0; i < v.size(); ++i) {
		scalardiv[i] = v[i] / scalar;
	}
	return scalardiv;
}

// scalar / array
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator/(T const &scalar, quokka::valarray<T, d> const &v) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> scalardiv;
	for (size_t i = 0; i < v.size(); ++i) {
		scalardiv[i] = scalar / v[i];
	}
	return scalardiv;
}

// array /= scalar
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator/=(quokka::valarray<T, d> &v, T const &scalar)
{
	for (size_t i = 0; i < v.size(); ++i) {
		v[i] /= scalar;
	}
}

// abs(array)
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto abs(quokka::valarray<T, d> const &v) -> quokka::valarray<T, d>
{
	quokka::valarray<T, d> abs_v;
	for (size_t i = 0; i < v.size(); ++i) {
		abs_v[i] = std::abs(v[i]);
	}
	return abs_v;
}

// min(array)
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto min(quokka::valarray<T, d> const &v) -> T
{
	static_assert(d >= 1);
	T min_val = v[0]; // v must have at least 1 element

	for (size_t i = 0; i < v.size(); ++i) {
		min_val = std::min(min_val, v[i]);
	}
	return min_val;
}

// max(array)
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto max(quokka::valarray<T, d> const &v) -> T
{
	static_assert(d >= 1);
	T max_val = v[0]; // v must have at least 1 element

	for (size_t i = 0; i < v.size(); ++i) {
		max_val = std::max(max_val, v[i]);
	}
	return max_val;
}

// sum(array)
template <typename T, int d> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sum(quokka::valarray<T, d> const &v) -> T
{
	T sum_val = 0;
	for (size_t i = 0; i < v.size(); ++i) {
		sum_val += v[i];
	}
	return sum_val;
}

// array > array
template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator>(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<bool, d>
{
	quokka::valarray<bool, d> comp;
	for (size_t i = 0; i < a.size(); ++i) {
		comp[i] = a[i] > b[i];
	}
	return comp;
}

// array > scalar
template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator>(quokka::valarray<T, d> const &a, T const &scalar) -> quokka::valarray<bool, d>
{
	quokka::valarray<bool, d> comp;
	for (size_t i = 0; i < a.size(); ++i) {
		comp[i] = a[i] > scalar;
	}
	return comp;
}

// array < array
template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator<(quokka::valarray<T, d> const &a, quokka::valarray<T, d> const &b) -> quokka::valarray<bool, d>
{
	quokka::valarray<bool, d> comp;
	for (size_t i = 0; i < a.size(); ++i) {
		comp[i] = a[i] < b[i];
	}
	return comp;
}

// array < scalar
template <typename T, int d>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto operator<(quokka::valarray<T, d> const &a, T const &scalar) -> quokka::valarray<bool, d>
{
	quokka::valarray<bool, d> comp;
	for (size_t i = 0; i < a.size(); ++i) {
		comp[i] = a[i] < scalar;
	}
	return comp;
}

#endif // VALARRAY_HPP_
