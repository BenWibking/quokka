#include <cassert>
#include <cmath>

#include "AMReX_BLassert.H"
#include "math/interpolate.hpp"

#define LIKELY_IN_CACHE_SIZE 8

/** @brief find index of a sorted array such that arr[i] <= key < arr[i + 1].
 *
 * If an starting index guess is in-range, the array values around this
 * index are first checked.  This allows for repeated calls for well-ordered
 * keys (a very common case) to use the previous index as a very good guess.
 *
 * If the guess value is not useful, bisection of the array is used to
 * find the index.  If there is no such index, the return values are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @param guess initial guess of index
 * @return index
 */
AMREX_GPU_HOST_DEVICE
int64_t binary_search_with_guess(const double key, const double *arr, int64_t len, int64_t guess)
{
	int64_t imin = 0;
	int64_t imax = len;

	/* Handle keys outside of the arr range first */
	if (key > arr[len - 1]) {
		return len;
	} else if (key < arr[0]) {
		return -1;
	}

	/*
	 * If len <= 4 use linear search.
	 * From above we know key >= arr[0] when we start.
	 */
	if (len <= 4) {
		int64_t i;

		for (i = 1; i < len && key >= arr[i]; ++i)
			;
		return i - 1;
	}

	if (guess > len - 3) {
		guess = len - 3;
	}
	if (guess < 1) {
		guess = 1;
	}

	/* check most likely values: guess - 1, guess, guess + 1 */
	if (key < arr[guess]) {
		if (key < arr[guess - 1]) {
			imax = guess - 1;
			/* last attempt to restrict search to items in cache */
			if (guess > LIKELY_IN_CACHE_SIZE && key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
				imin = guess - LIKELY_IN_CACHE_SIZE;
			}
		} else {
			/* key >= arr[guess - 1] */
			return guess - 1;
		}
	} else {
		/* key >= arr[guess] */
		if (key < arr[guess + 1]) {
			return guess;
		} else {
			/* key >= arr[guess + 1] */
			if (key < arr[guess + 2]) {
				return guess + 1;
			} else {
				/* key >= arr[guess + 2] */
				imin = guess + 2;
				/* last attempt to restrict search to items in
				 * cache */
				if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
					imax = guess + LIKELY_IN_CACHE_SIZE;
				}
			}
		}
	}

	/* finally, find index by bisection */
	while (imin < imax) {
		const int64_t imid = imin + ((imax - imin) >> 1);
		if (key >= arr[imid]) {
			imin = imid + 1;
		} else {
			imax = imid;
		}
	}
	return imin - 1;
}

#undef LIKELY_IN_CACHE_SIZE

AMREX_GPU_HOST_DEVICE
void interpolate_arrays(double *x, double *y, int len, double *arr_x, double *arr_y, int arr_len)
{
	/* Note: arr_x must be sorted in ascending order,
		and arr_len must be >= 3. */

	// check if x is within the range of arr_x
	AMREX_ASSERT(x[0] >= arr_x[0] && x[len - 1] <= arr_x[arr_len - 1]);

	int64_t j = 0;
	for (int i = 0; i < len; i++) {
		j = binary_search_with_guess(x[i], arr_x, arr_len, j);

		if (j == -1) {
			y[i] = NAN;
		} else if (j == arr_len) {
			y[i] = NAN;
		} else if (j == arr_len - 1) {
			y[i] = arr_y[j];
		} else if (x[i] == arr_x[j]) { // avoid roundoff error
			y[i] = arr_y[j];
		} else {
			const double slope = (arr_y[j + 1] - arr_y[j]) / (arr_x[j + 1] - arr_x[j]);
			y[i] = slope * (x[i] - arr_x[j]) + arr_y[j];
		}
		AMREX_ASSERT(!std::isnan(y[i]));
	}
}

AMREX_GPU_HOST_DEVICE
double interpolate_value(double x, double const *arr_x, double const *arr_y, int arr_len)
{
	/* Note: arr_x must be sorted in ascending order,
		and arr_len must be >= 3. */

	int64_t j = 0;
	j = binary_search_with_guess(x, arr_x, arr_len, j);

	double y = NAN;
	if (j == -1) {
		y = NAN;
	} else if (j == arr_len) {
		y = NAN;
	} else if (j == arr_len - 1) {
		y = arr_y[j];
	} else if (x == arr_x[j]) { // avoid roundoff error
		y = arr_y[j];
	} else {
		const double slope = (arr_y[j + 1] - arr_y[j]) / (arr_x[j + 1] - arr_x[j]);
		y = slope * (x - arr_x[j]) + arr_y[j];
	}
	AMREX_ASSERT(!std::isnan(y));
	return y;
}
