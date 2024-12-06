//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file GrackleDataReader.cpp
/// \brief Implements methods for reading the cooling rate tables used by
/// Grackle. Significantly modified from the original version in Grackle to use
/// modern C++ constructs.
///

/***********************************************************************
/ Initialize Cloudy cooling data
/ Copyright (c) 2013, Enzo/Grackle Development Team.
/
/ Distributed under the terms of the Enzo Public Licence.
/ The full license is in the file ENZO_LICENSE, distributed with this
/ software.
************************************************************************/

#include "GrackleDataReader.hpp"
#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_Print.H"
#include "AMReX_TableData.H"
#include "fmt/core.h"
#include "math/FastMath.hpp"

namespace quokka::GrackleLikeCooling
{

static const bool grackle_verbose = true;

void initialize_cloudy_data(grackle_data &my_cloudy, char const *group_name, std::string &grackle_data_file, code_units &my_units)
{
	// Initialize vectors
	my_cloudy.grid_parameters.resize(CLOUDY_MAX_DIMENSION);
	my_cloudy.grid_dimension.resize(CLOUDY_MAX_DIMENSION);
	for (int64_t q = 0; q < CLOUDY_MAX_DIMENSION; q++) {
		my_cloudy.grid_dimension[q] = 0;
	}

	if (grackle_verbose) {
		amrex::Print() << std::format("Initializing Grackle-like cooling: {}.\n", group_name);
		amrex::Print() << std::format("grackle_data_file: {}.\n", grackle_data_file);
	}

	// Get unit conversion factors (assuming z=0)
	double co_length_units = NAN;
	double co_density_units = NAN;
	co_length_units = my_units.length_units;
	co_density_units = my_units.density_units;
	double tbase1 = my_units.time_units;
	double xbase1 = co_length_units;
	double dbase1 = co_density_units;
	double mh = 1.67e-24;
	double CoolUnit = (xbase1 * xbase1 * mh * mh) / (tbase1 * tbase1 * tbase1 * dbase1);

	const double small_fastlog_value = FastMath::log10(1.0e-99 / CoolUnit);

	// Read cooling data from hdf5 file
	hid_t file_id = 0;
	hid_t dset_id = 0;
	hid_t attr_id = 0;
	herr_t status = 0;
	herr_t h5_error = -1;

	file_id = H5Fopen(grackle_data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id != h5_error, "Failed to open Grackle data file!");

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(H5Aexists(file_id, "old_style") == 0, "Old-style Grackle data tables are not supported!");

	// Open cooling dataset and get grid dimensions
	std::string parameter_name;
	parameter_name = std::format("/CoolingRates/{}/Cooling", group_name);

	dset_id = H5Dopen2(file_id, parameter_name.c_str(),
			   H5P_DEFAULT); // new API in HDF5 1.8.0+

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dset_id != h5_error, "Can't open Cooling table!");

	{
		// Grid rank
		attr_id = H5Aopen_name(dset_id, "Rank");

		int64_t temp_int = 0;
		status = H5Aread(attr_id, HDF5_I8, &temp_int);
		my_cloudy.grid_rank = temp_int;

		status = H5Aclose(attr_id);
	}

	{
		// Grid dimension
		std::vector<int64_t> temp_int_arr(my_cloudy.grid_rank);
		attr_id = H5Aopen_name(dset_id, "Dimension");

		status = H5Aread(attr_id, HDF5_I8, temp_int_arr.data());

		for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {
			my_cloudy.grid_dimension[q] = temp_int_arr[q];
		}

		status = H5Aclose(attr_id);
	}

	// Grid parameters
	for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {

		if (q < my_cloudy.grid_rank - 1) {
			parameter_name = std::format("Parameter{}", q + 1);
		} else {
			parameter_name = "Temperature";
		}

		auto *temp_data = new double[my_cloudy.grid_dimension[q]]; // NOLINT(cppcoreguidelines-owning-memory)

		attr_id = H5Aopen_name(dset_id, parameter_name.c_str());

		status = H5Aread(attr_id, HDF5_R8, temp_data);

		my_cloudy.grid_parameters[q] = amrex::Table1D<double>(temp_data, 0, static_cast<int>(my_cloudy.grid_dimension[q]));

		for (int w = 0; w < my_cloudy.grid_dimension[q]; w++) {
			if (q < my_cloudy.grid_rank - 1) {
				my_cloudy.grid_parameters[q](w) = temp_data[w];
			} else {
				double const T = temp_data[w]; // NOLINT
				// convert temperature to log
				my_cloudy.grid_parameters[q](w) = log10(T);
				// compute min/max
				my_cloudy.T_min = std::min(T, my_cloudy.T_min);
				my_cloudy.T_max = std::max(T, my_cloudy.T_max);
			}
		}

		if (grackle_verbose) {
			amrex::Print() << std::format("\t{}: {} to {} ({} steps).\n", parameter_name, my_cloudy.grid_parameters[q](0),
						      my_cloudy.grid_parameters[q](static_cast<int>(my_cloudy.grid_dimension[q]) - 1),
						      my_cloudy.grid_dimension[q]);
		}

		status = H5Aclose(attr_id);
	}

	my_cloudy.data_size = 1;

	for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {
		my_cloudy.data_size *= my_cloudy.grid_dimension[q];
	}

	{
		// Read Cooling data
		auto *temp_data = new double[my_cloudy.data_size]; // NOLINT(cppcoreguidelines-owning-memory)

		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read Cooling dataset!");

		amrex::GpuArray<int, 3> lo{0, 0, 0};
		amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[2]), static_cast<int>(my_cloudy.grid_dimension[1]),
					   static_cast<int>(my_cloudy.grid_dimension[0])};

		// N.B.: Table3D uses column-major (Fortran-order) indexing, but Grackle
		// tables use row-major (C-order) indexing!
		my_cloudy.cooling_data = amrex::Table3D<double>(temp_data, lo, hi);

		for (int64_t q = 0; q < my_cloudy.data_size; q++) {
			// Convert to code units
			double value = temp_data[q] / CoolUnit;
			// Convert to not-quite-log10 (using FastMath)
			temp_data[q] = value > 0 ? FastMath::log10(value) : small_fastlog_value;
		}

		status = H5Dclose(dset_id);
	}

	{
		// Read Heating data
		auto *temp_data = new double[my_cloudy.data_size]; // NOLINT(cppcoreguidelines-owning-memory)

		parameter_name = std::format("/CoolingRates/{}/Heating", group_name);

		dset_id = H5Dopen2(file_id, parameter_name.c_str(),
				   H5P_DEFAULT); // new API in HDF5 1.8.0+

		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read Heating dataset!");

		amrex::GpuArray<int, 3> lo{0, 0, 0};
		amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[2]), static_cast<int>(my_cloudy.grid_dimension[1]),
					   static_cast<int>(my_cloudy.grid_dimension[0])};

		// N.B.: Table3D uses column-major (Fortran-order) indexing, but Grackle
		// tables use row-major (C-order) indexing!
		my_cloudy.heating_data = amrex::Table3D<double>(temp_data, lo, hi);

		for (int64_t q = 0; q < my_cloudy.data_size; q++) {
			// Convert to code units
			double value = temp_data[q] / CoolUnit;
			// Convert to not-quite-log10 (using FastMath)
			temp_data[q] = value > 0 ? FastMath::log10(value) : small_fastlog_value;
		}

		status = H5Dclose(dset_id);
	}

	if (std::string(group_name) == "Primordial") {
		// Read mean molecular weight table
		auto *temp_data = new double[my_cloudy.data_size]; // NOLINT(cppcoreguidelines-owning-memory)

		amrex::GpuArray<int, 3> lo{0, 0, 0};
		amrex::GpuArray<int, 3> hi{static_cast<int>(my_cloudy.grid_dimension[2]), static_cast<int>(my_cloudy.grid_dimension[1]),
					   static_cast<int>(my_cloudy.grid_dimension[0])};

		// N.B.: Table3D uses column-major (Fortran-order) indexing, but Grackle
		// tables use row-major (C-order) indexing!
		my_cloudy.mmw_data = amrex::Table3D<double>(temp_data, lo, hi);

		parameter_name = std::format("/CoolingRates/{}/MMW", group_name);

		dset_id = H5Dopen2(file_id, parameter_name.c_str(),
				   H5P_DEFAULT); // new API in HDF5 1.8.0+

		status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);

		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read MMW dataset!");

		status = H5Dclose(dset_id);

		// compute min/max
		for (int64_t i = 0; i < my_cloudy.data_size; ++i) {
			amrex::Real const mmw = temp_data[i]; // NOLINT
			my_cloudy.mmw_min = std::min(mmw, my_cloudy.mmw_min);
			my_cloudy.mmw_max = std::max(mmw, my_cloudy.mmw_max);
		}
	}

	status = H5Fclose(file_id);

	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(my_cloudy.grid_rank <= CLOUDY_MAX_DIMENSION, "Error: rank of Cloudy cooling data must be less than or equal to "
										      "CLOUDY_MAX_DIMENSION");
}

auto extract_2d_table(amrex::Table3D<double> const &table3D, int redshift_index) -> amrex::TableData<double, 2>
{
	// table3d dimensions (F-ordering) are: temperature, redshift, density
	// (but the table3d data is stored with C-ordering)
	auto lo = table3D.begin;
	auto hi = table3D.end;

	// N.B.: Table3D uses column-major (Fortran-order) indexing, but
	// Grackle tables use row-major (C-order) indexing, so we reverse the indices
	// here
	amrex::Array<int, 2> newlo{lo[2], lo[0]};
	amrex::Array<int, 2> newhi{hi[2] - 1, hi[0] - 1};
	amrex::TableData<double, 2> tableData(newlo, newhi, amrex::The_Managed_Arena());
	auto table = tableData.table();

	for (int i = newlo[0]; i <= newhi[0]; ++i) {
		for (int j = newlo[1]; j <= newhi[1]; ++j) {
			// swap index ordering so we can use Table2D's F-ordering accessor ()
			table(i, j) = table3D(j, redshift_index, i);
		}
	}
	// N.B.: table should now be F-ordered: density, temperature
	//  and the Table2D accessor function (which is F-ordered) can be used.
	return tableData;
}

auto copy_1d_table(amrex::Table1D<double> const &table1D) -> amrex::TableData<double, 1>
{
	auto lo = table1D.begin;
	auto hi = table1D.end;
	amrex::Array<int, 1> newlo{lo};
	amrex::Array<int, 1> newhi{hi - 1};

	amrex::TableData<double, 1> tableData(newlo, newhi, amrex::The_Managed_Arena());
	auto table = tableData.table();

	for (int i = newlo[0]; i <= newhi[0]; ++i) {
		table(i) = table1D(i);
	}
	return tableData;
}

} // namespace quokka::GrackleLikeCooling