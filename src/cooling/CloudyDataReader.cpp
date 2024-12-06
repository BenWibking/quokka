//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file CloudyDataReader.cpp
/// \brief Implements methods for reading the cooling rate tables produced by
/// Cloudy Cooling Tools.

/***********************************************************************
/ Initialize Cloudy cooling data
/ Copyright (c) 2013, Enzo/Grackle Development Team.
/
/ Distributed under the terms of the Enzo Public Licence.
/ The full license is in the file ENZO_LICENSE, distributed with this
/ software.
************************************************************************/

#include <vector>

#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_Print.H"
#include "AMReX_TableData.H"

#include "CloudyDataReader.hpp"
#include "math/FastMath.hpp"

namespace quokka::TabulatedCooling
{

void initialize_cloudy_data(cloudy_cooling_tools_data &my_cloudy, std::string const &grackle_data_file, code_units const &my_units)
{
	// Initialize vectors
	my_cloudy.grid_parameters.resize(CLOUDY_MAX_DIMENSION);
	my_cloudy.grid_parametersVec.resize(CLOUDY_MAX_DIMENSION);
	my_cloudy.grid_dimension.resize(CLOUDY_MAX_DIMENSION);
	for (int64_t q = 0; q < CLOUDY_MAX_DIMENSION; q++) {
		my_cloudy.grid_dimension[q] = 0;
	}

	// Get unit conversion factors
	const double co_length_units = my_units.length_units;
	const double co_density_units = my_units.density_units;
	const double tbase1 = my_units.time_units;
	const double xbase1 = co_length_units;
	const double dbase1 = co_density_units;
	const double mh = 1.67e-24;
	const double CoolUnit = (xbase1 * xbase1 * mh * mh) / (tbase1 * tbase1 * tbase1 * dbase1);
	const double small_fastlog_value = FastMath::log10(1.0e-99 / CoolUnit);

	amrex::Print() << "Initializing Cloudy cooling.\n";
	amrex::Print() << std::format("cloudy_table_file: {}.\n", grackle_data_file);

	// Read cooling data from hdf5 file
	herr_t const h5_error = -1;
	hid_t const file_id = H5Fopen(grackle_data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id != h5_error, "Failed to open Grackle data file!");

	// Open cooling dataset and get grid dimensions
	std::string parameter_name = "/Cooling";
	hid_t dset_id = H5Dopen2(file_id, parameter_name.c_str(), H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dset_id != h5_error, "Can't open Cooling table!");

	// Grid rank
	{
		hid_t const attr_id = H5Aopen_name(dset_id, "Rank");
		int64_t temp_int = 0;
		H5Aread(attr_id, HDF5_I8, &temp_int);
		my_cloudy.grid_rank = temp_int;
		H5Aclose(attr_id);
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(my_cloudy.grid_rank <= CLOUDY_MAX_DIMENSION,
						 "Error: rank of Cloudy cooling data must be less than or equal to "
						 "CLOUDY_MAX_DIMENSION");
	}

	// Grid dimension
	{
		std::vector<int64_t> temp_int_arr(my_cloudy.grid_rank);
		hid_t const attr_id = H5Aopen_name(dset_id, "Dimension");
		H5Aread(attr_id, HDF5_I8, temp_int_arr.data());

		for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {
			my_cloudy.grid_dimension[q] = temp_int_arr[q];
		}
		H5Aclose(attr_id);
	}

	// Read grid parameters
	for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {
		if (q < my_cloudy.grid_rank - 1) {
			parameter_name = std::format("/Parameter{}", q + 1);
		} else {
			parameter_name = "/Temperature";
		}

		my_cloudy.grid_parametersVec[q] = amrex::Gpu::PinnedVector<double>(my_cloudy.grid_dimension[q]);
		dset_id = H5Dopen2(file_id, parameter_name.c_str(), H5P_DEFAULT);
		H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, my_cloudy.grid_parametersVec[q].dataPtr());
		my_cloudy.grid_parameters[q] =
		    amrex::Table1D<double>(my_cloudy.grid_parametersVec[q].dataPtr(), 0, static_cast<int>(my_cloudy.grid_dimension[q]));

		for (int w = 0; w < my_cloudy.grid_dimension[q]; w++) {
			if (q == my_cloudy.grid_rank - 1) {
				double const T = my_cloudy.grid_parameters[q](w);
				// convert temperature to log
				my_cloudy.grid_parameters[q](w) = log10(T);
				// compute min/max
				my_cloudy.T_min = std::min(T, my_cloudy.T_min);
				my_cloudy.T_max = std::max(T, my_cloudy.T_max);
			}
		}
		H5Dclose(dset_id);

		amrex::Print() << std::format("\t{}: {} to {} ({} steps).\n", parameter_name, my_cloudy.grid_parameters[q](0),
					      my_cloudy.grid_parameters[q](static_cast<int>(my_cloudy.grid_dimension[q]) - 1), my_cloudy.grid_dimension[q]);
	}

	// compute data table size
	my_cloudy.data_size = 1;
	for (int64_t q = 0; q < my_cloudy.grid_rank; q++) {
		my_cloudy.data_size *= my_cloudy.grid_dimension[q];
	}

	// Read cooling data
	{
		my_cloudy.cooling_dataVec = amrex::Gpu::PinnedVector<double>(my_cloudy.data_size);

		parameter_name = "/Cooling";
		dset_id = H5Dopen2(file_id, parameter_name.c_str(),
				   H5P_DEFAULT); // new API in HDF5 1.8.0+
		const hid_t status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, my_cloudy.cooling_dataVec.dataPtr());
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read Cooling dataset!");

		// N.B.: Table2D uses column-major (Fortran-order) indexing, but HDF5 tables use row-major (C-order) indexing!
		amrex::GpuArray<int, 2> const lo{0, 0};
		amrex::GpuArray<int, 2> const hi{static_cast<int>(my_cloudy.grid_dimension[1]), static_cast<int>(my_cloudy.grid_dimension[0])};
		my_cloudy.cooling_data = amrex::Table2D<double>(my_cloudy.cooling_dataVec.dataPtr(), lo, hi);

		for (int64_t q = 0; q < my_cloudy.data_size; q++) {
			// Convert to code units
			double const value = my_cloudy.cooling_dataVec[q] / CoolUnit;
			// Convert to not-quite-log10 (using FastMath)
			my_cloudy.cooling_dataVec[q] = value > 0 ? FastMath::log10(value) : small_fastlog_value;
		}
		H5Dclose(dset_id);
	}

	// Read heating data
	{
		my_cloudy.heating_dataVec = amrex::Gpu::PinnedVector<double>(my_cloudy.data_size);

		parameter_name = "/Heating";
		dset_id = H5Dopen2(file_id, parameter_name.c_str(),
				   H5P_DEFAULT); // new API in HDF5 1.8.0+
		const hid_t status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, my_cloudy.heating_dataVec.dataPtr());
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read Heating dataset!");

		// N.B.: Table2D uses column-major (Fortran-order) indexing, but HDF5 tables use row-major (C-order) indexing!
		amrex::GpuArray<int, 2> const lo{0, 0};
		amrex::GpuArray<int, 2> const hi{static_cast<int>(my_cloudy.grid_dimension[1]), static_cast<int>(my_cloudy.grid_dimension[0])};
		my_cloudy.heating_data = amrex::Table2D<double>(my_cloudy.heating_dataVec.dataPtr(), lo, hi);

		for (int64_t q = 0; q < my_cloudy.data_size; q++) {
			// Convert to code units
			double const value = my_cloudy.heating_dataVec[q] / CoolUnit;
			// Convert to not-quite-log10 (using FastMath)
			my_cloudy.heating_dataVec[q] = value > 0 ? FastMath::log10(value) : small_fastlog_value;
		}
		H5Dclose(dset_id);
	}

	// Read mean molecular weight table
	{
		my_cloudy.mmw_dataVec = amrex::Gpu::PinnedVector<double>(my_cloudy.data_size);

		// N.B.: Table2D uses column-major (Fortran-order) indexing, but HDF5 tables use row-major (C-order) indexing!
		amrex::GpuArray<int, 2> const lo{0, 0};
		amrex::GpuArray<int, 2> const hi{static_cast<int>(my_cloudy.grid_dimension[1]), static_cast<int>(my_cloudy.grid_dimension[0])};
		my_cloudy.mmw_data = amrex::Table2D<double>(my_cloudy.mmw_dataVec.dataPtr(), lo, hi);

		parameter_name = "/MMW";
		dset_id = H5Dopen2(file_id, parameter_name.c_str(),
				   H5P_DEFAULT); // new API in HDF5 1.8.0+
		const hid_t status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, my_cloudy.mmw_dataVec.dataPtr());
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != h5_error, "Failed to read MMW dataset!");
		H5Dclose(dset_id);

		// compute min/max
		for (double const &mmw : my_cloudy.mmw_dataVec) {
			my_cloudy.mmw_min = std::min(mmw, my_cloudy.mmw_min);
			my_cloudy.mmw_max = std::max(mmw, my_cloudy.mmw_max);
		}
	}

	// close HDF5 file
	H5Fclose(file_id);
}

auto extract_2d_table(amrex::Table2D<double> const &table2D) -> amrex::TableData<double, 2>
{
	// Table2D dimensions (F-ordering) are: temperature, redshift, density
	// (but the Table2D data is stored with C-ordering)
	auto lo = table2D.begin;
	auto hi = table2D.end;

	// N.B.: Table2D uses column-major (Fortran-order) indexing, but HDF5 tables use row-major (C-order) indexing,
	// so we reverse the indices here
	amrex::Array<int, 2> newlo{lo[1], lo[0]};
	amrex::Array<int, 2> newhi{hi[1] - 1, hi[0] - 1};
	amrex::TableData<double, 2> tableData(newlo, newhi, amrex::The_Pinned_Arena());
	auto table = tableData.table();

	for (int i = newlo[0]; i <= newhi[0]; ++i) {
		for (int j = newlo[1]; j <= newhi[1]; ++j) {
			// swap index ordering so we can use Table2D's F-ordering accessor ()
			table(i, j) = table2D(j, i);
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

	amrex::TableData<double, 1> tableData(newlo, newhi, amrex::The_Pinned_Arena());
	auto table = tableData.table();

	for (int i = newlo[0]; i <= newhi[0]; ++i) {
		table(i) = table1D(i);
	}
	return tableData;
}

} // namespace quokka::TabulatedCooling