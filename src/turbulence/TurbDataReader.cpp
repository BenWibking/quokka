//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file TurbDataReader.cpp
/// \brief Reads turbulent driving fields generated as cubic HDF5 arrays.
///

#include "TurbDataReader.hpp"
#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_Print.H"
#include "AMReX_TableData.H"
#include <string>

auto read_dataset(hid_t &file_id, char const *dataset_name) -> amrex::Table3D<double>
{
	// open dataset
	hid_t dset_id = 0;
	dset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dset_id != -1, "Can't open table!");

	// get dimensions
	hid_t const dspace = H5Dget_space(dset_id);
	const int ndims = H5Sget_simple_extent_ndims(dspace);
	std::vector<hsize_t> dims(ndims);
	H5Sget_simple_extent_dims(dspace, dims.data(), nullptr);

	uint64_t data_size = 1;
	for (int idim = 0; idim < ndims; ++idim) {
		data_size *= dims[idim];
	}

	// allocate array for dataset storage
	auto *temp_data = new double[data_size];

	// read dataset
	herr_t status = H5Dread(dset_id, HDF5_R8, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(status != -1, "Failed to read dataset!");

	// close dataset
	H5Dclose(dset_id);

	// WARNING: Table3D uses column-major (Fortran-order) indexing, but HDF5
	// tables use row-major (C-order) indexing!
	amrex::GpuArray<int, 3> const lo{0, 0, 0};
	amrex::GpuArray<int, 3> const hi{static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2])};
	auto table = amrex::Table3D<double>(temp_data, lo, hi);
	return table;
}

void initialize_turbdata(turb_data &data, std::string &data_file)
{
	amrex::Print() << "Initializing turbulence data...\n";
	amrex::Print() << std::format("data_file: {}.\n", data_file);

	herr_t const h5_error = -1;

	// open file
	hid_t file_id = 0;
	file_id = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(file_id != h5_error, "Failed to open data file!");

	data.dvx = read_dataset(file_id, "pertx");
	data.dvy = read_dataset(file_id, "perty");
	data.dvz = read_dataset(file_id, "pertz");

	// close file
	H5Fclose(file_id);
}

auto get_tabledata(amrex::Table3D<double> &in_t) -> amrex::TableData<double, 3>
{
	amrex::Array<int, 3> tlo{in_t.begin[0], in_t.begin[1], in_t.begin[2]};
	amrex::Array<int, 3> thi{in_t.end[0] - 1, in_t.end[1] - 1, in_t.end[2] - 1};
	amrex::TableData<double, 3> tableData(tlo, thi, amrex::The_Pinned_Arena());
	auto h_table = tableData.table();

	// amrex::Print() << "Copying tableData on indices " << tlo << " to " << thi << ".\n";

	// fill tableData
	for (int i = tlo[0]; i <= thi[0]; ++i) {
		for (int j = tlo[1]; j <= thi[1]; ++j) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				h_table(i, j, k) = in_t(i, j, k);
			}
		}
	}

	return tableData;
}

auto computeRms(amrex::TableData<amrex::Real, 3> &dvx, amrex::TableData<amrex::Real, 3> &dvy, amrex::TableData<amrex::Real, 3> &dvz) -> amrex::Real
{
	amrex::Array<int, 3> tlo = dvx.lo();
	amrex::Array<int, 3> thi = dvx.hi();
	auto const &dvx_table = dvx.const_table();
	auto const &dvy_table = dvy.const_table();
	auto const &dvz_table = dvz.const_table();

	// compute rms power
	amrex::Real rms_sq = 0;
	amrex::Long N = 0;
	for (int i = tlo[0]; i <= thi[0]; ++i) {
		for (int j = tlo[1]; j <= thi[1]; ++j) {
			for (int k = tlo[2]; k <= thi[2]; ++k) {
				amrex::Real const vx = dvx_table(i, j, k);
				amrex::Real const vy = dvy_table(i, j, k);
				amrex::Real const vz = dvz_table(i, j, k);
				rms_sq += vx * vx + vy * vy + vz * vz;
				++N;
			}
		}
	}
	rms_sq /= static_cast<amrex::Real>(N);
	return std::sqrt(rms_sq);
}
