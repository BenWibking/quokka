//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
//! \file projection.cpp
///  \brief AMReX I/O for 2D projections

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FPC.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_IntVect.H"
#include "AMReX_Orientation.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_REAL.H"

#include "projection.hpp"

namespace quokka::diagnostics
{

namespace detail
{
auto direction_to_string(const amrex::Direction dir) -> std::string
{
	if (dir == amrex::Direction::x) {
		return std::string("x");
	}
#if AMREX_SPACEDIM >= 2
	if (dir == amrex::Direction::y) {
		return std::string("y");
	}
#endif
#if AMREX_SPACEDIM == 3
	if (dir == amrex::Direction::z) {
		return std::string("z");
	}
#endif

	amrex::Error("invalid direction in quokka::diagnostics::direction_to_string!");
	return std::string("");
}

void printLowerDimIntVect(std::ostream &a_File, const amrex::IntVect &a_IntVect, int skipDim)
{
	int doneDim = 0;
	a_File << '(';
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		if (idim != skipDim) {
			a_File << a_IntVect[idim];
			doneDim++;
			if (doneDim < AMREX_SPACEDIM - 1) {
				a_File << ",";
			}
		}
	}
	a_File << ')';
}

void printLowerDimBox(std::ostream &a_File, const amrex::Box &a_box, int skipDim)
{
	a_File << '(';
	printLowerDimIntVect(a_File, a_box.smallEnd(), skipDim);
	a_File << ' ';
	printLowerDimIntVect(a_File, a_box.bigEnd(), skipDim);
	a_File << ' ';
	printLowerDimIntVect(a_File, a_box.type(), skipDim);
	a_File << ')';
}

void Write2DMultiLevelPlotfile(const std::string &a_pltfile, int a_nlevels, const amrex::Vector<const amrex::MultiFab *> &a_slice,
			       const amrex::Vector<std::string> &a_varnames, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Real &a_time,
			       const amrex::Vector<int> &a_steps, const amrex::Vector<amrex::IntVect> &a_rref)
{
	// Write a 2D AMReX Plotfile with the contents of the vector of MultiFabs `a_slice'.

	const std::string levelPrefix = "Level_";
	const std::string mfPrefix = "Cell";
	const std::string versionName = "HyperCLaw-V1.1";

	bool const callBarrier(false);
	amrex::PreBuildDirectorHierarchy(a_pltfile, levelPrefix, a_nlevels, callBarrier);
	amrex::ParallelDescriptor::Barrier();

	if (amrex::ParallelDescriptor::MyProc() == amrex::ParallelDescriptor::NProcs() - 1) {

		// Write 2D pltfile header
		amrex::Vector<amrex::BoxArray> boxArrays(a_nlevels);
		for (int level(0); level < boxArrays.size(); ++level) {
			boxArrays[level] = a_slice[level]->boxArray();
		}

		amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
		std::string const HeaderFileName(a_pltfile + "/Header");
		std::ofstream HeaderFile;
		HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
		if (!HeaderFile.good()) {
			amrex::FileOpenFailed(HeaderFileName);
		}
		Write2DPlotfileHeader(HeaderFile, a_nlevels, boxArrays, a_varnames, a_geoms, a_time, a_steps, a_rref, versionName, levelPrefix, mfPrefix);
		HeaderFile.flush();
		HeaderFile.close();
	}

	// Write a 2D version of the MF at each level
	for (int level = 0; level < a_nlevels; ++level) {
		VisMF2D(*a_slice[level], amrex::MultiFabFileFullPrefix(level, a_pltfile, levelPrefix, mfPrefix));
	}
}

void Write2DPlotfileHeader(std::ostream &HeaderFile, int nlevels, const amrex::Vector<amrex::BoxArray> &bArray, const amrex::Vector<std::string> &varnames,
			   const amrex::Vector<amrex::Geometry> &geom, const amrex::Real &time, const amrex::Vector<int> &level_steps,
			   const amrex::Vector<amrex::IntVect> &ref_ratio, const std::string &versionName, const std::string &levelPrefix,
			   const std::string &mfPrefix)
{
	int const finest_level(nlevels - 1);
	HeaderFile.precision(17);

	int const lowerSpaceDim = AMREX_SPACEDIM - 1;

	HeaderFile << versionName << '\n';
	HeaderFile << varnames.size() << '\n';
	for (const auto &varname : varnames) {
		HeaderFile << varname << "\n";
	}
	HeaderFile << lowerSpaceDim << '\n';
	HeaderFile << time << '\n';
	HeaderFile << finest_level << '\n';
	for (int idim = 0; idim < lowerSpaceDim; ++idim) {
		HeaderFile << geom[0].ProbLo(idim) << ' ';
	}
	HeaderFile << '\n';
	for (int idim = 0; idim < lowerSpaceDim; ++idim) {
		HeaderFile << geom[0].ProbHi(idim) << ' ';
	}
	HeaderFile << '\n';
	for (int i = 0; i < finest_level; ++i) {
		HeaderFile << ref_ratio[i][0] << ' ';
	}
	HeaderFile << '\n';
	for (int i = 0; i <= finest_level; ++i) {
		printLowerDimBox(HeaderFile, geom[i].Domain(), 2);
		HeaderFile << ' ';
	}
	HeaderFile << '\n';
	for (int i = 0; i <= finest_level; ++i) {
		HeaderFile << level_steps[i] << ' ';
	}
	HeaderFile << '\n';
	for (int i = 0; i <= finest_level; ++i) {
		for (int idim = 0; idim < lowerSpaceDim; ++idim) {
			HeaderFile << geom[i].CellSizeArray()[idim] << ' ';
		}
		HeaderFile << '\n';
	}
	HeaderFile << static_cast<int>(geom[0].Coord()) << '\n';
	HeaderFile << "0\n";

	for (int level = 0; level <= finest_level; ++level) {
		HeaderFile << level << ' ' << bArray[level].size() << ' ' << time << '\n';
		HeaderFile << level_steps[level] << '\n';

		const amrex::IntVect &domain_lo = geom[level].Domain().smallEnd();
		for (int i = 0; i < bArray[level].size(); ++i) {
			// Need to shift because the RealBox ctor we call takes the
			// physical location of index (0,0,0).  This does not affect
			// the usual cases where the domain index starts with 0.
			const amrex::Box &b = amrex::shift(bArray[level][i], -domain_lo);
			amrex::RealBox const loc = amrex::RealBox(b, geom[level].CellSize(), geom[level].ProbLo());
			for (int idim = 0; idim < lowerSpaceDim; ++idim) {
				HeaderFile << loc.lo(idim) << ' ' << loc.hi(idim) << '\n';
			}
		}
		HeaderFile << amrex::MultiFabHeaderPath(level, levelPrefix, mfPrefix) << '\n';
	}
}

void VisMF2D(const amrex::MultiFab &a_mf, const std::string &a_mf_name)
{
	auto whichRD = amrex::FArrayBox::getDataDescriptor();
	bool const doConvert(*whichRD != amrex::FPC::NativeRealDescriptor());

	amrex::Long bytesWritten(0);

	std::string const filePrefix(a_mf_name + "_D_");

	bool const calcMinMax = false;
	amrex::VisMF::Header::Version const currentVersion = amrex::VisMF::Header::Version_v1;
	amrex::VisMF::How const how = amrex::VisMF::How::NFiles;
	amrex::VisMF::Header hdr(a_mf, how, currentVersion, calcMinMax);

	int const nOutFiles = std::max(1, std::min(amrex::ParallelDescriptor::NProcs(), 256));
	bool const groupSets = false;
	bool const setBuf = true;

	amrex::NFilesIter nfi(nOutFiles, filePrefix, groupSets, setBuf);

	// Check if mf has sparse data
	bool useSparseFPP = false;
	const amrex::Vector<int> &pmap = a_mf.DistributionMap().ProcessorMap();
	std::set<int> procsWithData;
	amrex::Vector<int> procsWithDataVector;
	for (const int i : pmap) {
		procsWithData.insert(i);
	}
	if (static_cast<int>(procsWithData.size()) < nOutFiles) {
		useSparseFPP = true;
		for (const int it : procsWithData) {
			procsWithDataVector.push_back(it);
		}
	}

	if (useSparseFPP) {
		nfi.SetSparseFPP(procsWithDataVector);
	} else {
		nfi.SetDynamic();
	}
	for (; nfi.ReadyToWrite(); ++nfi) {
		int const whichRDBytes(whichRD->numBytes());
		int nFABs(0);
		amrex::Long writeDataItems(0);
		amrex::Long writeDataSize(0);
		for (amrex::MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
			const amrex::FArrayBox &fab = a_mf[mfi];
			std::stringstream hss;
			write_2D_header(hss, fab, fab.nComp());
			bytesWritten += static_cast<std::streamoff>(hss.tellp());
			bytesWritten += fab.box().numPts() * a_mf.nComp() * whichRDBytes;
			++nFABs;
		}
		std::unique_ptr<std::vector<char>> allFabData{};
		bool canCombineFABs = false;
		if ((nFABs > 1 || doConvert) && amrex::VisMF::GetUseSingleWrite()) {
			allFabData = std::make_unique<std::vector<char>>(bytesWritten);
		} // ---- else { no need to make a copy for one fab }
		canCombineFABs = allFabData != nullptr;

		if (canCombineFABs) {
			amrex::Long writePosition = 0;
			for (amrex::MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
				int hLength(0);
				const amrex::FArrayBox &fab = a_mf[mfi];
				writeDataItems = fab.box().numPts() * a_mf.nComp();
				writeDataSize = writeDataItems * whichRDBytes;
				char *afPtr = allFabData->data() + writePosition; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				std::stringstream hss;
				write_2D_header(hss, fab, fab.nComp());
				hLength = static_cast<int>(hss.tellp());
				auto tstr = hss.str();
				std::memcpy(afPtr, tstr.c_str(), hLength); // ---- the fab header
				amrex::Real const *fabdata = fab.dataPtr();
#ifdef AMREX_USE_GPU
				std::unique_ptr<amrex::FArrayBox> hostfab;
				if (fab.arena()->isManaged() || fab.arena()->isDevice()) {
					hostfab = std::make_unique<amrex::FArrayBox>(fab.box(), fab.nComp(), amrex::The_Pinned_Arena());
					amrex::Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(), fab.size() * sizeof(amrex::Real));
					amrex::Gpu::streamSynchronize();
					fabdata = hostfab->dataPtr();
				}
#endif
				memcpy(afPtr + hLength, fabdata, writeDataSize); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				writePosition += hLength + writeDataSize;
			}
			nfi.Stream().write(allFabData->data(), bytesWritten);
			nfi.Stream().flush();
			// delete[] allFabData;
		} else {
			for (amrex::MFIter mfi(a_mf); mfi.isValid(); ++mfi) {
				int hLength = 0;
				const amrex::FArrayBox &fab = a_mf[mfi];
				writeDataItems = fab.box().numPts() * a_mf.nComp();
				writeDataSize = writeDataItems * whichRDBytes;
				std::stringstream hss;
				write_2D_header(hss, fab, fab.nComp());
				hLength = static_cast<int>(hss.tellp());
				auto tstr = hss.str();
				nfi.Stream().write(tstr.c_str(), hLength); // ---- the fab header
				nfi.Stream().flush();
				amrex::Real const *fabdata = fab.dataPtr();
#ifdef AMREX_USE_GPU
				std::unique_ptr<amrex::FArrayBox> hostfab;
				if (fab.arena()->isManaged() || fab.arena()->isDevice()) {
					hostfab = std::make_unique<amrex::FArrayBox>(fab.box(), fab.nComp(), amrex::The_Pinned_Arena());
					amrex::Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(), fab.size() * sizeof(amrex::Real));
					amrex::Gpu::streamSynchronize();
					fabdata = hostfab->dataPtr();
				}
#endif
				nfi.Stream().write(reinterpret_cast<const char *>(fabdata), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
						   writeDataSize);
				nfi.Stream().flush();
			}
		}
	}

	int coordinatorProc(amrex::ParallelDescriptor::IOProcessorNumber());
	if (nfi.GetDynamic()) {
		coordinatorProc = nfi.CoordinatorProc();
	}
	hdr.CalculateMinMax(a_mf, coordinatorProc);

	Find2FOffsets(a_mf, filePrefix, hdr, currentVersion, nfi, nOutFiles, amrex::ParallelDescriptor::Communicator());

	Write2DMFHeader(a_mf_name, hdr, coordinatorProc, amrex::ParallelDescriptor::Communicator());
}

void Write2DMFHeader(const std::string &a_mf_name, amrex::VisMF::Header &hdr, int coordinatorProc, MPI_Comm comm)
{
	const int myProc(amrex::ParallelDescriptor::MyProc(comm));
	if (myProc == coordinatorProc) {

		std::string MFHdrFileName(a_mf_name);

		MFHdrFileName += "_H";

		amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);

		std::ofstream MFHdrFile;

		MFHdrFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
		MFHdrFile.open(MFHdrFileName.c_str(), std::ios::out | std::ios::trunc);

		if (!MFHdrFile.good()) {
			amrex::FileOpenFailed(MFHdrFileName);
		}

		MFHdrFile.setf(std::ios::floatfield, std::ios::scientific);
		MFHdrFile << hdr.m_vers << '\n';
		MFHdrFile << static_cast<int>(hdr.m_how) << '\n';
		MFHdrFile << hdr.m_ncomp << '\n';
		if (hdr.m_ngrow == hdr.m_ngrow[0]) {
			MFHdrFile << hdr.m_ngrow[0] << '\n';
		} else {
			MFHdrFile << hdr.m_ngrow << '\n';
		}

		MFHdrFile << "(" << hdr.m_ba.size() << " 0" << '\n';
		for (int i = 0; i < hdr.m_ba.size(); ++i) {
			printLowerDimBox(MFHdrFile, hdr.m_ba[i], 2);
			MFHdrFile << '\n';
		}
		MFHdrFile << ") \n";

		MFHdrFile << hdr.m_fod << '\n';

		MFHdrFile << hdr.m_min.size() << "," << hdr.m_min[0].size() << '\n';
		MFHdrFile.precision(16);
		for (auto &hdr_i : hdr.m_min) {
			for (const double j : hdr_i) {
				MFHdrFile << j << ",";
			}
			MFHdrFile << "\n";
		}

		MFHdrFile << "\n";

		MFHdrFile << hdr.m_max.size() << "," << hdr.m_max[0].size() << '\n';
		for (auto &hdr_i : hdr.m_max) {
			for (const double j : hdr_i) {
				MFHdrFile << j << ",";
			}
			MFHdrFile << "\n";
		}

		MFHdrFile.flush();
		MFHdrFile.close();
	}
}

void Find2FOffsets(const amrex::FabArray<amrex::FArrayBox> &mf, const std::string &filePrefix, amrex::VisMF::Header &hdr,
		   amrex::VisMF::Header::Version /*whichVersion*/, amrex::NFilesIter &nfi, int nOutFiles, MPI_Comm comm)
{
	bool const groupSets = false;

	const int myProc(amrex::ParallelDescriptor::MyProc(comm));
	const int nProcs(amrex::ParallelDescriptor::NProcs(comm));
	int coordinatorProc(amrex::ParallelDescriptor::IOProcessorNumber(comm));
	if (nfi.GetDynamic()) {
		coordinatorProc = nfi.CoordinatorProc();
	}

	auto whichRD = amrex::FArrayBox::getDataDescriptor();
	int const whichRDBytes(whichRD->numBytes());
	int const nComps(mf.nComp());

	if (myProc == coordinatorProc) { // ---- calculate offsets
		const amrex::BoxArray &mfBA = mf.boxArray();
		const amrex::DistributionMapping &mfDM = mf.DistributionMap();
		amrex::Vector<amrex::Long> fabHeaderBytes(mfBA.size(), 0);
		int const nFiles(amrex::NFilesIter::ActualNFiles(nOutFiles));
		int whichFileNumber(-1);
		std::string whichFileName;
		amrex::Vector<amrex::Long> currentOffset(nProcs, 0L);

		// ---- find the length of the fab header instead of asking the file system
		for (int i(0); i < mfBA.size(); ++i) {
			std::stringstream hss;
			amrex::FArrayBox const tempFab(mf.fabbox(i), nComps, false); // ---- no alloc
			write_2D_header(hss, tempFab, tempFab.nComp());
			fabHeaderBytes[i] = static_cast<std::streamoff>(hss.tellp());
		}

		std::map<int, amrex::Vector<int>> rankBoxOrder; // ---- [rank, boxarray index array]
		for (int i(0); i < mfBA.size(); ++i) {
			rankBoxOrder[mfDM[i]].push_back(i);
		}

		amrex::Vector<int> fileNumbers;
		if (nfi.GetDynamic()) {
			fileNumbers = nfi.FileNumbersWritten();
		} else if (nfi.GetSparseFPP()) { // if sparse, write to (file number = rank)
			fileNumbers.resize(nProcs);
			for (int i(0); i < nProcs; ++i) {
				fileNumbers[i] = i;
			}
		} else {
			fileNumbers.resize(nProcs);
			for (int i(0); i < nProcs; ++i) {
				fileNumbers[i] = amrex::NFilesIter::FileNumber(nFiles, i, groupSets);
			}
		}
		const amrex::Vector<amrex::Vector<int>> &fileNumbersWriteOrder = nfi.FileNumbersWriteOrder();

		for (const auto &fn : fileNumbersWriteOrder) {
			for (const int rank : fn) {
				auto rboIter = rankBoxOrder.find(rank);

				if (rboIter != rankBoxOrder.end()) {
					amrex::Vector<int> const &index = rboIter->second;
					whichFileNumber = fileNumbers[rank];
					whichFileName = amrex::VisMF::BaseName(amrex::NFilesIter::FileName(whichFileNumber, filePrefix));

					for (const int idx : index) {
						hdr.m_fod[idx].m_name = whichFileName;
						hdr.m_fod[idx].m_head = currentOffset[whichFileNumber];
						currentOffset[whichFileNumber] += mf.fabbox(idx).numPts() * nComps * whichRDBytes + fabHeaderBytes[idx];
					}
				}
			}
		}
	}
}

void write_2D_header(std::ostream &os, const amrex::FArrayBox &f, int nvar)
{
	os << "FAB " << amrex::FPC::NativeRealDescriptor();
	amrex::StreamRetry sr(os, "FABio_write_header", 4);
	while (sr.TryOutput()) {
		printLowerDimBox(os, f.box(), 2);
		os << ' ' << nvar << '\n';
	}
}

auto transform_box_to_2D(amrex::Direction const &dir, amrex::Box const &box) -> amrex::Box
{
	// transform box dimensions (Nx, Ny, Nz) -> (Nx', Ny', 1) where *one* of Nx, Ny, Nz == 1.
	// NOTE: smallBox is assumed to be {0, 0, 0}.
	amrex::IntVect dim = box.bigEnd();
	amrex::IntVect bigEnd;

	if (dir == amrex::Direction::x) { // y-z plane
		bigEnd = amrex::IntVect(amrex::Dim3{dim[1], dim[2], 0});
	} else if (dir == amrex::Direction::y) { // x-z plane
		bigEnd = amrex::IntVect(amrex::Dim3{dim[0], dim[2], 0});
	} else if (dir == amrex::Direction::z) { // x-y plane
		bigEnd = amrex::IntVect(amrex::Dim3{dim[0], dim[1], 0});
	} else {
		amrex::Abort("detail::transform_box_to_2D: invalid direction!");
	}

	return amrex::Box(amrex::IntVect(amrex::Dim3{0, 0, 0}), bigEnd);
}

auto transform_realbox_to_2D(amrex::Direction const &dir, amrex::RealBox const &box) -> amrex::RealBox
{
	// transform box dimensions (Nx, Ny, Nz) -> (Nx', Ny', 1) where *one* of Nx, Ny, Nz == 1.
	// NOTE: smallBox is assumed to be {0, 0, 0}.
	amrex::Real const *hi = box.hi();
	amrex::Real const *lo = box.lo();
	std::array<amrex::Real, 3> new_hi{};
	std::array<amrex::Real, 3> new_lo{};

	if (dir == amrex::Direction::x) { // y-z plane
		new_lo = {lo[1], lo[2], lo[0]};
		new_hi = {hi[1], hi[2], hi[0]};
	} else if (dir == amrex::Direction::y) { // x-z plane
		new_lo = {lo[0], lo[2], lo[1]};
		new_hi = {hi[0], hi[2], hi[1]};
	} else if (dir == amrex::Direction::z) { // x-y plane
		new_lo = {lo[0], lo[1], lo[2]};
		new_hi = {hi[0], hi[1], hi[2]};
	} else {
		amrex::Abort("detail::transform_box_to_2D: invalid direction!");
	}

	return amrex::RealBox(new_lo, new_hi);
}

} // namespace detail

void WriteProjection(const amrex::Direction dir, std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> const &proj, amrex::Real time, int istep)
{
	// write projections to plotfile
	auto const &firstFab = proj.begin()->second;
	amrex::Vector<std::string> varnames;

	// NOTE: Write2DMultiLevelPlotfile assumes the slice lies in the x-y plane
	//  (i.e. normal to the z axis) and the Geometry object corresponds to this.
	//  For a z-projection, this works as expected. For an {x,y}-projection,
	//  it is necessary to transform the geometry so that the data is stored in
	//  the x-y plane.
	amrex::Geometry geom3d{};
	geom3d.Setup(); // read from ParmParse
	const amrex::Box box2d = detail::transform_box_to_2D(dir, firstFab.box());
	const amrex::RealBox domain2d = detail::transform_realbox_to_2D(dir, geom3d.ProbDomain());
	const amrex::Geometry geom2d(box2d, &domain2d);
	//amrex::Print() << box2d << "\n";
	//amrex::Print() << domain2d << "\n";

	// construct output multifab on rank 0
	const amrex::BoxArray ba(box2d);
	const amrex::DistributionMapping dm(amrex::Vector<int>{0});
	const int ncomp = static_cast<int>(proj.size());
	amrex::MultiFab mf_all(ba, dm, ncomp, 0);

	// copy all projections into a single Multifab with x-y geometry
	auto iter = proj.begin();
	for (int icomp = 0; icomp < ncomp; ++icomp) {
		const std::string &varname = iter->first;
		const amrex::BaseFab<amrex::Real> &baseFab = iter->second;
		varnames.push_back(varname);
		//amrex::Print() << "varname: " << varname << " icomp: " << icomp << "\n";

		// copy mf_comp into mf_all
		auto output_arr = mf_all.arrays();
		auto const &input_arr = baseFab.const_array();

		if (dir == amrex::Direction::x) {
			amrex::ParallelFor(
			    mf_all, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output_arr[bx](i, j, k, icomp) = input_arr[bx](0, i, j); });
		} else if (dir == amrex::Direction::y) {
			amrex::ParallelFor(
			    mf_all, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output_arr[bx](i, j, k, icomp) = input_arr[bx](i, 0, j); });
		} else if (dir == amrex::Direction::z) {
			amrex::ParallelFor(
			    mf_all, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept { output_arr[bx](i, j, k, icomp) = input_arr[bx](i, j, 0); });
		}
		amrex::Gpu::streamSynchronize();

		++iter;
	}

	// write mf_all to disk
	const std::string basename = "proj_" + detail::direction_to_string(dir) + "_plt";
	const std::string filename = amrex::Concatenate(basename, istep, 5);
	const amrex::Vector<const amrex::MultiFab *> mfs = {&mf_all};
	amrex::Print() << "Writing projection " << filename << "\n";

	detail::Write2DMultiLevelPlotfile(filename, 1, mfs, varnames, {geom2d}, time, {istep}, {});
}

} // namespace quokka::diagnostics
