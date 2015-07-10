/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_TRANS_HEADER_INCLUDED
#define OPM_TRANS_HEADER_INCLUDED

#include <opm/core/grid.h>
#include <opm/autodiff/GridHelpers.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/core/pressure/tpfa/trans_tpfa.h>
#include <opm/core/pressure/tpfa/TransTpfa.hpp>
#include <opm/core/utility/Units.hpp>
#include <opm/parser/eclipse/EclipseState/EclipseState.hpp>
#include <opm/parser/eclipse/EclipseState/Grid/EclipseGrid.hpp>
#include <opm/core/utility/platform_dependent/disable_warnings.h>

#include <Eigen/Eigen>

#ifdef HAVE_DUNE_CORNERPOINT
#include <dune/common/version.hh>
#include <dune/grid/CpGrid.hpp>
#include <dune/grid/common/mcmgmapper.hh>
#endif

#include <opm/core/utility/platform_dependent/reenable_warnings.h>

#include <cstddef>
#include <cmath>

namespace Opm
{

    /// Class containing static geological properties that are
    /// derived from grid and petrophysical properties:
    ///   - htrans
    ///   - transmissibilities
    class Trans
    {
    public:
        typedef Eigen::ArrayXd Vector;
        /// Construct contained derived geological properties
        /// from grid and property information.
        template <class Props, class Grid>
        Trans(const Grid&              grid,
              const Props&             props ,
              Opm::EclipseStateConstPtr eclState)
            : htrans_(AutoDiffGrid::numCellFaces(grid))
            , trans_(Opm::AutoDiffGrid::numFaces(grid))
            , tranx_(Opm::AutoDiffGrid::numCells(grid))
            , trany_(Opm::AutoDiffGrid::numCells(grid))
            , tranz_(Opm::AutoDiffGrid::numCells(grid))
        {
            int numCells = AutoDiffGrid::numCells(grid);
            int numFaces = AutoDiffGrid::numFaces(grid);
            const int *cartDims = AutoDiffGrid::cartDims(grid);
            int numCartesianCells =
                cartDims[0]
                * cartDims[1]
                * cartDims[2];

            // get the net-to-gross cell thickness from the EclipseState
            std::vector<double> ntg(numCartesianCells, 1.0);
            if (eclState->hasDoubleGridProperty("NTG")) {
                ntg = eclState->getDoubleGridProperty("NTG")->getData();
            }

            
            // Get original grid cell volume.
            EclipseGridConstPtr eclgrid = eclState->getEclipseGrid();
            // Transmissibility

            Grid* ug = const_cast<Grid*>(& grid);

            //            if (! use_local_perm) {
            //                tpfa_htrans_compute(ug, props.permeability(), htrans.data());
                //            }
                //            else {
            tpfa_loc_trans_compute_(grid,props.permeability(),htrans_);
                //            }

            std::vector<double> mult;
            multiplyHalfIntersections_(grid, eclState, ntg, htrans_, mult);

            // combine the half-face transmissibilites into the final face
            // transmissibilites.
            tpfa_trans_compute(ug, htrans_.data(), trans_.data());

            // multiply the face transmissibilities with their appropriate
            // transmissibility multipliers
            for (int faceIdx = 0; faceIdx < numFaces; faceIdx++) {
                trans_[faceIdx] *= mult[faceIdx];
            }
            // Convert to metric unit.
            const double factor = Opm::prefix::centi * Opm::unit::Poise
            / Opm::unit::cubic(Opm::unit::meter)
            / Opm::unit::day
            / Opm::unit::barsa;
            for (size_t  idx = 0; idx < trans_.size(); ++idx) {
                trans_[idx] = unit::convert::to(trans_[idx], factor);
                if (std::fabs(trans_[idx]) < 1e-10) {
                    trans_[idx] = 0.0;
                }
            }
            for (size_t  idx = 0; idx < htrans_.size(); ++idx) {
                htrans_[idx] = unit::convert::to(htrans_[idx], factor);
                if (std::fabs(htrans_[idx]) < 1e-10) {
                    htrans_[idx] = 0.0;
                }
            }
            std::cout << "Num Cells: " << Opm::AutoDiffGrid::numCells(grid) << std::endl;
            std::cout << "Num Faces: " << Opm::AutoDiffGrid::numFaces(grid) << std::endl;
            std::cout << "Num Cell Faces: " << AutoDiffGrid::numCellFaces(grid) << std::endl;
            getCellTrans(grid, trans_);

        }
        const Vector& transmissibility() const { return trans_  ;}
        const Vector& htrans()           const { return htrans_ ;}
        const Vector& tranx()            const { return tranx_; }
        const Vector& trany()            const { return trany_; }
        const Vector& tranz()            const { return tranz_; }
        Vector&       transmissibility()       { return trans_  ;}
        Vector&       htrans()                 { return htrans_ ;}
        Vector& tranx()            { return tranx_; }
        Vector& trany()            { return trany_; }
        Vector& tranz()            { return tranz_; }
  
    private:
        template <class Grid>
        void multiplyHalfIntersections_(const Grid &grid,
                                        Opm::EclipseStateConstPtr eclState,
                                        const std::vector<double> &ntg,
                                        Vector &halfIntersectTransmissibility,
                                        std::vector<double> &intersectionTransMult);

        template <class Grid>
        void tpfa_loc_trans_compute_(const Grid &grid,
                                     const double* perm,
                                     Vector &hTrans);

        template <class Grid>
        void getCellTrans(const Grid& grid, const Vector& trans);

        Vector trans_;
        Vector htrans_;
        Vector tranx_;
        Vector trany_;
        Vector tranz_;
    };


    template <class GridType>
    inline void Trans::multiplyHalfIntersections_(const GridType &grid,
                                                   Opm::EclipseStateConstPtr eclState,
                                                   const std::vector<double> &ntg,
                                                   Vector &halfIntersectTransmissibility,
                                                   std::vector<double> &intersectionTransMult)
    {
        int numCells = Opm::AutoDiffGrid::numCells(grid);

        int numIntersections = Opm::AutoDiffGrid::numFaces(grid);
        intersectionTransMult.resize(numIntersections);
        std::fill(intersectionTransMult.begin(), intersectionTransMult.end(), 1.0);

        std::shared_ptr<const Opm::TransMult> multipliers = eclState->getTransMult();
        auto cell2Faces = Opm::UgGridHelpers::cell2Faces(grid);
        auto faceCells  = Opm::AutoDiffGrid::faceCells(grid);
        const int* global_cell = Opm::UgGridHelpers::globalCell(grid);
        int cellFaceIdx = 0;

        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            // loop over all logically-Cartesian faces of the current cell
            auto cellFacesRange = cell2Faces[cellIdx];

            for(auto cellFaceIter = cellFacesRange.begin(), cellFaceEnd = cellFacesRange.end();
                cellFaceIter != cellFaceEnd; ++cellFaceIter, ++cellFaceIdx)
            {
                // the index of the current cell in arrays for the logically-Cartesian grid
                int cartesianCellIdx = global_cell[cellIdx];

                // The index of the face in the compressed grid
                int faceIdx = *cellFaceIter;

                // the logically-Cartesian direction of the face
                int faceTag = Opm::UgGridHelpers::faceTag(grid, cellFaceIter);

                // Translate the C face tag into the enum used by opm-parser's TransMult class
                Opm::FaceDir::DirEnum faceDirection;
                if (faceTag == 0) // left
                    faceDirection = Opm::FaceDir::XMinus;
                else if (faceTag == 1) // right
                    faceDirection = Opm::FaceDir::XPlus;
                else if (faceTag == 2) // back
                    faceDirection = Opm::FaceDir::YMinus;
                else if (faceTag == 3) // front
                    faceDirection = Opm::FaceDir::YPlus;
                else if (faceTag == 4) // bottom
                    faceDirection = Opm::FaceDir::ZMinus;
                else if (faceTag == 5) // top
                    faceDirection = Opm::FaceDir::ZPlus;
                else
                    OPM_THROW(std::logic_error, "Unhandled face direction: " << faceTag);

                // Account for NTG in horizontal one-sided transmissibilities
                switch (faceDirection) {
                case Opm::FaceDir::XMinus:
                case Opm::FaceDir::XPlus:
                case Opm::FaceDir::YMinus:
                case Opm::FaceDir::YPlus:
                    halfIntersectTransmissibility[cellFaceIdx] *= ntg[cartesianCellIdx];
                    break;
                default:
                    // do nothing for the top and bottom faces
                    break;
                }

                // Multiplier contribution on this face for MULT[XYZ] logical cartesian multipliers
                intersectionTransMult[faceIdx] *=
                    multipliers->getMultiplier(cartesianCellIdx, faceDirection);

                // Multiplier contribution on this fase for region multipliers
                const int cellIdxInside  = faceCells(faceIdx, 0);
                const int cellIdxOutside = faceCells(faceIdx, 1);

                // Do not apply region multipliers in the case of boundary connections
                if (cellIdxInside < 0 || cellIdxOutside < 0) {
                    continue;
                }
                const int cartesianCellIdxInside = global_cell[cellIdxInside];
                const int cartesianCellIdxOutside = global_cell[cellIdxOutside];
                //  Only apply the region multipliers from the inside
                if (cartesianCellIdx == cartesianCellIdxInside) {
                    intersectionTransMult[faceIdx] *= multipliers->getRegionMultiplier(cartesianCellIdxInside,cartesianCellIdxOutside,faceDirection);
                }


            }
        }
    }

    template <class GridType>
    inline void Trans::getCellTrans(const GridType& grid, const Vector& faceTrans)
    {

        int numCells = Opm::AutoDiffGrid::numCells(grid);
        int numFaces = Opm::AutoDiffGrid::numFaces(grid);
        // allocate space
        const int *cartDims = AutoDiffGrid::cartDims(grid);
        const int nx = cartDims[0];
        const int ny = cartDims[1];
        const int nz = cartDims[2];
        auto faceCells  = Opm::AutoDiffGrid::faceCells(grid);
        auto globalCell = Opm::AutoDiffGrid::globalCell(grid);
        // fill the arrays
        for (int faceIdx = 0; faceIdx < numFaces; ++faceIdx) {
            // calculate the logically Cartesian IJK position of the inside and outside
            // cells of the face
 
            //int insideCellIdx = Opm::AutoDiffGrid::faceCells(grid)face_cells[2*faceIdx + 0];
            //int outsideCellIdx = grid.face_cells[2*faceIdx + 1];
            const int insideCellIdx  = faceCells(faceIdx, 0);
            const int outsideCellIdx = faceCells(faceIdx, 1);

            if (insideCellIdx < 0 || outsideCellIdx < 0)
                continue; // ignore boundary faces

            int cartesianInsideCellIdx =  globalCell[insideCellIdx];
            int cartesianOutsideCellIdx =  globalCell[outsideCellIdx];

            int iIn = cartesianInsideCellIdx%nx;
            int jIn = cartesianInsideCellIdx/nx % ny;
            int kIn = cartesianInsideCellIdx/(nx*ny);

            int iOut = cartesianOutsideCellIdx%nx;
            int jOut = cartesianOutsideCellIdx/nx % ny;
            int kOut = cartesianOutsideCellIdx/(nx*ny);

            bool isNNC = true;
            // try to detect "ordinary" neighbors. This is the case if
            // the logically Cartesian indices of the cell differ only
            // by one in a single direction. The depth axis is special
            // as cells can be skipped (e.g. due to pinchouts) and the
            // direction seems to be reverse...
            if (jIn == jOut && kIn == kOut) {
                if (iIn + 1 == iOut) {
                    isNNC=false;
                    tranx_[insideCellIdx] = faceTrans[faceIdx];
                }
                else if (iOut + 1 == iIn) {
                    isNNC=false;
                    tranx_[outsideCellIdx] = faceTrans[faceIdx];
                }
            }
            else if (iIn == iOut && kIn == kOut) {
                if (jIn + 1 == jOut) {
                    isNNC=false;
                    trany_[insideCellIdx] = faceTrans[faceIdx];
                }
                else if (jOut + 1 == jIn) {
                    isNNC=false;
                    trany_[outsideCellIdx] = faceTrans[faceIdx];
                }
            }
            else if (iIn == iOut && jIn == jOut) {
                isNNC=false;
                if (kIn < kOut) {
                    tranz_[insideCellIdx] = faceTrans[faceIdx];
                }
                else {
                    tranz_[outsideCellIdx] = faceTrans[faceIdx];
                }
            }
        }
    }


    template <class GridType>
    inline void Trans::tpfa_loc_trans_compute_(const GridType& grid,
                                               const double* perm,
                                               Vector& hTrans){

        // Using Local coordinate system for the transmissibility calculations
        // hTrans(cellFaceIdx) = K(cellNo,j) * sum( C(:,i) .* N(:,j), 2) / sum(C.*C, 2)
        // where K is a diagonal permeability tensor, C is the distance from cell centroid
        // to face centroid and N is the normal vector  pointing outwards with norm equal to the face area.
        // Off-diagonal permeability values are ignored without warning
        int numCells = AutoDiffGrid::numCells(grid);
        int cellFaceIdx = 0;
        auto cell2Faces = Opm::UgGridHelpers::cell2Faces(grid);
        auto faceCells = Opm::UgGridHelpers::faceCells(grid);

        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            // loop over all logically-Cartesian faces of the current cell
            auto cellFacesRange = cell2Faces[cellIdx];

            for(auto cellFaceIter = cellFacesRange.begin(), cellFaceEnd = cellFacesRange.end();
                cellFaceIter != cellFaceEnd; ++cellFaceIter, ++cellFaceIdx)
            {
                // The index of the face in the compressed grid
                const int faceIdx = *cellFaceIter;

                // the logically-Cartesian direction of the face
                const int faceTag = Opm::UgGridHelpers::faceTag(grid, cellFaceIter);

                // d = 0: XPERM d = 4: YPERM d = 8: ZPERM ignores off-diagonal permeability values.
                const int d = std::floor(faceTag/2) * 4;

                // compute the half transmissibility
                double dist = 0.0;
                double cn = 0.0;
                double sgn = 2.0 * (faceCells(faceIdx, 0) == cellIdx) - 1;
                const int dim = Opm::UgGridHelpers::dimensions(grid);
                for (int indx = 0; indx < dim; ++indx) {
                    const double Ci = Opm::UgGridHelpers::faceCentroid(grid, faceIdx)[indx] - 
                        Opm::UgGridHelpers::cellCentroidCoordinate(grid, cellIdx, indx);
                    dist += Ci*Ci;
                    cn += sgn * Ci * Opm::UgGridHelpers::faceNormal(grid, faceIdx)[indx];
                }

                if (cn < 0){
                    switch (d) {
                    case 0:
                        OPM_MESSAGE("Warning: negative X-transmissibility value in cell: " << cellIdx << " replace by absolute value") ;
                                break;
                    case 4:
                        OPM_MESSAGE("Warning: negative Y-transmissibility value in cell: " << cellIdx << " replace by absolute value") ;
                                break;
                    case 8:
                        OPM_MESSAGE("Warning: negative Z-transmissibility value in cell: " << cellIdx << " replace by absolute value") ;
                                break;
                    default:
                        OPM_THROW(std::logic_error, "Inconsistency in the faceTag in cell: " << cellIdx);

                    }
                    cn = -cn;
                }
                hTrans[cellFaceIdx] = perm[cellIdx*dim*dim + d] * cn / dist;
            }
        }

    }

}

#endif // OPM_TRANS_HEADER_INCLUDED
