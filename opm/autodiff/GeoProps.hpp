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

#ifndef OPM_GEOPROPS_HEADER_INCLUDED
#define OPM_GEOPROPS_HEADER_INCLUDED

#include <opm/core/grid.h>
#include <opm/autodiff/GridHelpers.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
//#include <opm/core/pressure/tpfa/trans_tpfa.h>
#include <opm/core/pressure/tpfa/TransTpfa.hpp>

#include <opm/parser/eclipse/EclipseState/EclipseState.hpp>
#include <opm/parser/eclipse/Deck/DeckKeyword.hpp>

#include <opm/core/utility/platform_dependent/disable_warnings.h>

#include <Eigen/Eigen>

#ifdef HAVE_DUNE_CORNERPOINT
#include <dune/common/version.hh>
#include <dune/grid/CpGrid.hpp>
#include <dune/grid/common/mcmgmapper.hh>
#endif

#include <opm/core/utility/platform_dependent/reenable_warnings.h>

#include <cstddef>

namespace Opm
{

    /// Class containing static geological properties that are
    /// derived from grid and petrophysical properties:
    ///   - pore volume
    ///   - transmissibilities
    ///   - gravity potentials
    class DerivedGeology
    {
    public:
        typedef Eigen::ArrayXd Vector;

        /// Construct contained derived geological properties
        /// from grid and property information.
        template <class Props, class Grid>
        DerivedGeology(const Grid&              grid,
                       const Props&             props ,
                       Opm::EclipseStateConstPtr eclState,
                       const bool               use_local_perm,
                       const double*            grav = 0

                )
            : pvol_ (Opm::AutoDiffGrid::numCells(grid))
            , trans_(Opm::AutoDiffGrid::numFaces(grid))
            , gpot_ (Vector::Zero(Opm::AutoDiffGrid::cell2Faces(grid).noEntries(), 1))
            , z_(Opm::AutoDiffGrid::numCells(grid))
        {
            int numCells = AutoDiffGrid::numCells(grid);
            int numFaces = AutoDiffGrid::numFaces(grid);
            const int *cartDims = AutoDiffGrid::cartDims(grid);
            int numCartesianCells =
                cartDims[0]
                * cartDims[1]
                * cartDims[2];
            
            //output grid information.
            std::cout << "num cells: " << numCells << " , num cart cells "<<numCartesianCells << "  cartDims: "
                      << cartDims[0] << ", "<<cartDims[1] << ", " << cartDims[2] << std::endl;
            // output the cell index.
            //            std::cout << "global cell index:" << std::endl;
            // for (int i = 0; i < numCells; ++i) {
            //     std::cout << AutoDiffGrid::globalCell(grid)[i] << std::endl;
            //         }
            // output cartesian cell index.
            for (int cell = 0; cell < numCells; ++cell) {
                int nx = cartDims[0];
                int ny = cartDims[1];
                int nz = cartDims[2];
                int cartIndex = AutoDiffGrid::globalCell(grid)[cell];
                int i = cartIndex% nx;
                int j = cartIndex/nx % ny;
                int k = cartIndex/(nx*ny) % nz;
                std::cout << i << "    " << j << "    " << k << std::endl;

            }
            exit(1);
            // get the pore volume multipliers from the EclipseState
            std::vector<double> multpv(numCartesianCells, 1.0);
            if (eclState->hasDoubleGridProperty("MULTPV")) {
                multpv = eclState->getDoubleGridProperty("MULTPV")->getData();
            }

            // get the net-to-gross cell thickness from the EclipseState
            std::vector<double> ntg(numCartesianCells, 1.0);
            if (eclState->hasDoubleGridProperty("NTG")) {
                ntg = eclState->getDoubleGridProperty("NTG")->getData();
            }

            // Pore volume
            for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
                int cartesianCellIdx = AutoDiffGrid::globalCell(grid)[cellIdx];
                pvol_[cellIdx] =
                    props.porosity()[cellIdx]
                    * multpv[cartesianCellIdx]
                    * ntg[cartesianCellIdx]
                    * AutoDiffGrid::cellVolume(grid, cellIdx);
            }

            // Transmissibility

            Vector htrans(AutoDiffGrid::numCellFaces(grid));
            Grid* ug = const_cast<Grid*>(& grid);

            if (! use_local_perm) {
                tpfa_htrans_compute(ug, props.permeability(), htrans.data());
            }
            else {
                tpfa_loc_trans_compute_(grid,props.permeability(),htrans);
            }

            std::vector<double> mult;
            multiplyHalfIntersections_(grid, eclState, ntg, htrans, mult);

            // combine the half-face transmissibilites into the final face
            // transmissibilites.
            tpfa_trans_compute(ug, htrans.data(), trans_.data());

            // multiply the face transmissibilities with their appropriate
            // transmissibility multipliers
            for (int faceIdx = 0; faceIdx < numFaces; faceIdx++) {
                trans_[faceIdx] *= mult[faceIdx];
            }

            // Compute z coordinates
            for (int c = 0; c<numCells; ++c){
                z_[c] = Opm::UgGridHelpers::cellCentroidCoordinate(grid, c, 2);
            }


            // Gravity potential
            std::fill(gravity_, gravity_ + 3, 0.0);
            if (grav != 0) {
                const typename Vector::Index nd = AutoDiffGrid::dimensions(grid);
                typedef typename AutoDiffGrid::ADCell2FacesTraits<Grid>::Type Cell2Faces;
                Cell2Faces c2f=AutoDiffGrid::cell2Faces(grid);

                std::size_t i = 0;
                for (typename Vector::Index c = 0; c < numCells; ++c) {
                    const double* const cc = AutoDiffGrid::cellCentroid(grid, c);

                    typename Cell2Faces::row_type faces=c2f[c];
                    typedef typename Cell2Faces::row_type::iterator Iter;

                    for (Iter f=faces.begin(), end=faces.end(); f!=end; ++f, ++i) {
                        const double* const fc = AutoDiffGrid::faceCentroid(grid, *f);

                        for (typename Vector::Index d = 0; d < nd; ++d) {
                            gpot_[i] += grav[d] * (fc[d] - cc[d]);
                        }
                    }
                }
                std::copy(grav, grav + nd, gravity_);
            }
        }

        // TODO: variant of this method for Dune::CpGrid
        void setTranx(const UnstructuredGrid& grid, const std::vector<double>& tranxValues)
        {
            setTran_(grid, tranxValues, /*axis=*/0);
        }

        // TODO: variant of this method for Dune::CpGrid
        void setTrany(const UnstructuredGrid& grid, const std::vector<double>& tranyValues)
        {
            setTran_(grid, tranyValues, /*axis=*/1);
        }

        // TODO: variant of this method for Dune::CpGrid
        void setTranz(const UnstructuredGrid& grid, const std::vector<double>& tranzValues)
        {
            setTran_(grid, tranzValues, /*axis=*/2);
        }

        // TODO: variant of this method for Dune::CpGrid
        void setNnc(const UnstructuredGrid& grid, Opm::DeckKeywordConstPtr nncKeyword)
        {
            int nx = grid.cartdims[0];
            int ny = grid.cartdims[1];
            //int nz = grid.cartdims[2];

            // create a map from a pair of cell indices to a transmissibility value.
            std::map< std::pair<int, int>, double> faceToValueMap;
            for (int recordIdx = 0; recordIdx < nncKeyword->size(); ++recordIdx) {
                auto nncRecord = nncKeyword->getRecord(recordIdx);
                int cellIdx1 =
                    (nncRecord->getItem("I1")->getInt(0) - 1) +
                    (nncRecord->getItem("J1")->getInt(0) - 1)*nx +
                    (nncRecord->getItem("K1")->getInt(0) - 1)*nx*ny;
                int cellIdx2 =
                    (nncRecord->getItem("I2")->getInt(0) - 1) +
                    (nncRecord->getItem("J2")->getInt(0) - 1)*nx +
                    (nncRecord->getItem("K2")->getInt(0) - 1)*nx*ny;

                double tranValue = nncRecord->getItem("TRAN")->getSIDouble(0);

                std::pair<int, int> indices(cellIdx1, cellIdx2);
                faceToValueMap[indices] = tranValue;
            }

            // iterate over all faces of the grid and apply the transmissibilities to
            // those found in the map
            for (int faceIdx = 0; faceIdx < grid.number_of_faces; ++faceIdx) {
                int insideCellIdx = grid.face_cells[2*faceIdx + 0];
                int outsideCellIdx = grid.face_cells[2*faceIdx + 1];

                if (insideCellIdx < 0 || outsideCellIdx < 0) {
                    // ignore boundary faces
                    continue;
                }

                // retrieve the "uncompressed" cell indices
                int insideCellCartesianIdx = insideCellIdx;
                int outsideCellCartesianIdx = outsideCellIdx;
                if (grid.global_cell) {
                    insideCellCartesianIdx = grid.global_cell[insideCellIdx];
                    outsideCellCartesianIdx = grid.global_cell[outsideCellIdx];
                }

                std::pair<int, int> indices(insideCellCartesianIdx, outsideCellCartesianIdx);
                if (faceToValueMap.count(indices) == 0)
                    indices = std::pair<int, int>(outsideCellCartesianIdx, insideCellCartesianIdx);
                if (faceToValueMap.count(indices) == 0)
                    continue; // face is not in the map

                trans_[faceIdx] = faceToValueMap.at(indices);

                // this is hacky, but the whole method is primarily intended as a
                // debugging tool anyway...
                if (faceToValueMap.at(indices) < -1e50) {
                    std::cout << "Connection from cell " << insideCellCartesianIdx << " to cell "
                              << outsideCellCartesianIdx << " appears more than once!\n";
                }
                faceToValueMap[indices] = -1e100;
            }

            // print the NNCs which are not present in the grid but which have been
            // specified by the NNC keyword...
            auto nncIt = faceToValueMap.begin();
            const auto& nncEndIt = faceToValueMap.end();
            for (; nncIt != nncEndIt; ++ nncIt) {
                if (nncIt->second < -1e50)
                    continue;

                int cartesianIdx1 = nncIt->first.first;
                int cartesianIdx2 = nncIt->first.second;

                int i1 = cartesianIdx1 % nx;
                int j1 = cartesianIdx1/nx % ny;
                int k1 = cartesianIdx1/(nx*ny);

                int i2 = cartesianIdx2 % nx;
                int j2 = cartesianIdx2/nx % ny;
                int k2 = cartesianIdx2/(nx*ny);

                std::cout << "No connection from cell ("
                          << i1+1 << ", "
                          << j1+1 << ", "
                          << k1+1
                          << ") to cell ("
                          << i2+1 << ", "
                          << j2+1 << ", "
                          << k2+1
                          << ") found in grid, but specified in the NNC keyword!\n";
            }
        }

        const Vector& poreVolume()       const { return pvol_   ;}
        const Vector& transmissibility() const { return trans_  ;}
        const Vector& gravityPotential() const { return gpot_   ;}
        const Vector& z()                const { return z_      ;}
        const double* gravity()          const { return gravity_;}

    private:
        void assertLogicallyCartesianData_(const UnstructuredGrid& grid,
                                           const std::vector<double>& tranxValues)
        {
            int nx = grid.cartdims[0];
            int ny = grid.cartdims[1];
            int nz = grid.cartdims[2];

            if (tranxValues.size() != nx*ny*nz)
                OPM_THROW(std::runtime_error,
                          "The array of TRANX values must of size " << nx*ny*nz
                          << "(is: " << tranxValues.size());
        }

        // TODO: variant of this method for Dune::CpGrid
        void setTran_(const UnstructuredGrid& grid,
                      const std::vector<double>& tranValues,
                      int axis)
        {
            assertLogicallyCartesianData_(grid, tranValues);

            for (int faceIdx = 0; faceIdx < grid.number_of_faces; ++faceIdx) {
                int insideCellIdx = grid.face_cells[2*faceIdx + 0];
                int outsideCellIdx = grid.face_cells[2*faceIdx + 1];

                if (insideCellIdx < 0 || outsideCellIdx < 0) {
                    // ignore boundary faces
                    continue;
                }

                // retrieve the "uncompressed" cell indices
                int insideCellCartesianIdx = insideCellIdx;
                int outsideCellCartesianIdx = outsideCellIdx;
                if (grid.global_cell) {
                    insideCellCartesianIdx = grid.global_cell[insideCellIdx];
                    outsideCellCartesianIdx = grid.global_cell[outsideCellIdx];
                }

                int neighborDir = neighborDirection_(grid,
                                                     insideCellCartesianIdx,
                                                     outsideCellCartesianIdx);

                if (neighborDir == axis + 0) { // + direction face
                    if (std::isfinite(tranValues[insideCellCartesianIdx])) {
                        trans_[faceIdx] = tranValues[insideCellCartesianIdx];
                    }
                }
                else if (neighborDir == axis + 3) { // - direction face
                    if (std::isfinite(tranValues[outsideCellCartesianIdx])) {
                        trans_[faceIdx] = tranValues[outsideCellCartesianIdx];
                    }
                }
            }
        }

        int neighborDirection_(const UnstructuredGrid& grid,
                               const int insideCartesianIdx,
                               const int outsideCartesianIdx) const
        {
            int nx = grid.cartdims[0];
            int ny = grid.cartdims[1];
            int nz = grid.cartdims[2];

            int insideI = insideCartesianIdx%nx;
            int insideJ = (insideCartesianIdx/nx)%ny;
            int insideK = (insideCartesianIdx/nx/ny)%nz;

            int outsideI = outsideCartesianIdx%nx;
            int outsideJ = (outsideCartesianIdx/nx)%ny;
            int outsideK = (outsideCartesianIdx/nx/ny)%nz;

            if (insideJ == outsideJ && insideK == outsideK) {
                if (insideI + 1 == outsideI)
                    return 0; // +X direction
                else if (insideI == outsideI + 1)
                    return 3; // -X direction
            }
            else if (insideI == outsideI && insideK == outsideK) {
                if (insideJ + 1 == outsideJ)
                    return 1; // +Y direction
                else if (insideJ == outsideJ + 1)
                    return 4; // -Y direction
            }
            else if (insideI == outsideI && insideJ == outsideJ) {
                if (insideK > outsideJ)
                    return 2; // +Z direction
                else
                    return 6; // -Z direction
            }

            return -1; // non-neighboring connection
        }

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

        Vector pvol_ ;
        Vector trans_;
        Vector gpot_ ;
        Vector z_;
        double gravity_[3]; // Size 3 even if grid is 2-dim.




    };


    template <class GridType>
    inline void DerivedGeology::multiplyHalfIntersections_(const GridType &grid,
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
    inline void DerivedGeology::tpfa_loc_trans_compute_(const GridType& grid,
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



#endif // OPM_GEOPROPS_HEADER_INCLUDED
