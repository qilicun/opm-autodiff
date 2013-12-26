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

#include <opm/autodiff/twophase/FullyImplicitCompressibleSolver.hpp>


#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/autodiff/BlackoilPropsAdInterface.hpp>
#include <opm/autodiff/GeoProps.hpp>

#include <opm/core/grid.h>
#include <opm/core/linalg/LinearSolverInterface.hpp>
#include <opm/core/props/rock/RockCompressibility.hpp>
#include <opm/core/simulator/BlackoilState.hpp>
#include <opm/core/simulator/WellState.hpp>
#include <opm/core/utility/ErrorMacros.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

// A debugging utility.
#define DUMP(foo)                                                       \
    do {                                                                \
        std::cout << "==========================================\n"     \
                  << #foo ":\n"                                         \
                  << collapseJacs(foo) << std::endl;                    \
    } while (0)



namespace Opm {

typedef AutoDiffBlock<double> ADB;
typedef ADB::V V;
typedef ADB::M M;
typedef Eigen::Array<double,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::RowMajor> DataBlock;


namespace {


    std::vector<int>
    buildAllCells(const int nc)
    {
        std::vector<int> all_cells(nc);

        for (int c = 0; c < nc; ++c) { all_cells[c] = c; }

        return all_cells;
    }



    template <class GeoProps>
    AutoDiffBlock<double>::M
    gravityOperator(const UnstructuredGrid& grid,
                    const HelperOps&        ops ,
                    const GeoProps&         geo )
    {
        const int nc = grid.number_of_cells;

        std::vector<int> f2hf(2 * grid.number_of_faces, -1);
        for (int c = 0, i = 0; c < nc; ++c) {
            for (; i < grid.cell_facepos[c + 1]; ++i) {
                const int f = grid.cell_faces[ i ];
                const int p = 0 + (grid.face_cells[2*f + 0] != c);

                f2hf[2*f + p] = i;
            }
        }

        typedef AutoDiffBlock<double>::V V;
        typedef AutoDiffBlock<double>::M M;

        const V& gpot  = geo.gravityPotential();
        const V& trans = geo.transmissibility();

        const HelperOps::IFaces::Index ni = ops.internal_faces.size();

        typedef Eigen::Triplet<double> Tri;
        std::vector<Tri> grav;  grav.reserve(2 * ni);
        for (HelperOps::IFaces::Index i = 0; i < ni; ++i) {
            const int f  = ops.internal_faces[ i ];
            const int c1 = grid.face_cells[2*f + 0];
            const int c2 = grid.face_cells[2*f + 1];

            assert ((c1 >= 0) && (c2 >= 0));

            const double dG1 = gpot[ f2hf[2*f + 0] ];
            const double dG2 = gpot[ f2hf[2*f + 1] ];
            const double t   = trans[ f ];

            grav.push_back(Tri(i, c1,   t * dG1));
            grav.push_back(Tri(i, c2, - t * dG2));
        }

        M G(ni, nc);  G.setFromTriplets(grav.begin(), grav.end());

        return G;
    }



    V computePerfPress(const UnstructuredGrid& grid, const Wells& wells, const V& rho, const double grav)
    {
        const int nw = wells.number_of_wells;
        const int nperf = wells.well_connpos[nw];
        const int dim = grid.dimensions;
        V wdp = V::Zero(nperf,1);
        assert(wdp.size() == rho.size());

        // Main loop, iterate over all perforations,
        // using the following formula:
        //    wdp(perf) = g*(perf_z - well_ref_z)*rho(perf)
        // where the total density rho(perf) is taken to be
        //    sum_p (rho_p*saturation_p) in the perforation cell.
        // [although this is computed on the outside of this function].
        for (int w = 0; w < nw; ++w) {
            const double ref_depth = wells.depth_ref[w];
            for (int j = wells.well_connpos[w]; j < wells.well_connpos[w + 1]; ++j) {
                const int cell = wells.well_cells[j];
                const double cell_depth = grid.cell_centroids[dim * cell + dim - 1];
                wdp[j] = rho[j]*grav*(cell_depth - ref_depth);
            }
        }
        return wdp;
    }




} // Anonymous namespace




    FullyImplicitCompressibleSolver::
    FullyImplicitCompressibleSolver(const UnstructuredGrid&         grid ,
                                const BlackoilPropsAdInterface& fluid,
                                const DerivedGeology&           geo  ,
                                const RockCompressibility*      rock_comp_props,
                                const Wells&                    wells,
                                const LinearSolverInterface&    linsolver)
        : grid_  (grid)
        , fluid_ (fluid)
        , geo_   (geo)
        , rock_comp_props_(rock_comp_props)
        , wells_ (wells)
        , linsolver_ (linsolver)
        , cells_ (buildAllCells(grid.number_of_cells))
        , ops_   (grid)
        , wops_  (wells)
        , grav_  (gravityOperator(grid_, ops_, geo_))
        , rq_    (fluid.numPhases())
        , residual_ ( { std::vector<ADB>(fluid.numPhases(), ADB::null()),
                        ADB::null(),
                        ADB::null(),
                        ADB::null() } )
    {
    }





    void
    FullyImplicitCompressibleSolver::
    step(const double   dt,
         BlackoilState& x ,
         WellState&     xw)
    {
        const V pvdt = geo_.poreVolume() / dt;

        {
            const SolutionState state = constantState(x, xw);
            computeAccum(state, 0);
        }

        const double atol  = 1.0e-12;
        const double rtol  = 5.0e-8;
        const int    maxit = 15;

        assemble(pvdt, x, xw);

        const double r0  = residualNorm();
        int          it  = 0;
        std::cout << "\nIteration         Residual\n"
                  << std::setw(9) << it << std::setprecision(9)
                  << std::setw(18) << r0 << std::endl;
        bool resTooLarge = r0 > atol;
        while (resTooLarge && (it < maxit)) {
            const V dx = solveJacobianSystem();

            updateState(dx, x, xw);

            assemble(pvdt, x, xw);

            const double r = residualNorm();

            resTooLarge = (r > atol) && (r > rtol*r0);

            it += 1;
            std::cout << std::setw(9) << it << std::setprecision(9)
                      << std::setw(18) << r << std::endl;
        }

        if (resTooLarge) {
            std::cerr << "Failed to compute converged solution in " << it << " iterations. Ignoring!\n";
            // OPM_THROW(std::runtime_error, "Failed to compute converged solution in " << it << " iterations.");
        }
    }





    FullyImplicitCompressibleSolver::ReservoirResidualQuant::ReservoirResidualQuant()
        : accum(2, ADB::null())
        , mflux(   ADB::null())
        , b    (   ADB::null())
        , head (   ADB::null())
        , mob  (   ADB::null())
    {
    }





    FullyImplicitCompressibleSolver::SolutionState::SolutionState(const int np)
        : pressure  (    ADB::null())
        , saturation(np, ADB::null())
        , qs        (    ADB::null())
        , bhp       (    ADB::null())
    {
    }





    FullyImplicitCompressibleSolver::
    WellOps::WellOps(const Wells& wells)
        : w2p(wells.well_connpos[ wells.number_of_wells ],
              wells.number_of_wells)
        , p2w(wells.number_of_wells,
              wells.well_connpos[ wells.number_of_wells ])
    {
        const int        nw   = wells.number_of_wells;
        const int* const wpos = wells.well_connpos;

        typedef Eigen::Triplet<double> Tri;

        std::vector<Tri> scatter, gather;
        scatter.reserve(wpos[nw]);
        gather .reserve(wpos[nw]);

        for (int w = 0, i = 0; w < nw; ++w) {
            for (; i < wpos[ w + 1 ]; ++i) {
                scatter.push_back(Tri(i, w, 1.0));
                gather .push_back(Tri(w, i, 1.0));
            }
        }

        w2p.setFromTriplets(scatter.begin(), scatter.end());
        p2w.setFromTriplets(gather .begin(), gather .end());
    }





    FullyImplicitCompressibleSolver::SolutionState
    FullyImplicitCompressibleSolver::constantState(const BlackoilState& x,
                                               const WellState&     xw)
    {
        const int nc = grid_.number_of_cells;
        const int np = x.numPhases();

        // The block pattern assumes the following primary variables:
        //    pressure
        //    water saturation (if water present)
        //    well rates per active phase and well
        //    well bottom-hole pressure
        // Note that oil is assumed to always be present, but is never
        // a primary variable.
        std::vector<int> bpat(np, nc);
        bpat.push_back(xw.bhp().size() * np);
        bpat.push_back(xw.bhp().size());

        SolutionState state(np);

        // Pressure.
        assert (not x.pressure().empty());
        const V p = Eigen::Map<const V>(& x.pressure()[0], nc, 1);
        state.pressure = ADB::constant(p, bpat);

        // Saturation.
        assert (not x.saturation().empty());
        const DataBlock s = Eigen::Map<const DataBlock>(& x.saturation()[0], nc, np);
        V so = V::Ones(nc, 1);
        const V sw  = s.col(0);
        so -= sw;
        state.saturation[0] = ADB::constant(sw, bpat);
        state.saturation[1] = ADB::constant(so, bpat);

        // Well rates.
        assert (not xw.wellRates().empty());
        // Need to reshuffle well rates, from ordered by wells, then phase,
        // to ordered by phase, then wells.
        const int nw = wells_.number_of_wells;
        // The transpose() below switches the ordering.
        const DataBlock wrates = Eigen::Map<const DataBlock>(& xw.wellRates()[0], nw, np).transpose();
        const V qs = Eigen::Map<const V>(wrates.data(), nw*np);
        state.qs = ADB::constant(qs, bpat);

        // Well bottom-hole pressure.
        assert (not xw.bhp().empty());
        const V bhp = Eigen::Map<const V>(& xw.bhp()[0], xw.bhp().size());
        state.bhp = ADB::constant(bhp, bpat);

        return state;
    }





    FullyImplicitCompressibleSolver::SolutionState
    FullyImplicitCompressibleSolver::variableState(const BlackoilState& x,
                                               const WellState&     xw)
    {
        const int nc = grid_.number_of_cells;
        const int np = x.numPhases();

        std::vector<V> vars0;

        // Initial pressure.
        assert (not x.pressure().empty());
        const V p = Eigen::Map<const V>(& x.pressure()[0], nc, 1);
        vars0.push_back(p);

        // Initial saturation.
        assert (not x.saturation().empty());
        const DataBlock s = Eigen::Map<const DataBlock>(& x.saturation()[0], nc, np);
        const V sw = s.col(0);
        vars0.push_back(sw);


        // Initial well rates.
        assert (not xw.wellRates().empty());
        // Need to reshuffle well rates, from ordered by wells, then phase,
        // to ordered by phase, then wells.
        const int nw = wells_.number_of_wells;
        // The transpose() below switches the ordering.
        const DataBlock wrates = Eigen::Map<const DataBlock>(& xw.wellRates()[0], nw, np).transpose();
        const V qs = Eigen::Map<const V>(wrates.data(), nw*np);
        vars0.push_back(qs);

        // Initial well bottom-hole pressure.
        assert (not xw.bhp().empty());
        const V bhp = Eigen::Map<const V>(& xw.bhp()[0], xw.bhp().size());
        vars0.push_back(bhp);

        std::vector<ADB> vars = ADB::variables(vars0);

        SolutionState state(np);

        // Pressure.
        int nextvar = 0;
        state.pressure = vars[ nextvar++ ];

        // Saturation.
        const std::vector<int>& bpat = vars[0].blockPattern();
        {
            ADB so  = ADB::constant(V::Ones(nc, 1), bpat);
            ADB& sw = vars[ nextvar++ ];
            state.saturation[0] = sw;
            so = so - sw;
            state.saturation[1] = so;
         }


        // Qs.
        state.qs = vars[ nextvar++ ];

        // Bhp.
        state.bhp = vars[ nextvar++ ];

        assert(nextvar == int(vars.size()));

        return state;
    }





    void
    FullyImplicitCompressibleSolver::computeAccum(const SolutionState& state,
                                              const int            aix  )
    {

        const ADB&              press = state.pressure;
        const std::vector<ADB>& sat   = state.saturation;

        const ADB pv_mult = poroMult(press);

        for (int phase = 0; phase < 2; ++phase) {
            rq_[phase].b = fluidReciprocFVF(phase, press, cells_);
            rq_[phase].accum[aix] = pv_mult * rq_[phase].b * sat[phase];
        }

    }





    void
    FullyImplicitCompressibleSolver::
    assemble(const V&             pvdt,
             const BlackoilState& x   ,
             const WellState&     xw  )
    {
        // Create the primary variables.
        const SolutionState state = variableState(x, xw);

        // -------- Mass balance equations --------

        // Compute b_p and the accumulation term b_p*s_p for each phase,
        // except gas. For gas, we compute b_g*s_g + Rs*b_o*s_o.
        // These quantities are stored in rq_[phase].accum[1].
        // The corresponding accumulation terms from the start of
        // the timestep (b^0_p*s^0_p etc.) were already computed
        // in step() and stored in rq_[phase].accum[0].
        computeAccum(state, 1);

        // Set up the common parts of the mass balance equations
        // for each active phase.
        const V transi = subset(geo_.transmissibility(), ops_.internal_faces);
        const std::vector<ADB> kr = computeRelPerm(state);
        for (int phase = 0; phase < fluid_.numPhases(); ++phase) {
            computeMassFlux(phase, transi, kr, state);
            residual_.mass_balance[ phase ] =
                pvdt*(rq_[phase].accum[1] - rq_[phase].accum[0])
                + ops_.div*rq_[phase].mflux;

        }

        // -------- Extra (optional) sg or rs equation, and rs contributions to the mass balance equations --------

        // Add the extra (flux) terms to the gas mass balance equations
        // from gas dissolved in the oil phase.
        // The extra terms in the accumulation part of the equation are already handled.

        // -------- Well equation, and well contributions to the mass balance equations --------

        // Contribution to mass balance will have to wait.

        const int nc = grid_.number_of_cells;
        const int np = wells_.number_of_phases;
        const int nw = wells_.number_of_wells;
        const int nperf = wells_.well_connpos[nw];

        const std::vector<int> well_cells(wells_.well_cells, wells_.well_cells + nperf);
        const V transw = Eigen::Map<const V>(wells_.WI, nperf);

        const ADB& bhp = state.bhp;

        const DataBlock well_s = wops_.w2p * Eigen::Map<const DataBlock>(wells_.comp_frac, nw, np).matrix();

        // Extract variables for perforation cell pressures
        // and corresponding perforation well pressures.
        const ADB p_perfcell = subset(state.pressure, well_cells);
        // Finally construct well perforation pressures and well flows.

        // Compute well pressure differentials.
        // Construct pressure difference vector for wells.
        const int dim = grid_.dimensions;
        const double* g = geo_.gravity();
        if (g) {
            // Guard against gravity in anything but last dimension.
            for (int dd = 0; dd < dim - 1; ++dd) {
                assert(g[dd] == 0.0);
            }
        }
        ADB cell_rho_total = ADB::constant(V::Zero(nc), state.pressure.blockPattern());
        for (int phase = 0; phase < 2; ++phase) {
            const ADB cell_rho = fluidDensity(phase, state.pressure, cells_);
            cell_rho_total += state.saturation[phase] * cell_rho;
        }
        ADB inj_rho_total = ADB::constant(V::Zero(nperf), state.pressure.blockPattern());
        assert(np == wells_.number_of_phases);
        const DataBlock compi = Eigen::Map<const DataBlock>(wells_.comp_frac, nw, np);
        for (int phase = 0; phase < 2; ++phase) {
            const ADB cell_rho = fluidDensity(phase, state.pressure, cells_);
            const V fraction = compi.col(phase);
            inj_rho_total += (wops_.w2p * fraction.matrix()).array() * subset(cell_rho, well_cells);
        }
        const V rho_perf_cell = subset(cell_rho_total, well_cells).value();
        const V rho_perf_well = inj_rho_total.value();
        V prodperfs = V::Constant(nperf, -1.0);
        for (int w = 0; w < nw; ++w) {
            if (wells_.type[w] == PRODUCER) {
                std::fill(prodperfs.data() + wells_.well_connpos[w],
                          prodperfs.data() + wells_.well_connpos[w+1], 1.0);
            }
        }
        const Selector<double> producer(prodperfs);
        const V rho_perf = producer.select(rho_perf_cell, rho_perf_well);
        const V well_perf_dp = computePerfPress(grid_, wells_, rho_perf, g ? g[dim-1] : 0.0);

        const ADB p_perfwell = wops_.w2p * bhp + well_perf_dp;
        const ADB nkgradp_well = transw * (p_perfcell - p_perfwell);
        // DUMP(nkgradp_well);
        const Selector<double> cell_to_well_selector(nkgradp_well.value());
        ADB well_rates_all = ADB::constant(V::Zero(nw*np), state.bhp.blockPattern());
        ADB perf_total_mob = subset(rq_[0].mob, well_cells) + subset(rq_[1].mob, well_cells);
        std::vector<ADB> well_contribs(np, ADB::null());
        std::vector<ADB> well_perf_rates(np, ADB::null());
        for (int phase = 0; phase < np; ++phase) {
            const ADB& cell_b = rq_[phase].b;
            const ADB perf_b = subset(cell_b, well_cells);
            const ADB& cell_mob = rq_[phase].mob;
            const V well_fraction = compi.col(phase);
            // Using total mobilities for all phases for injection.
            const ADB perf_mob_injector = (wops_.w2p * well_fraction.matrix()).array() * perf_total_mob;
            const ADB perf_mob = producer.select(subset(cell_mob, well_cells),
                                                 perf_mob_injector);
            const ADB perf_flux = perf_mob * (nkgradp_well); // No gravity term for perforations.
            well_perf_rates[phase] = (perf_flux*perf_b);
            const ADB well_rates = wops_.p2w * well_perf_rates[phase];
            well_rates_all += superset(well_rates, Span(nw, 1, phase*nw), nw*np);

            // const ADB well_contrib = superset(perf_flux*perf_b, well_cells, nc);
            well_contribs[phase] = superset(perf_flux*perf_b, well_cells, nc);
            // DUMP(well_contribs[phase]);
            residual_.mass_balance[phase] += well_contribs[phase];
        }

        // Set the well flux equation
        residual_.well_flux_eq = state.qs + well_rates_all;
        // DUMP(residual_.well_flux_eq);

        // Handling BHP and SURFACE_RATE wells.
        V bhp_targets(nw);
        V rate_targets(nw);
        M rate_distr(nw, np*nw);
        for (int w = 0; w < nw; ++w) {
            const WellControls* wc = wells_.ctrls[w];
            if (wc->type[wc->current] == BHP) {
                bhp_targets[w] = wc->target[wc->current];
                rate_targets[w] = -1e100;
            } else if (wc->type[wc->current] == SURFACE_RATE) {
                bhp_targets[w] = -1e100;
                rate_targets[w] = wc->target[wc->current];
                for (int phase = 0; phase < np; ++phase) {
                    rate_distr.insert(w, phase*nw + w) = wc->distr[phase];
                }
            } else {
                OPM_THROW(std::runtime_error, "Can only handle BHP and SURFACE_RATE type controls.");
            }
        }
        const ADB bhp_residual = bhp - bhp_targets;
        const ADB rate_residual = rate_distr * state.qs - rate_targets;
        // Choose bhp residual for positive bhp targets.
        Selector<double> bhp_selector(bhp_targets);
        residual_.well_eq = bhp_selector.select(bhp_residual, rate_residual);
        // DUMP(residual_.well_eq);
    }





    V FullyImplicitCompressibleSolver::solveJacobianSystem() const
    {
        const int np = fluid_.numPhases();
        ADB mass_res = residual_.mass_balance[0];
        for (int phase = 1; phase < np; ++phase) {
            mass_res = vertcat(mass_res, residual_.mass_balance[phase]);
        }
        const ADB well_res = vertcat(residual_.well_flux_eq, residual_.well_eq);
        const ADB total_residual = collapseJacs(vertcat(mass_res, well_res));
        // DUMP(total_residual);

        const Eigen::SparseMatrix<double, Eigen::RowMajor> matr = total_residual.derivative()[0];

        V dx(V::Zero(total_residual.size()));
        Opm::LinearSolverInterface::LinearSolverReport rep
            = linsolver_.solve(matr.rows(), matr.nonZeros(),
                               matr.outerIndexPtr(), matr.innerIndexPtr(), matr.valuePtr(),
                               total_residual.value().data(), dx.data());
        if (!rep.converged) {
            OPM_THROW(std::runtime_error,
                      "FullyImplicitCompressibleSolver::solveJacobianSystem(): "
                      "Linear solver convergence failure.");
        }
        return dx;
    }





    namespace {
        struct Chop01 {
            double operator()(double x) const { return std::max(std::min(x, 1.0), 0.0); }
        };
    }





    void FullyImplicitCompressibleSolver::updateState(const V& dx,
                                                  BlackoilState& state,
                                                  WellState& well_state) const
    {
        const int np = fluid_.numPhases();
        const int nc = grid_.number_of_cells;
        const int nw = wells_.number_of_wells;
        const V null;
        assert(null.size() == 0);
        const V zero = V::Zero(nc);
        const V one = V::Constant(nc, 1.0);

        // Extract parts of dx corresponding to each part.
        const V dp = subset(dx, Span(nc));
        int varstart = nc;
        const V dsw =subset(dx, Span(nc, 1, varstart));
        varstart += dsw.size();
        const V dqs = subset(dx, Span(np*nw, 1, varstart));
        varstart += dqs.size();
        const V dbhp = subset(dx, Span(nw, 1, varstart));
        varstart += dbhp.size();
        assert(varstart == dx.size());

        // Pressure update.
        const double dpmaxrel = 0.8;
        const V p_old = Eigen::Map<const V>(&state.pressure()[0], nc, 1);
        const V absdpmax = dpmaxrel*p_old.abs();
        const V dp_limited = sign(dp) * dp.abs().min(absdpmax);
        const V p = (p_old - dp_limited).max(zero);
        std::copy(&p[0], &p[0] + nc, state.pressure().begin());

        // Saturation updates.
        const double dsmax = 0.3;
        const DataBlock s_old = Eigen::Map<const DataBlock>(& state.saturation()[0], nc, np);
        V so = one;
        const V sw_old = s_old.col(0);
        const V dsw_limited = sign(dsw) * dsw.abs().min(dsmax);
        const V sw = (sw_old - dsw_limited).unaryExpr(Chop01());
        so -= sw;
        for (int c = 0; c < nc; ++c) {
            state.saturation()[c*np] = sw[c];
            state.saturation()[c*np + 1] = so[c];
        }

        // Qs update.
        // Since we need to update the wellrates, that are ordered by wells,
        // from dqs which are ordered by phase, the simplest is to compute
        // dwr, which is the data from dqs but ordered by wells.
        const DataBlock wwr = Eigen::Map<const DataBlock>(dqs.data(), np, nw).transpose();
        const V dwr = Eigen::Map<const V>(wwr.data(), nw*np);
        const V wr_old = Eigen::Map<const V>(&well_state.wellRates()[0], nw*np);
        const V wr = wr_old - dwr;
        std::copy(&wr[0], &wr[0] + wr.size(), well_state.wellRates().begin());

        // Bhp update.
        const V bhp_old = Eigen::Map<const V>(&well_state.bhp()[0], nw, 1);
        const V bhp = bhp_old - dbhp;
        std::copy(&bhp[0], &bhp[0] + bhp.size(), well_state.bhp().begin());

    }





    std::vector<ADB>
    FullyImplicitCompressibleSolver::computeRelPerm(const SolutionState& state) const
    {
        const int               nc   = grid_.number_of_cells;
        const std::vector<int>& bpat = state.pressure.blockPattern();

        const ADB null = ADB::constant(V::Zero(nc, 1), bpat);

        const ADB sw = state.saturation[0];
        const ADB so = state.saturation[1];
        const ADB sg = null;
        return fluid_.relperm(sw, so, sg, cells_);
    }





    std::vector<ADB>
    FullyImplicitCompressibleSolver::computeRelPermWells(const SolutionState& state,
                                                     const DataBlock& well_s,
                                                     const std::vector<int>& well_cells) const
    {
        const int nw = wells_.number_of_wells;
        const int nperf = wells_.well_connpos[nw];
        const std::vector<int>& bpat = state.pressure.blockPattern();

        const ADB null = ADB::constant(V::Zero(nperf), bpat);

        const ADB sw = state.saturation[0];
        const ADB so = state.saturation[1];
        const ADB sg = null;
        return fluid_.relperm(sw, so, sg, well_cells);
    }





    void
    FullyImplicitCompressibleSolver::computeMassFlux(const int               phase ,
                                                 const V&                transi,
                                                 const std::vector<ADB>& kr    ,
                                                 const SolutionState&    state )
    {
        const ADB mu    = fluidViscosity(phase, state.pressure, cells_);
        const ADB tr_mult = transMult(state.pressure);

        rq_[ phase ].mob = tr_mult * kr[ phase ] / mu;

        const ADB rho   = fluidDensity(phase, state.pressure, cells_);

        ADB& head = rq_[ phase ].head;

        // compute gravity potensial using the face average as in eclipse and MRST
        const ADB rhoavg = ops_.caver * rho;

        const ADB dp = ops_.ngrad * state.pressure - geo_.gravity()[2] * (rhoavg * (ops_.ngrad * geo_.z().matrix()));

        head = transi*dp;
        //head      = transi*(ops_.ngrad * state.pressure) + gflux;

        UpwindSelector<double> upwind(grid_, ops_, head.value());

        const ADB& b       = rq_[ phase ].b;
        const ADB& mob     = rq_[ phase ].mob;
        rq_[ phase ].mflux = upwind.select(b * mob) * head;
    }





    double
    FullyImplicitCompressibleSolver::residualNorm() const
    {
        double r = 0;
        for (std::vector<ADB>::const_iterator
                 b = residual_.mass_balance.begin(),
                 e = residual_.mass_balance.end();
             b != e; ++b)
        {
            r = std::max(r, (*b).value().matrix().norm());
        }
        r = std::max(r, residual_.well_flux_eq.value().matrix().norm());
        r = std::max(r, residual_.well_eq.value().matrix().norm());

        return r;
    }





    ADB
    FullyImplicitCompressibleSolver::fluidViscosity(const int               phase,
                                                const ADB&              p    ,
                                                const std::vector<int>& cells) const
    {
        const ADB null = ADB::constant(V::Zero(grid_.number_of_cells, 1), p.blockPattern());
        switch (phase) {
        case Water:
            return fluid_.muWat(p, cells);
        case Oil: {
            return fluid_.muOil(p, null, cells);
        }
        default:
            OPM_THROW(std::runtime_error, "Unknown phase index " << phase);
        }
    }





    ADB
    FullyImplicitCompressibleSolver::fluidReciprocFVF(const int               phase,
                                                  const ADB&              p    ,
                                                  const std::vector<int>& cells) const
    {
        const ADB null = ADB::constant(V::Zero(grid_.number_of_cells, 1), p.blockPattern());
        switch (phase) {
        case Water:
            return fluid_.bWat(p, cells);
        case Oil: {
            return fluid_.bOil(p, null, cells);
        }
        default:
            OPM_THROW(std::runtime_error, "Unknown phase index " << phase);
        }
    }





    ADB
    FullyImplicitCompressibleSolver::fluidDensity(const int               phase,
                                              const ADB&              p    ,
                                              const std::vector<int>& cells) const
    {
        const double* rhos = fluid_.surfaceDensity();
        ADB b = fluidReciprocFVF(phase, p, cells);
        ADB rho = V::Constant(p.size(), 1, rhos[phase]) * b;
        return rho;
    }





    ADB
    FullyImplicitCompressibleSolver::poroMult(const ADB& p) const
    {
        const int n = p.size();
        if (rock_comp_props_ && rock_comp_props_->isActive()) {
            V pm(n);
            V dpm(n);
            for (int i = 0; i < n; ++i) {
                pm[i] = rock_comp_props_->poroMult(p.value()[i]);
                dpm[i] = rock_comp_props_->poroMultDeriv(p.value()[i]);
            }
            ADB::M dpm_diag = spdiag(dpm);
            const int num_blocks = p.numBlocks();
            std::vector<ADB::M> jacs(num_blocks);
            for (int block = 0; block < num_blocks; ++block) {
                jacs[block] = dpm_diag * p.derivative()[block];
            }
            return ADB::function(pm, jacs);
        } else {
            return ADB::constant(V::Constant(n, 1.0), p.blockPattern());
        }
    }





    ADB
    FullyImplicitCompressibleSolver::transMult(const ADB& p) const
    {
        const int n = p.size();
        if (rock_comp_props_ && rock_comp_props_->isActive()) {
            V tm(n);
            V dtm(n);
            for (int i = 0; i < n; ++i) {
                tm[i] = rock_comp_props_->transMult(p.value()[i]);
                dtm[i] = rock_comp_props_->transMultDeriv(p.value()[i]);
            }
            ADB::M dtm_diag = spdiag(dtm);
            const int num_blocks = p.numBlocks();
            std::vector<ADB::M> jacs(num_blocks);
            for (int block = 0; block < num_blocks; ++block) {
                jacs[block] = dtm_diag * p.derivative()[block];
            }
            return ADB::function(tm, jacs);
        } else {
            return ADB::constant(V::Constant(n, 1.0), p.blockPattern());
        }
    }


} // namespace Opm
