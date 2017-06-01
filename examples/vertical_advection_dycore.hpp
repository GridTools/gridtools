/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include <gridtools.hpp>

#include "benchmarker.hpp"
#include "defs.hpp"
#include "vertical_advection_repository.hpp"
#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>

/*
  This file shows an implementation of the "vertical advection" stencil used in COSMO for U field
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace vertical_advection_dycore {
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, 1 >, level< 1, -2 > > kbody;
    typedef gridtools::interval< level< 0, -1 >, level< 1, -2 > > kbody_low;
    typedef gridtools::interval< level< 0, -1 >, level< 0, -1 > > kminimum;
    typedef gridtools::interval< level< 1, -1 >, level< 1, -1 > > kmaximum;

    typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;

    template < typename T >
    struct u_forward_function {
        typedef accessor< 0 > utens_stage;
        typedef accessor< 1, enumtype::in, extent< 0, 1, 0, 0 > > wcon;
        typedef accessor< 2 > u_stage;
        typedef accessor< 3 > u_pos;
        typedef accessor< 4 > utens;
        typedef accessor< 5 > dtr_stage;
        typedef accessor< 6, enumtype::inout > acol;
        typedef accessor< 7, enumtype::inout > bcol;
        typedef accessor< 8, enumtype::inout, extent< 0, 0, 0, 0, 0, -1 > > ccol;
        typedef accessor< 9, enumtype::inout > dcol;

        typedef boost::mpl::vector< utens_stage, wcon, u_stage, u_pos, utens, dtr_stage, acol, bcol, ccol, dcol >
            arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, kbody interval) {
            // TODO use Average function here
            T gav = (T)-0.25 * (eval(wcon(1, 0, 0)) + eval(wcon(0, 0, 0)));
            T gcv = (T)0.25 * (eval(wcon(1, 0, 1)) + eval(wcon(0, 0, 1)));

            T as = gav * BET_M;
            T cs = gcv * BET_M;

            eval(acol()) = gav * BET_P;
            eval(ccol()) = gcv * BET_P;
            eval(bcol()) = eval(dtr_stage()) - eval(acol()) - eval(ccol());

            T correctionTerm =
                -as * (eval(u_stage(0, 0, -1)) - eval(u_stage())) - cs * (eval(u_stage(0, 0, 1)) - eval(u_stage()));
            // update the d column
            computeDColumn(eval, correctionTerm);
            thomas_forward(eval, interval);
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, kmaximum interval) {
            T gav = -(T)0.25 * (eval(wcon(1, 0, 0)) + eval(wcon()));
            T as = gav * BET_M;

            eval(acol()) = gav * BET_P;
            eval(bcol()) = eval(dtr_stage()) - eval(acol());

            T correctionTerm = -as * (eval(u_stage(0, 0, -1)) - eval(u_stage()));

            // update the d column
            computeDColumn(eval, correctionTerm);
            thomas_forward(eval, interval);
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, kminimum interval) {
            T gcv = (T)0.25 * (eval(wcon(1, 0, 1)) + eval(wcon(0, 0, 1)));
            T cs = gcv * BET_M;

            eval(ccol()) = gcv * BET_P;
            eval(bcol()) = eval(dtr_stage()) - eval(ccol());

            T correctionTerm = -cs * (eval(u_stage(0, 0, 1)) - eval(u_stage()));
            // update the d column
            computeDColumn(eval, correctionTerm);
            thomas_forward(eval, interval);
        }

      private:
        template < typename Evaluation >
        GT_FUNCTION static void computeDColumn(Evaluation &eval, const T correctionTerm) {
            eval(dcol()) = eval(dtr_stage()) * eval(u_pos()) + eval(utens()) + eval(utens_stage()) + correctionTerm;
        }

        template < typename Evaluation >
        GT_FUNCTION static void thomas_forward(Evaluation &eval, kbody) {
            T divided = (T)1.0 / (eval(bcol()) - (eval(ccol(0, 0, -1)) * eval(acol())));
            eval(ccol()) = eval(ccol()) * divided;
            eval(dcol()) = (eval(dcol()) - (eval(dcol(0, 0, -1)) * eval(acol()))) * divided;
        }

        template < typename Evaluation >
        GT_FUNCTION static void thomas_forward(Evaluation &eval, kmaximum) {
            T divided = (T)1.0 / (eval(bcol()) - eval(ccol(0, 0, -1)) * eval(acol()));
            eval(dcol()) = (eval(dcol()) - eval(dcol(0, 0, -1)) * eval(acol())) * divided;
        }

        template < typename Evaluation >
        GT_FUNCTION static void thomas_forward(Evaluation &eval, kminimum) {
            T divided = (T)1.0 / eval(bcol());
            eval(ccol()) = eval(ccol()) * divided;
            eval(dcol()) = eval(dcol()) * divided;
        }
    };

    template < typename T >
    struct u_backward_function {
        typedef accessor< 0, enumtype::inout > utens_stage;
        typedef accessor< 1 > u_pos;
        typedef accessor< 2 > dtr_stage;
        typedef accessor< 3 > ccol;
        typedef accessor< 4 > dcol;
        typedef accessor< 5, enumtype::inout > data_col;

        typedef boost::mpl::vector< utens_stage, u_pos, dtr_stage, ccol, dcol, data_col > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, kbody_low interval) {
            eval(utens_stage()) = eval(dtr_stage()) * (thomas_backward(eval, interval) - eval(u_pos()));
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, kmaximum interval) {
            eval(utens_stage()) = eval(dtr_stage()) * (thomas_backward(eval, interval) - eval(u_pos()));
        }

      private:
        template < typename Evaluation >
        GT_FUNCTION static T thomas_backward(Evaluation &eval, kbody_low) {
            T datacol = eval(dcol()) - eval(ccol()) * eval(data_col(0, 0, 1));
            eval(data_col()) = datacol;
            return datacol;
        }

        template < typename Evaluation >
        GT_FUNCTION static T thomas_backward(Evaluation &eval, kmaximum) {
            T datacol = eval(dcol());
            eval(data_col()) = datacol;
            return datacol;
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    // std::ostream& operator<<(std::ostream& s, u_forward_function<double> const) {
    //    return s << "u_forward_function";
    //}
    std::ostream &operator<<(std::ostream &s, u_backward_function< double > const) {
        return s << "u_backward_function";
    }

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t_steps, bool verify) {

        const int halo_size = 3;

        typedef vertical_advection::repository::storage_type storage_type;
        typedef vertical_advection::repository::scalar_storage_type scalar_storage_type;

        vertical_advection::repository repository(d1, d2, d3, halo_size);
        repository.init_fields();

        repository.generate_reference();

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef arg< 0, storage_type > p_utens_stage;
        typedef arg< 1, storage_type > p_u_stage;
        typedef arg< 2, storage_type > p_wcon;
        typedef arg< 3, storage_type > p_u_pos;
        typedef arg< 4, storage_type > p_utens;
        typedef arg< 5, scalar_storage_type > p_dtr_stage;
        typedef tmp_arg< 6, storage_type > p_acol;
        typedef tmp_arg< 7, storage_type > p_bcol;
        typedef tmp_arg< 8, storage_type > p_ccol;
        typedef tmp_arg< 9, storage_type > p_dcol;
        typedef tmp_arg< 10, storage_type > p_data_col;

        // An array of placeholders to be passed to the domain
        // I'm using mpl::vector, but the final API should look slightly simpler
        typedef boost::mpl::vector< p_utens_stage,
            p_u_stage,
            p_wcon,
            p_u_pos,
            p_utens,
            p_dtr_stage,
            p_acol,
            p_bcol,
            p_ccol,
            p_dcol,
            p_data_col > accessor_list;

        gridtools::aggregator_type< accessor_list > domain(repository.utens_stage(),
            repository.u_stage(),
            repository.wcon(),
            repository.u_pos(),
            repository.utens(),
            repository.dtr_stage());

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        uint_t dj[5] = {halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::stencil *
#else
        boost::shared_ptr< gridtools::stencil >
#endif
#endif
            vertical_advection = gridtools::make_computation< vertical_advection::va_backend >(
                domain,
                grid,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    define_caches(cache< K, flush, kbody >(p_ccol())),
                    gridtools::make_stage< u_forward_function< double > >(p_utens_stage(),
                        p_wcon(),
                        p_u_stage(),
                        p_u_pos(),
                        p_utens(),
                        p_dtr_stage(),
                        p_acol(),
                        p_bcol(),
                        p_ccol(),
                        p_dcol()) // esf_descriptor
                    ),
                gridtools::make_multistage(
                    execute< backward >(),
                    gridtools::make_stage< u_backward_function< double > >(
                        p_utens_stage(), p_u_pos(), p_dtr_stage(), p_ccol(), p_dcol(), p_data_col())));

        vertical_advection->ready();

        vertical_advection->steady();

        vertical_advection->run();

        repository.utens_stage().sync();

        bool result = true;
        if (verify) {
#ifdef CXX11_ENABLED
#if FLOAT_PRECISION == 4
            verifier verif(1e-6);
#else
            verifier verif(1e-12);
#endif
            array< array< uint_t, 2 >, 3 > halos{{{halo_size, halo_size}, {halo_size, halo_size}, {0, 0}}};
            result = verif.verify(grid, repository.utens_stage_ref(), repository.utens_stage(), halos);
#else
#if FLOAT_PRECISION == 4
            verifier verif(1e-6, halo_size);
#else
            verifier verif(1e-12, halo_size);
#endif
            result = verif.verify(grid, repository.utens_stage_ref(), repository.utens_stage());
#endif
        }
#ifdef BENCHMARK
        benchmarker::run(vertical_advection, t_steps);
#endif
        vertical_advection->finalize();

        return result;
    }

} // namespace vertical_advection
