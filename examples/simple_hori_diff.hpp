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

#include <stencil-composition/stencil-composition.hpp>
#include "horizontal_diffusion_repository.hpp"
#include <tools/verifier.hpp>
#include "benchmarker.hpp"
#include "backend_select.hpp"

/**
  @file
  This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

using namespace expressions;

namespace shorizontal_diffusion {
    // These are the stencil operators that compose the multistage stencil in this test
    struct wlap_function {
        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -1, 1, -1, 1 > > in;
        typedef accessor< 2, enumtype::in > crlato;
        typedef accessor< 3, enumtype::in > crlatu;

        typedef boost::mpl::vector< out, in, crlato, crlatu > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in(1, 0, 0)) + eval(in(-1, 0, 0)) - (gridtools::float_type)2 * eval(in()) +
                          eval(crlato()) * (eval(in(0, 1, 0)) - eval(in())) +
                          eval(crlatu()) * (eval(in(0, -1, 0)) - eval(in()));
        }
    };

    struct divflux_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in > in;
        typedef accessor< 2, enumtype::in, extent< -1, 1, -1, 1 > > lap;
        typedef accessor< 3, enumtype::in > crlato;
        typedef accessor< 4, enumtype::in > coeff;

        typedef boost::mpl::vector< out, in, lap, crlato, coeff > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            gridtools::float_type fluxx = eval(lap(1, 0, 0)) - eval(lap());
            gridtools::float_type fluxx_m = eval(lap(0, 0, 0)) - eval(lap(-1, 0, 0));

            gridtools::float_type fluxy = eval(crlato()) * (eval(lap(0, 1, 0)) - eval(lap()));
            gridtools::float_type fluxy_m = eval(crlato()) * (eval(lap(0, 0, 0)) - eval(lap(0, -1, 0)));

            eval(out()) = eval(in()) + ((fluxx_m - fluxx) + (fluxy_m - fluxy)) * eval(coeff());
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, wlap_function const) { return s << "wlap_function"; }
    std::ostream &operator<<(std::ostream &s, divflux_function const) { return s << "flx_function"; }

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;
        uint_t halo_size = 2;

        typedef horizontal_diffusion::repository::storage_type storage_type;
        typedef horizontal_diffusion::repository::j_storage_type j_storage_type;

        horizontal_diffusion::repository repository(d1, d2, d3, halo_size);
        repository.init_fields();

        repository.generate_reference_simple();

        // Definition of the actual data fields that are used for input/output
        storage_type &in = repository.in();
        storage_type &out = repository.out();
        storage_type &coeff = repository.coeff();
        j_storage_type &crlato = repository.crlato();
        j_storage_type &crlatu = repository.crlatu();

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef tmp_arg< 0, storage_type > p_lap;
        typedef arg< 1, storage_type > p_coeff;
        typedef arg< 2, storage_type > p_in;
        typedef arg< 3, storage_type > p_out;
        typedef arg< 4, j_storage_type > p_crlato;
        typedef arg< 5, j_storage_type > p_crlatu;

        // An array of placeholders to be passed to the domain
        typedef boost::mpl::vector< p_lap, p_coeff, p_in, p_out, p_crlato, p_crlatu > accessor_list;
        gridtools::aggregator_type< accessor_list > domain(coeff, in, out, crlato, crlatu);

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        auto grid = make_grid(di, dj, d3);

        auto simple_hori_diff = gridtools::make_computation< gridtools::backend_t >(
            domain,
            grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                define_caches(cache< IJ, cache_io_policy::local >(p_lap())),
                gridtools::make_stage< wlap_function >(p_lap(), p_in(), p_crlato(), p_crlatu()), // esf_descriptor
                gridtools::make_stage< divflux_function >(p_out(), p_in(), p_lap(), p_crlato(), p_coeff())));

        simple_hori_diff->ready();
        simple_hori_diff->steady();

        simple_hori_diff->run();

        out.sync();

        bool result = true;
        if (verify) {
#if FLOAT_PRECISION == 4
            verifier verif(1e-6);
#else
            verifier verif(1e-12);
#endif
            array< array< uint_t, 2 >, 3 > halos{{{halo_size, halo_size}, {halo_size, halo_size}, {0, 0}}};
            result = verif.verify(grid, repository.out_ref(), repository.out(), halos);
        }
#ifdef BENCHMARK
        benchmarker::run(simple_hori_diff, t_steps);
#endif

        simple_hori_diff->finalize();

        return result; /// lapse_time.wall<5000000 &&
    }

} // namespace simple_hori_diff
