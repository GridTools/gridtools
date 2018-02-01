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

#include "backend_select.hpp"
#include "benchmarker.hpp"
#include "horizontal_diffusion_repository.hpp"
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/stencil-functions/stencil-functions.hpp>
#include <tools/verifier.hpp>

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

using namespace expressions;

namespace horizontal_diffusion {
    struct lap_function {
        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -1, 1, -1, 1 > > in;

        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation eval) {
            eval(out()) = (gridtools::float_type)4 * eval(in()) -
                          (eval(in(1, 0, 0)) + eval(in(0, 1, 0)) + eval(in(-1, 0, 0)) + eval(in(0, -1, 0)));
        }
    };

    struct flx_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -1, 2, -1, 1 > > in;

        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation eval) {
            auto lap_hi = call< lap_function >::with(eval, in(1, 0, 0));
            auto lap_lo = call< lap_function >::with(eval, in(0, 0, 0));
            auto flx = lap_hi - lap_lo;

            eval(out()) = flx * (eval(in(1, 0, 0)) - eval(in(0, 0, 0))) > 0 ? 0 : flx;
        }
    };

    struct fly_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -1, 1, -1, 2 > > in;

        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation eval) {
            auto lap_hi = call< lap_function >::with(eval, in(0, 1, 0));
            auto lap_lo = call< lap_function >::with(eval, in(0, 0, 0));
            auto fly = lap_hi - lap_lo;

            eval(out()) = fly * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0 ? 0 : fly;
        }
    };

    struct out_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -2, 2, -2, 2 > > in;
        typedef accessor< 2, enumtype::in > coeff;

        typedef boost::mpl::vector< out, in, coeff > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            auto flx_hi = call< flx_function >::with(eval, in(0, 0, 0));
            auto flx_lo = call< flx_function >::with(eval, in(-1, 0, 0));

            auto fly_hi = call< fly_function >::with(eval, in(0, 0, 0));
            auto fly_lo = call< fly_function >::with(eval, in(0, -1, 0));

            eval(out()) = eval(in()) - eval(coeff()) * (flx_hi - flx_lo + fly_hi - fly_lo);
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;
        uint_t halo_size = 2;

        typedef horizontal_diffusion::repository::storage_type storage_type;

        horizontal_diffusion::repository repository(d1, d2, d3, halo_size);
        repository.init_fields();

        repository.generate_reference();

        storage_type &in = repository.in();
        storage_type &out = repository.out();
        storage_type &coeff = repository.coeff();

        typedef arg< 0, storage_type > p_coeff;
        typedef arg< 1, storage_type > p_in;
        typedef arg< 2, storage_type > p_out;

        typedef boost::mpl::vector< p_coeff, p_in, p_out > accessor_list;

        gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out), (p_coeff() = coeff));

        halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        auto grid = make_grid(di, dj, d3);

        auto horizontal_diffusion = gridtools::make_computation< backend_t >(
            domain,
            grid,
            gridtools::make_multistage(
                execute< parallel >(), gridtools::make_stage< out_function >(p_out(), p_in(), p_coeff())));

        horizontal_diffusion->ready();
        horizontal_diffusion->steady();
        horizontal_diffusion->run();

        repository.out().sync();

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
        benchmarker::run(horizontal_diffusion, t_steps);
#endif
        horizontal_diffusion->finalize();

        return result;
    }

} // namespace horizontal_diffusion
