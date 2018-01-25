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
    struct out_function {
        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -2, 2, -2, 2 > > in;
        typedef accessor< 2, enumtype::in > coeff;

        typedef boost::mpl::vector< out, in, coeff > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            auto in_ijm2 = eval(in(0, -2, 0));

            auto in_imjm = eval(in(-1, -1, 0));
            auto in_ijm = eval(in(0, -1, 0));
            auto in_ipjm = eval(in(1, -1, 0));

            auto in_im2j = eval(in(-2, 0, 0));
            auto in_imj = eval(in(-1, 0, 0));
            auto in_ij = eval(in(0, 0, 0));
            auto in_ipj = eval(in(1, 0, 0));
            auto in_ip2j = eval(in(2, 0, 0));

            auto in_imjp = eval(in(-1, 1, 0));
            auto in_ijp = eval(in(0, 1, 0));
            auto in_ipjp = eval(in(1, 1, 0));

            auto in_ijp2 = eval(in(0, 2, 0));

            auto lap_ij = 4 * in_ij - (in_ipj + in_ijp + in_imj + in_ijm);
            auto lap_imj = 4 * in_imj - (in_ij + in_imjp + in_im2j + in_imjm);
            auto lap_ipj = 4 * in_ipj - (in_ip2j + in_ipjp + in_ij + in_ipjm);
            auto lap_ijm = 4 * in_ijm - (in_ipjm + in_ij + in_imjm + in_ijm2);
            auto lap_ijp = 4 * in_ijp - (in_ipjp + in_ijp2 + in_imjp + in_ij);

            auto flx_ij = lap_ipj - lap_ij;
            flx_ij = flx_ij * (in_ipj - in_ij) > 0 ? 0 : flx_ij;

            auto flx_imj = lap_ij - lap_imj;
            flx_imj = flx_imj * (in_ij - in_imj) > 0 ? 0 : flx_imj;

            auto fly_ij = lap_ijp - lap_ij;
            fly_ij = fly_ij * (in_ijp - in_ij) > 0 ? 0 : fly_ij;

            auto fly_ijm = lap_ij - lap_ijm;
            fly_ijm = fly_ijm * (in_ij - in_ijm) > 0 ? 0 : fly_ijm;

            eval(out()) = in_ij - eval(coeff()) * (flx_ij - flx_imj + fly_ij - fly_ijm);
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
