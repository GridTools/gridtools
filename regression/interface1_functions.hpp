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

//#include <gridtools.hpp>
#include "backend_select.hpp"
#include "cache_flusher.hpp"
#include "defs.hpp"
#include "horizontal_diffusion_repository.hpp"
#include <gridtools/stencil-composition/caches/define_caches.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/stencil-functions/stencil-functions.hpp>
#include <gridtools/tools/verifier.hpp>

/**
  @file
  This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO
 */

using gridtools::accessor;
using gridtools::arg;
using gridtools::extent;
using gridtools::level;

using namespace gridtools;
using namespace enumtype;

namespace horizontal_diffusion_functions {
    // These are the stencil operators that compose the multistage stencil in this test
    struct lap_function {
        typedef accessor<0, enumtype::inout> out;
        typedef accessor<1, enumtype::in, extent<-1, 1, -1, 1>> in;

        typedef boost::mpl::vector<out, in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation eval) {
            auto x = (gridtools::float_type)4.0 * eval(in()) -
                     (eval(in(-1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0)) + eval(in(1, 0, 0)));
            eval(out()) = x;
        }
    };

    struct flx_function {

        typedef accessor<0, enumtype::inout> out;
        typedef accessor<1, enumtype::in, extent<-1, 2, -1, 1>> in;
        //    typedef const accessor<2, range<0, 1, 0, 0> > lap;

        typedef boost::mpl::vector<out, in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation eval) {
#ifdef FUNCTIONS_MONOLITHIC
            gridtools::float_type _x_ =
                (gridtools::float_type)4.0 * eval(in()) -
                (eval(in(-1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0)) + eval(in(1, 0, 0)));
            gridtools::float_type _y_ =
                (gridtools::float_type)4.0 * eval(in(1, 0, 0)) -
                (eval(in(0, 0, 0)) + eval(in(1, -1, 0)) + eval(in(1, 1, 0)) + eval(in(2, 0, 0)));
#else
#ifdef FUNCTIONS_PROCEDURES
            gridtools::float_type _x_;
            gridtools::call_proc<lap_function>::at<0, 0, 0>::with(eval, _x_, in());
            gridtools::float_type _y_;
            gridtools::call_proc<lap_function>::at<1, 0, 0>::with(eval, _y_, in());
#else
#ifdef FUNCTIONS_PROCEDURES_OFFSETS
            gridtools::float_type _x_;
            gridtools::call_proc<lap_function>::with(eval, _x_, in());
            gridtools::float_type _y_;
            gridtools::call_proc<lap_function>::with(eval, _y_, in(1, 0, 0));
#else
#ifdef FUNCTIONS_OFFSETS
            gridtools::float_type _x_ = gridtools::call<lap_function>::with(eval, in(0, 0, 0));
            gridtools::float_type _y_ = gridtools::call<lap_function>::with(eval, in(1, 0, 0));
#else
            gridtools::float_type _x_ = gridtools::call<lap_function>::at<0, 0, 0>::with(eval, in());
            gridtools::float_type _y_ = gridtools::call<lap_function>::at<1, 0, 0>::with(eval, in());
#endif
#endif
#endif
#endif
            eval(out()) = _y_ - _x_;
            eval(out()) = (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0, 0))) > 0.0) ? 0.0 : eval(out());
        }
    };

    struct fly_function {

        typedef accessor<0, enumtype::inout> out;
        typedef accessor<1, enumtype::in, extent<-1, 1, -1, 2>> in;
        //    typedef const accessor<2, range<0, 0, 0, 1> > lap;

        typedef boost::mpl::vector<out, in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation eval) {

#ifdef FUNCTIONS_MONOLITHIC
            gridtools::float_type _x_ =
                (gridtools::float_type)4.0 * eval(in()) -
                (eval(in(-1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0)) + eval(in(1, 0, 0)));
            gridtools::float_type _y_ =
                (gridtools::float_type)4.0 * eval(in(0, 1, 0)) -
                (eval(in(-1, 1, 0)) + eval(in(0, 0, 0)) + eval(in(0, 2, 0)) + eval(in(1, 1, 0)));
#else
#ifdef FUNCTIONS_PROCEDURES
            gridtools::float_type _x_;
            gridtools::call_proc<lap_function>::at<0, 0, 0>::with(eval, _x_, in());
            gridtools::float_type _y_;
            gridtools::call_proc<lap_function>::at<0, 1, 0>::with(eval, _y_, in());
#else
#ifdef FUNCTIONS_PROCEDURES_OFFSETS
            gridtools::float_type _x_;
            gridtools::call_proc<lap_function>::with(eval, _x_, in());
            gridtools::float_type _y_;
            gridtools::call_proc<lap_function>::with(eval, _y_, in(0, 1, 0));
#else
#ifdef FUNCTIONS_OFFSETS
            gridtools::float_type _x_ = gridtools::call<lap_function>::with(eval, in(0, 0, 0));
            gridtools::float_type _y_ = gridtools::call<lap_function>::with(eval, in(0, 1, 0));
#else
            gridtools::float_type _x_ = gridtools::call<lap_function>::at<0, 0, 0>::with(eval, in());
            gridtools::float_type _y_ = gridtools::call<lap_function>::at<0, 1, 0>::with(eval, in());
#endif
#endif
#endif
#endif
            eval(out()) = _y_ - _x_;
            eval(out()) = (eval(out()) * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0.0) ? 0.0 : eval(out());
        }
    };

    struct out_function {

        typedef accessor<0, enumtype::inout> out;
        typedef accessor<1, enumtype::in> in;
        typedef accessor<2, enumtype::in, extent<-1, 0, 0, 0>> flx;
        typedef accessor<3, enumtype::in, extent<0, 0, -1, 0>> fly;
        typedef accessor<4, enumtype::in> coeff;

        typedef boost::mpl::vector<out, in, flx, fly, coeff> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation eval) {
            eval(out()) =
                eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0, 0)) + eval(fly()) - eval(fly(0, -1, 0)));
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, lap_function const) { return s << "lap_function"; }
    std::ostream &operator<<(std::ostream &s, flx_function const) { return s << "flx_function"; }
    std::ostream &operator<<(std::ostream &s, fly_function const) { return s << "fly_function"; }
    std::ostream &operator<<(std::ostream &s, out_function const) { return s << "out_function"; }

    void handle_error(int) { std::cout << "error" << std::endl; }

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;
        uint_t halo_size = 2;

        typedef horizontal_diffusion::repository::storage_type storage_type;

        horizontal_diffusion::repository repository(d1, d2, d3, halo_size);
        repository.init_fields();

        repository.generate_reference();

        // Definition of the actual data fields that are used for input/output
        storage_type &in = repository.in();
        storage_type &out = repository.out();
        storage_type &coeff = repository.coeff();

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef tmp_arg<0, storage_type> p_flx;
        typedef tmp_arg<1, storage_type> p_fly;
        typedef arg<2, storage_type> p_coeff;
        typedef arg<3, storage_type> p_in;
        typedef arg<4, storage_type> p_out;

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grids(2,d1-2,2,d2-2);
        halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        auto grid_ = make_grid(di, dj, d3);

        auto horizontal_diffusion = gridtools::make_computation<backend_t>(grid_,
            p_coeff{} = coeff,
            p_in{} = in,
            p_out{} = out,
            gridtools::make_multistage // mss_descriptor
            (execute<forward>(),
                define_caches(cache<IJ, cache_io_policy::local>(p_flx(), p_fly())),
                // gridtools::make_stage<lap_function>(p_lap(), p_in()), // esf_descriptor
                gridtools::make_independent // independent_esf
                (gridtools::make_stage<flx_function>(p_flx(), p_in()),
                    gridtools::make_stage<fly_function>(p_fly(), p_in())),
                gridtools::make_stage<out_function>(p_out(), p_in(), p_flx(), p_fly(), p_coeff())));

        cache_flusher flusher(cache_flusher_size);

        horizontal_diffusion.run();

        repository.out().sync();

        bool result = true;
        if (verify) {
#if FLOAT_PRECISION == 4
            verifier verif(1e-6);
#else
            verifier verif(1e-12);
#endif

            array<array<uint_t, 2>, 3> halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
            bool result = verif.verify(grid_, repository.out_ref(), repository.out(), halos);
        }
        if (!result) {
            std::cout << "ERROR" << std::endl;
        }

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            flusher.flush();
            horizontal_diffusion.run();
        }
        std::cout << horizontal_diffusion.print_meter() << std::endl;
#endif

        return result;
    }

} // namespace horizontal_diffusion_functions
