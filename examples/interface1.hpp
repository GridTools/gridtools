/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "benchmarker.hpp"
#include "horizontal_diffusion_repository.hpp"
#include <tools/verifier.hpp>

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

// Temporary disable the expressions, as they are intrusive. The operators +,- are overloaded
//  for any type, which breaks most of the code after using expressions
#ifdef CXX11_ENABLED
using namespace expressions;
#endif

namespace horizontal_diffusion {
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_lap;
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_flx;
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_out;

    typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct lap_function {
        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -1, 1, -1, 1 > > in;

        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_lap) {
            eval(out()) = (gridtools::float_type)4 * eval(in()) -
                          (eval(in(1, 0, 0)) + eval(in(0, 1, 0)) + eval(in(-1, 0, 0)) + eval(in(0, -1, 0)));
        }
    };

    struct flx_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< 0, 1, 0, 0 > > in;
        typedef accessor< 2, enumtype::in, extent< 0, 1, 0, 0 > > lap;

        typedef boost::mpl::vector< out, in, lap > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_flx) {
            eval(out()) = eval(lap(1, 0, 0)) - eval(lap(0, 0, 0));
            if (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0, 0))) > 0) {
                eval(out()) = 0.;
            }
        }
    };

    struct fly_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< 0, 0, 0, 1 > > in;
        typedef accessor< 2, enumtype::in, extent< 0, 0, 0, 1 > > lap;

        typedef boost::mpl::vector< out, in, lap > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_flx) {
            eval(out()) = eval(lap(0, 1, 0)) - eval(lap(0, 0, 0));
            if (eval(out()) * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0) {
                eval(out()) = 0.;
            }
        }
    };

    struct out_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in > in;
        typedef accessor< 2, enumtype::in, extent< -1, 0, 0, 0 > > flx;
        typedef accessor< 3, enumtype::in, extent< 0, 0, -1, 0 > > fly;
        typedef accessor< 4, enumtype::in > coeff;

        typedef boost::mpl::vector< out, in, flx, fly, coeff > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_out) {
#if defined(CXX11_ENABLED) && !defined(CUDA_EXAMPLE)
            eval(out()) = eval(in()) - eval(coeff()) * (eval(flx() - flx(-1, 0, 0) + fly() - fly(0, -1, 0)));
#else
            eval(out()) =
                eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0, 0)) + eval(fly()) - eval(fly(0, -1, 0)));
#endif
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

#ifdef CUDA_EXAMPLE
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef horizontal_diffusion::repository::storage_type storage_type;
        typedef horizontal_diffusion::repository::tmp_storage_type tmp_storage_type;

        horizontal_diffusion::repository repository(d1, d2, d3, halo_size);
        repository.init_fields();

        repository.generate_reference();

        // Definition of the actual data fields that are used for input/output
        storage_type &in = repository.in();
        storage_type &out = repository.out();
        storage_type &coeff = repository.coeff();

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef arg< 0, tmp_storage_type > p_lap;
        typedef arg< 1, tmp_storage_type > p_flx;
        typedef arg< 2, tmp_storage_type > p_fly;
        typedef arg< 3, storage_type > p_coeff;
        typedef arg< 4, storage_type > p_in;
        typedef arg< 5, storage_type > p_out;

        // An array of placeholders to be passed to the domain
        // I'm using mpl::vector, but the final API should look slightly simpler
        typedef boost::mpl::vector< p_lap, p_flx, p_fly, p_coeff, p_in, p_out > accessor_list;

// construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are
// used, temporary and not
// It must be noted that the only fields to be passed to the constructor are the non-temporary.
// The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I
// don't particularly like this)
#if defined(CXX11_ENABLED)
        gridtools::aggregator_type< accessor_list > domain((p_out() = out), (p_in() = in), (p_coeff() = coeff));
#else
        gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&coeff, &in, &out));
#endif
        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        uint_t dj[5] = {halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

/*
  Here we do lot of stuff
  1) We pass to the intermediate representation ::run function the description
  of the stencil, which is a multi-stage stencil (mss)
  The mss includes (in order of execution) a laplacian, two fluxes which are independent
  and a final step that is the out_function
  2) The logical physical domain with the fields to use
  3) The actual grid dimensions
 */
#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::stencil *
#else
        boost::shared_ptr< gridtools::stencil >
#endif
#endif
            horizontal_diffusion = gridtools::make_computation< gridtools::BACKEND >(
                domain,
                grid,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    define_caches(cache< IJ, local >(p_lap(), p_flx(), p_fly())),
                    gridtools::make_stage< lap_function >(p_lap(), p_in()), // esf_descriptor
                    gridtools::make_independent                             // independent_esf
                    (gridtools::make_stage< flx_function >(p_flx(), p_in(), p_lap()),
                        gridtools::make_stage< fly_function >(p_fly(), p_in(), p_lap())),
                    gridtools::make_stage< out_function >(p_out(), p_in(), p_flx(), p_fly(), p_coeff())));

        horizontal_diffusion->ready();
        horizontal_diffusion->steady();
        horizontal_diffusion->run();

#ifdef __CUDACC__
        repository.update_cpu();
#endif

        bool result = true;

        if (verify) {
#ifdef CXX11_ENABLED
#if FLOAT_PRECISION == 4
            verifier verif(1e-6);
#else
            verifier verif(1e-12);
#endif
            array< array< uint_t, 2 >, 3 > halos{
                {{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
            result = verif.verify(grid, repository.out_ref(), repository.out(), halos);
#else
#if FLOAT_PRECISION == 4
            verifier verif(1e-6, halo_size);
#else
            verifier verif(1e-12, halo_size);
#endif
            result = verif.verify(grid, repository.out_ref(), repository.out());
#endif
        }

#ifdef BENCHMARK
        benchmarker::run(horizontal_diffusion, t_steps);
#endif
        horizontal_diffusion->finalize();

        return result;
    }

} // namespace horizontal_diffusion
