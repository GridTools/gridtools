/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

//#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "horizontal_diffusion_repository.hpp"
#include <stencil-composition/caches/define_caches.hpp>
#include <tools/verifier.hpp>
#include <stencil-composition/structured_grids/call_interfaces.hpp>
#include "./cache_flusher.hpp"
#include "./defs.hpp"

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
// using namespace expressions;
#endif

namespace horizontal_diffusion_functions {
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_lap;
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_flx;
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_out;

    typedef gridtools::interval< level< 0, -2 >, level< 1, 3 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct lap_function {
        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -1, 1, -1, 1 > > in;

        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluator >
        GT_FUNCTION static void Do(Evaluator const &eval, x_lap) {
            auto x = (gridtools::float_type)4.0 * eval(in()) -
                     (eval(in(-1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0)) + eval(in(1, 0, 0)));
            eval(out()) = x;
        }
    };

    struct flx_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -1, 2, -1, 1 > > in;
        //    typedef const accessor<2, range<0, 1, 0, 0> > lap;

        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluator >
        GT_FUNCTION static void Do(Evaluator const &eval, x_flx) {
#ifdef FUNCTIONS_MONOLITHIC
#pragma message "monolithic version"
            gridtools::float_type _x_ =
                (gridtools::float_type)4.0 * eval(in()) -
                (eval(in(-1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0)) + eval(in(1, 0, 0)));
            gridtools::float_type _y_ =
                (gridtools::float_type)4.0 * eval(in(1, 0, 0)) -
                (eval(in(0, 0, 0)) + eval(in(1, -1, 0)) + eval(in(1, 1, 0)) + eval(in(2, 0, 0)));
#else
#ifdef FUNCTIONS_PROCEDURES
            gridtools::float_type _x_;
            gridtools::call_proc< lap_function, x_flx >::at< 0, 0, 0 >::with(eval, _x_, in());
            gridtools::float_type _y_;
            gridtools::call_proc< lap_function, x_flx >::at< 1, 0, 0 >::with(eval, _y_, in());
#else
#ifdef FUNCTIONS_PROCEDURES_OFFSETS
            gridtools::float_type _x_;
            gridtools::call_proc< lap_function, x_flx >::with_offsets(eval, _x_, in());
            gridtools::float_type _y_;
            gridtools::call_proc< lap_function, x_flx >::with_offsets(eval, _y_, in(1, 0, 0));
#else
#ifdef FUNCTIONS_OFFSETS
            gridtools::float_type _x_ = gridtools::call< lap_function, x_flx >::with_offsets(eval, in(0, 0, 0));
            gridtools::float_type _y_ = gridtools::call< lap_function, x_flx >::with_offsets(eval, in(1, 0, 0));
#else
            gridtools::float_type _x_ = gridtools::call< lap_function, x_flx >::at< 0, 0, 0 >::with(eval, in());
            gridtools::float_type _y_ = gridtools::call< lap_function, x_flx >::at< 1, 0, 0 >::with(eval, in());
#endif
#endif
#endif
#endif
            eval(out()) = _y_ - _x_;
            eval(out()) = (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0, 0))) > 0.0) ? 0.0 : eval(out());
        }
    };

    struct fly_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in, extent< -1, 1, -1, 2 > > in;
        //    typedef const accessor<2, range<0, 0, 0, 1> > lap;

        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluator >
        GT_FUNCTION static void Do(Evaluator const &eval, x_flx) {

#ifdef FUNCTIONS_MONOLITHIC
#pragma message "monolithic version"
            gridtools::float_type _x_ =
                (gridtools::float_type)4.0 * eval(in()) -
                (eval(in(-1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0)) + eval(in(1, 0, 0)));
            gridtools::float_type _y_ =
                (gridtools::float_type)4.0 * eval(in(0, 1, 0)) -
                (eval(in(-1, 1, 0)) + eval(in(0, 0, 0)) + eval(in(0, 2, 0)) + eval(in(1, 1, 0)));
#else
#ifdef FUNCTIONS_PROCEDURES
            gridtools::float_type _x_;
            gridtools::call_proc< lap_function, x_flx >::at< 0, 0, 0 >::with(eval, _x_, in());
            gridtools::float_type _y_;
            gridtools::call_proc< lap_function, x_flx >::at< 0, 1, 0 >::with(eval, _y_, in());
#else
#ifdef FUNCTIONS_PROCEDURES_OFFSETS
            gridtools::float_type _x_;
            gridtools::call_proc< lap_function, x_flx >::with_offsets(eval, _x_, in());
            gridtools::float_type _y_;
            gridtools::call_proc< lap_function, x_flx >::with_offsets(eval, _y_, in(0, 1, 0));
#else
#ifdef FUNCTIONS_OFFSETS
            gridtools::float_type _x_ = gridtools::call< lap_function, x_flx >::with_offsets(eval, in(0, 0, 0));
            gridtools::float_type _y_ = gridtools::call< lap_function, x_flx >::with_offsets(eval, in(0, 1, 0));
#else
            gridtools::float_type _x_ = gridtools::call< lap_function, x_flx >::at< 0, 0, 0 >::with(eval, in());
            gridtools::float_type _y_ = gridtools::call< lap_function, x_flx >::at< 0, 1, 0 >::with(eval, in());
#endif
#endif
#endif
#endif
            eval(out()) = _y_ - _x_;
            eval(out()) = (eval(out()) * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0.0) ? 0.0 : eval(out());
        }
    };

    struct out_function {

        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1, enumtype::in > in;
        typedef accessor< 2, enumtype::in, extent< -1, 0, 0, 0 > > flx;
        typedef accessor< 3, enumtype::in, extent< 0, 0, -1, 0 > > fly;
        typedef accessor< 4, enumtype::in > coeff;

        typedef boost::mpl::vector< out, in, flx, fly, coeff > arg_list;

        template < typename Evaluator >
        GT_FUNCTION static void Do(Evaluator const &eval, x_out) {
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
        typedef arg< 0, tmp_storage_type > p_flx;
        typedef arg< 1, tmp_storage_type > p_fly;
        typedef arg< 2, storage_type > p_coeff;
        typedef arg< 3, storage_type > p_in;
        typedef arg< 4, storage_type > p_out;

        // An array of placeholders to be passed to the domain
        // I'm using mpl::vector, but the final API should look slightly simpler
        typedef boost::mpl::vector< p_flx, p_fly, p_coeff, p_in, p_out > accessor_list;

// construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are
// used, temporary and not
// It must be noted that the only fields to be passed to the constructor are the non-temporary.
// The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I
// don't particularly like this)
#if defined(CXX11_ENABLED) && !defined(CUDA_EXAMPLE)
        gridtools::aggregator_type< accessor_list > domain_((p_out() = out), (p_in() = in), (p_coeff() = coeff));
#else
        gridtools::aggregator_type< accessor_list > domain_(boost::fusion::make_vector(&coeff, &in, &out));
#endif
        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grids(2,d1-2,2,d2-2);
        uint_t di[5] = {halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        uint_t dj[5] = {halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        gridtools::grid< axis > grid_(di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

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
                domain_,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    define_caches(cache< IJ, local >(p_flx(), p_fly())),
                    // gridtools::make_stage<lap_function>(p_lap(), p_in()), // esf_descriptor
                    gridtools::make_independent // independent_esf
                    (gridtools::make_stage< flx_function >(p_flx(), p_in()),
                        gridtools::make_stage< fly_function >(p_fly(), p_in())),
                    gridtools::make_stage< out_function >(p_out(), p_in(), p_flx(), p_fly(), p_coeff())));

        horizontal_diffusion->ready();

        horizontal_diffusion->steady();

        cache_flusher flusher(cache_flusher_size);

        horizontal_diffusion->run();

#ifdef __CUDACC__
        repository.update_cpu();
#endif

        bool result = true;
        if (verify) {
#if FLOAT_PRECISION == 4
            verifier verif(1e-6);
#else
            verifier verif(1e-12);
#endif

#ifdef CXX11_ENABLED
            array< array< uint_t, 2 >, 3 > halos{
                {{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
            bool result = verif.verify(grid_, repository.out_ref(), repository.out(), halos);
#else
            result = verif.verify(repository.out_ref(), repository.out());
#endif
        }
        if (!result) {
            std::cout << "ERROR" << std::endl;
        }

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            flusher.flush();
            horizontal_diffusion->run();
        }
#endif

        horizontal_diffusion->finalize();
#ifdef BENCHMARK
        std::cout << horizontal_diffusion->print_meter() << std::endl;
#endif

        return result; /// lapse_time.wall<5000000 &&
    }

} // namespace horizontal_diffusion_functions
