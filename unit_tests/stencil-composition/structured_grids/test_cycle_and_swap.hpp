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
// disabling pedantic mode because I want to use a 2D layout map
//(to test the case in which the 3rd dimension is not k)
#define PEDANTIC_DISABLED

#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>

using namespace gridtools;
using namespace expressions;

namespace test_cycle_and_swap {
    using namespace gridtools;
    using namespace enumtype;

    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;

    struct functor {
        typedef inout_accessor< 0, extent<>, 3 > p_i;
        typedef boost::mpl::vector< p_i > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(p_i()) += eval(p_i());
        }
    };

    struct functor_avg {
        typedef inout_accessor< 0, extent<>, 4 > p_data;
        typedef dimension< 4 > time;

        // TODO Check if this previous code with static (not defined in device) could work with CUDA8
        // static x i;
        // static y j;

        // typedef decltype(i) i_t;
        // typedef decltype(j) j_t;

        typedef boost::mpl::vector< p_data > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            x i;
            y j;

            typedef decltype(i) i_t;
            typedef decltype(j) j_t;

            eval(p_data(time(1))) = (eval(p_data(i - 1)) + eval(p_data(i + 1))) * (float_t)0.5;
        }
    };

// TODO Check if this previous code with static (not defined in device) could work with CUDA8
//    functor_avg::i_t functor_avg::i;
//    functor_avg::j_t functor_avg::j;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    bool test_2D() {

        typedef gridtools::layout_map< 0, 1 > layout_t;
        typedef gridtools::BACKEND::storage_info< 0, layout_t > meta_t;
        typedef gridtools::BACKEND::storage_type< uint_t, meta_t >::type storage_type;
        typedef typename field< storage_type, 2 >::type field_t;

        meta_t meta_(1u, 1u);
        field_t i_data(meta_, 0, "in");
        i_data.get_value< 0, 0 >(0, 0) = 0;
        i_data.get_value< 1, 0 >(0, 0) = 1;

        uint_t di[5] = {0, 0, 0, 0, 1};
        uint_t dj[5] = {0, 0, 0, 0, 1};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = 0;

        typedef arg< 0, field_t > p_i_data;
        typedef boost::mpl::vector< p_i_data > accessor_list;

        aggregator_type< accessor_list > domain(boost::fusion::make_vector(&i_data));

        auto comp = gridtools::make_computation< gridtools::BACKEND >(
            domain, grid, gridtools::make_multistage(execute< forward >(), gridtools::make_stage< functor >(p_i_data())));

        comp->ready();
        comp->steady();
        comp->run();
#ifdef __CUDACC__
        i_data.d2h_update();
#endif
        swap< 0, 0 >::with< 1, 0 >::apply(i_data);
#ifdef __CUDACC__
        i_data.h2d_update();
#endif
        comp->run();
        comp->finalize();

        return (i_data(0, 0) == 2 && i_data.get_value< 1, 0 >(0, 0) == 0);
    }
    bool test_3D() {

        const uint_t d1 = 13;
        const uint_t d2 = 9;
        const uint_t d3 = 3;

        typedef gridtools::layout_map< 0, 1, 2 > layout_t;
        typedef gridtools::BACKEND::storage_info< 0, layout_t > meta_t;
        typedef gridtools::BACKEND::storage_type< uint_t, meta_t >::type storage_type;
        typedef typename field< storage_type, 2 >::type field_t;

        meta_t meta_(d1, d2, d3);
        field_t i_data(meta_, 0, "in");
        field_t reference(meta_, 0, "reference");

        i_data.get_value< 0, 0 >(0, 0, 0) = 0.;
        i_data.get_value< 1, 0 >(0, 0, 0) = 1.;

        const uint_t halo_size = 1;
        uint_t di[5] = {halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
        uint_t dj[5] = {halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        typedef arg< 0, field_t > p_i_data;
        typedef boost::mpl::vector< p_i_data > accessor_list;

        aggregator_type< accessor_list > domain(boost::fusion::make_vector(&i_data));

        auto comp = gridtools::make_computation< gridtools::BACKEND >(
            domain, grid, gridtools::make_multistage(execute< forward >(), gridtools::make_stage< functor_avg >(p_i_data())));

        // fill the input (snapshot 0) with some initial data
        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    i_data.get_value< 0, 0 >(i, j, k) = i + j * 10 + k * 100;
                }
            }
        }

        // compute a reference field using the same initial data. It is computed using a two time level field.
        // output reference will be stored in reference<1,0>
        for (uint_t k = 0; k < d3; ++k) {
            for (uint_t i = halo_size; i < d1 - halo_size; ++i) {
                for (uint_t j = halo_size; j < d2 - halo_size; ++j) {
                    reference.get_value< 0, 0 >(i, j, k) =
                        (i_data.get_value< 0, 0 >(i + 1, j, k) + i_data.get_value< 0, 0 >(i - 1, j, k)) * (float_t)0.5;
                }
            }
            for (uint_t i = halo_size + 1; i < d1 - halo_size - 1; ++i) {
                for (uint_t j = halo_size + 1; j < d2 - halo_size - 1; ++j) {
                    reference.get_value< 1, 0 >(i, j, k) =
                        (reference.get_value< 0, 0 >(i + 1, j, k) + reference.get_value< 0, 0 >(i - 1, j, k)) *
                        (float_t)0.5;
                }
            }
        }

        comp->ready();
        comp->steady();
        comp->run();
#ifdef __CUDACC__
        i_data.d2h_update();
#endif
        swap< 0, 0 >::with< 1, 0 >::apply(i_data);
#ifdef __CUDACC__
        i_data.h2d_update();
#endif

        // note that the second run will do wrong computations at the first line of the 2D domain of the coordinates,
        // because the first line of
        // grid points at the boundaries has not been computed at the first run. However we just dont validate these
        // points with the verifier
        comp->run();
        comp->finalize();

#if FLOAT_PRECISION == 4
        verifier verif(1e-6);
#else
        verifier verif(1e-12);
#endif
        array< array< uint_t, 2 >, 3 > halos{
            {{halo_size + 1, halo_size + 1}, {halo_size + 1, halo_size + 1}, {halo_size + 1, halo_size + 1}}};
        return verif.verify(grid, reference, i_data, halos);
    }

} // namespace test_cycle_and_swap
