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

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace domain_reassign {
#ifdef __CUDACC__
    typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
    //                   strides  1 x xy
    //                      dims  x y z
    typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#endif

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct test_functor {

        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    std::ostream &operator<<(std::ostream &s, test_functor const) { return s << "test_functor"; }

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test() {

        uint_t d1 = 32;
        uint_t d2 = 32;
        uint_t d3 = 32;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
        typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< float_type, storage_info_t >
            storage_t;

        storage_info_t meta_data_(d1, d2, d3);

        // Definition of the actual data fields that are used for input/output
        storage_t in(meta_data_, 0.);
        storage_t out(meta_data_, float_type(-1.));
        storage_t in2(meta_data_, 0.);
        storage_t out2(meta_data_, float_type(-1.));
        auto inv = make_host_view(in);
        auto outv = make_host_view(out);
        auto in2v = make_host_view(in2);
        auto out2v = make_host_view(out2);
        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    inv(i, j, k) = i + j + k;
                    in2v(i, j, k) = i + j + k + 3;
                }

        typedef arg< 0, storage_t > p_in;
        typedef arg< 1, storage_t > p_out;

        typedef boost::mpl::vector< p_in, p_out > accessor_list;

        gridtools::aggregator_type< accessor_list > domain(in, out);

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        auto copy = gridtools::make_computation< gridtools::BACKEND >(domain,
            grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(), gridtools::make_stage< test_functor >(p_in(), p_out())));

        copy->ready();
        copy->steady();
        copy->run();
        copy->finalize();

        bool success = true;
        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    if (inv(i, j, k) != outv(i, j, k)) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "in = " << inv(i, j, k) << ", out = " << outv(i, j, k) << std::endl;
                        success = false;
                    }
                }
        copy->reassign(in2, out2);
        copy->ready();
        copy->steady();
        copy->run();
        copy->finalize();

        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    if (in2v(i, j, k) != out2v(i, j, k)) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "in = " << in2v(i, j, k) << ", out = " << out2v(i, j, k) << std::endl;
                        success = false;
                    }
                }

        return success;
    }
} // namespace
