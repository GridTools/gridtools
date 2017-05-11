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
#include "gtest/gtest.h"
#include <boost/shared_ptr.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/reductions/reductions.hpp>
#include "cache_flusher.hpp"
#include "defs.hpp"
#include "tools/verifier.hpp"

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

namespace reduction {

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct sum_red {

        typedef accessor< 0, enumtype::in > in;
        typedef boost::mpl::vector< in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static float_type Do(Evaluation &eval, x_interval) {
            return eval(in());
        }
    };

    // These are the stencil operators that compose the multistage stencil in this test
    struct desf {

        typedef accessor< 0, enumtype::in > in;
        typedef accessor< 1, enumtype::inout > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        cache_flusher flusher(cache_flusher_size);

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef BACKEND::storage_traits_t::storage_info_t< __COUNTER__, 3 > meta_data_t;
        typedef BACKEND::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

        meta_data_t meta_data_(x, y, z);

        // Definition of the actual data fields that are used for input/output
        storage_t in(meta_data_, "in");
        storage_t out(meta_data_, "out");

        auto inv = make_host_view(in);

        float_type sum_ref = 0, prod_ref = 1;
        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    inv(i, j, k) = static_cast< float_type >((std::rand() % 100 + std::rand() % 100) * 0.002 + 0.51);
                    sum_ref += inv(i, j, k);
                    prod_ref *= inv(i, j, k);
                }

        typedef arg< 0, storage_t > p_in;
        typedef arg< 1, storage_t > p_out;

        typedef boost::mpl::vector< p_in, p_out > accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::aggregator_type< accessor_list > domain(in, out);

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        auto sum_red_ = make_computation< gridtools::BACKEND >(domain,
            grid,
            make_multistage(execute< forward >(), make_stage< desf >(p_in(), p_out())),
            make_reduction< sum_red, binop::sum >((float_type)(0.0), p_out()));

        sum_red_->ready();
        sum_red_->steady();

        float_type sum_redt = sum_red_->run();
        float_type precision;
#if FLOAT_PRECISION == 4
        precision = 1e-6;
#else
        precision = 1e-12;
#endif
        bool success = compare_below_threshold(sum_ref, sum_redt, precision);
#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            flusher.flush();
            sum_red_->run();
        }
        sum_red_->finalize();
        std::cout << "Sum Reduction : " << sum_red_->print_meter() << std::endl;
#endif

#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::computation< float_type > *
#else
        boost::shared_ptr< gridtools::computation< float_type > >
#endif
#endif
            prod_red_ = make_computation< gridtools::BACKEND >(domain,
                grid,
                make_multistage(execute< forward >(), make_stage< desf >(p_in(), p_out())),
                make_reduction< sum_red, binop::prod >((float_type)(1.0), p_out()));

        prod_red_->ready();
        prod_red_->steady();

        float_type prod_redt = prod_red_->run();

        success = success & compare_below_threshold(prod_ref, prod_redt, precision);
#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            flusher.flush();
            prod_red_->run();
        }
        prod_red_->finalize();
        std::cout << "Prod Reduction : " << prod_red_->print_meter() << std::endl;
#endif

        return success;
    }
} // namespace red
