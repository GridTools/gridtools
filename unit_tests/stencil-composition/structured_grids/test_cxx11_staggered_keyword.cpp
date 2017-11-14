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
#include "stencil-composition/stencil-composition.hpp"

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#elif defined(__AVX512F__)
#define BACKEND backend< Mic, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

namespace test_staggered_keyword {
    using namespace gridtools;
    using namespace enumtype;

    struct functor {
        static uint_t ok_i;
        static uint_t ok_j;

        typedef accessor< 0, gridtools::enumtype::inout > p_i;
        typedef accessor< 1 > p_j;
        typedef boost::mpl::vector< p_i, p_j > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            // std::cout<<"i: "<< eval(p_i(-5,-5,0)) <<", j: "<<eval(p_j(-5,-5,0))<< std::endl;
            if (eval(p_i(-5, -5, 0)) == 5)
                ok_i++;
            if (eval(p_j(-5, -5, 0)) == 5)
                ok_j++;
        }
    };
    uint_t functor::ok_i = 0;
    uint_t functor::ok_j = 0;

    bool test() {
        typedef BACKEND::storage_traits_t::storage_info_t< 0, 3 > meta_data_t;
        typedef BACKEND::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

        meta_data_t meta_((uint_t)30, (uint_t)20, (uint_t)1);
        storage_t i_data(meta_, [](int i, int j, int k) { return i; });
        storage_t j_data(meta_, [](int i, int j, int k) { return j; });

        halo_descriptor di{0, 0, 5, 30 - 1, 30};
        halo_descriptor dj{0, 0, 5, 20 - 1, 20};

        auto grid = make_grid(di, dj, (uint_t)1);

        typedef arg< 0, storage_t > p_i_data;
        typedef arg< 1, storage_t > p_j_data;
        typedef boost::mpl::vector< p_i_data, p_j_data > accessor_list;

        aggregator_type< accessor_list > domain(i_data, j_data);
        auto comp = gridtools::make_computation< gridtools::BACKEND >(
            domain,
            grid,
            gridtools::make_multistage(execute< forward >(),
                gridtools::make_stage< functor, staggered< 5, 5, 5, 5 > >(p_i_data(), p_j_data())));

        comp->ready();
        comp->steady();
        comp->run();

        return (functor::ok_i && functor::ok_j);
    }
} // namespace test_staggered_keyword

TEST(stencil, test_staggered_keyword) { EXPECT_TRUE(test_staggered_keyword::test()); }
