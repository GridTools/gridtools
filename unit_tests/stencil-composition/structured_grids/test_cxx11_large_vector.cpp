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
#define GT_VECTOR_LIMIT_SIZE 30
#include <stencil-composition/stencil-composition.hpp>

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;

// These are the stencil operators that compose the multistage stencil in this test
struct copy_functor_large {

    typedef accessor< 0, enumtype::inout, extent<>, 3 > out;
    typedef accessor< 1, enumtype::in > in0;
    typedef accessor< 2, enumtype::in > in1;
    typedef accessor< 3, enumtype::in > in2;
    typedef accessor< 4, enumtype::in > in3;
    typedef accessor< 5, enumtype::in > in4;
    typedef accessor< 6, enumtype::in > in5;
    typedef accessor< 7, enumtype::in > in6;
    typedef accessor< 8, enumtype::in > in7;
    typedef accessor< 9, enumtype::in > in8;
    typedef accessor< 10, enumtype::in > in9;
    typedef accessor< 11, enumtype::in > in10;
    typedef accessor< 12, enumtype::in > in11;
    typedef accessor< 13, enumtype::in > in12;
    typedef accessor< 14, enumtype::in > in13;
    typedef accessor< 15, enumtype::in > in14;
    typedef accessor< 16, enumtype::in > in15;
    typedef accessor< 17, enumtype::in > in16;
    typedef accessor< 18, enumtype::in > in17;
    typedef accessor< 19, enumtype::in > in18;
    typedef accessor< 20, enumtype::in > in19;
    typedef accessor< 21, enumtype::in > in20;
    typedef accessor< 22, enumtype::in > in21;
    typedef accessor< 23, enumtype::in > in22;

    typedef boost::mpl::vector< out,
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        in8,
        in9,
        in10,
        in11,
        in12,
        in13,
        in14,
        in15,
        in16,
        in17,
        in18,
        in19,
        in20,
        in21,
        in22 > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = eval(in3());
    }
};

TEST(large_vector, copy) {

    uint_t d1 = 4;
    uint_t d2 = 4;
    uint_t d3 = 4;

#ifdef __CUDACC__
#define BACKEND_ARCH Cuda
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND_ARCH Host
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    typedef storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 > storage_info_t;
    typedef storage_traits< BACKEND_ARCH >::data_store_t< float_type, storage_info_t > data_store_t;

    storage_info_t meta_data_(d1, d2, d3);

    // Definition of the actual data fields that are used for input/output
    data_store_t in(meta_data_, [](int i, int j, int k) { return i + j + k; }, "in");
    data_store_t out(meta_data_, -1.0, "out");

    typedef arg< 0, data_store_t > p_out;
    typedef arg< 1, data_store_t > p_in0;
    typedef arg< 2, data_store_t > p_in1;
    typedef arg< 3, data_store_t > p_in2;
    typedef arg< 4, data_store_t > p_in3;
    typedef arg< 5, data_store_t > p_in4;
    typedef arg< 6, data_store_t > p_in5;
    typedef arg< 7, data_store_t > p_in6;
    typedef arg< 8, data_store_t > p_in7;
    typedef arg< 9, data_store_t > p_in8;
    typedef arg< 10, data_store_t > p_in9;
    typedef arg< 11, data_store_t > p_in10;
    typedef arg< 12, data_store_t > p_in11;
    typedef arg< 13, data_store_t > p_in12;
    typedef arg< 14, data_store_t > p_in13;
    typedef arg< 15, data_store_t > p_in14;
    typedef arg< 16, data_store_t > p_in15;
    typedef arg< 17, data_store_t > p_in16;
    typedef arg< 18, data_store_t > p_in17;
    typedef arg< 19, data_store_t > p_in18;
    typedef arg< 20, data_store_t > p_in19;
    typedef arg< 21, data_store_t > p_in20;
    typedef arg< 22, data_store_t > p_in21;
    typedef arg< 23, data_store_t > p_in22;

    typedef boost::mpl::vector< p_out,
        p_in0,
        p_in1,
        p_in2,
        p_in3,
        p_in4,
        p_in5,
        p_in6,
        p_in7,
        p_in8,
        p_in9,
        p_in10,
        p_in11,
        p_in12,
        p_in13,
        p_in14,
        p_in15,
        p_in16,
        p_in17,
        p_in18,
        p_in19,
        p_in20,
        p_in21,
        p_in22 > accessor_list;
    gridtools::aggregator_type< accessor_list > domain((p_out() = out),
        (p_in0() = in),
        (p_in1() = in),
        (p_in2() = in),
        (p_in3() = in),
        (p_in4() = in),
        (p_in5() = in),
        (p_in6() = in),
        (p_in7() = in),
        (p_in8() = in),
        (p_in9() = in),
        (p_in10() = in),
        (p_in11() = in),
        (p_in12() = in),
        (p_in13() = in),
        (p_in14() = in),
        (p_in15() = in),
        (p_in16() = in),
        (p_in17() = in),
        (p_in18() = in),
        (p_in19() = in),
        (p_in20() = in),
        (p_in21() = in),
        (p_in22() = in));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
    uint_t di[5] = {0, 0, 0, d1 - 1, d1};
    uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    auto copy =
        gridtools::make_computation< gridtools::BACKEND >(domain,
            grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                                                              gridtools::make_stage< copy_functor_large >(p_out(),
                                                                  p_in0(),
                                                                  p_in1(),
                                                                  p_in2(),
                                                                  p_in3(),
                                                                  p_in4(),
                                                                  p_in5(),
                                                                  p_in6(),
                                                                  p_in7(),
                                                                  p_in8(),
                                                                  p_in9(),
                                                                  p_in10(),
                                                                  p_in11(),
                                                                  p_in12(),
                                                                  p_in13(),
                                                                  p_in14(),
                                                                  p_in15(),
                                                                  p_in16(),
                                                                  p_in17(),
                                                                  p_in18(),
                                                                  p_in19(),
                                                                  p_in20(),
                                                                  p_in21(),
                                                                  p_in22())));

    copy->ready();

    copy->steady();

    copy->run();

    out.sync();
    in.sync();

    auto in_v = make_host_view(in);
    auto out_v = make_host_view(out);
    // check consistency
    assert(check_consistency(in, in_v) && "view cannot be used safely.");
    assert(check_consistency(out, out_v) && "view cannot be used safely.");

    bool success = true;
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if ((in_v(i, j, k) != i + j + k) && (out_v(i, j, k) != i + j + k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "in = " << in_v(i, j, k) << ", out = " << out_v(i, j, k) << std::endl;
                    success = false;
                }
            }
        }
    }
    copy->finalize();

    ASSERT_TRUE(success);
}
