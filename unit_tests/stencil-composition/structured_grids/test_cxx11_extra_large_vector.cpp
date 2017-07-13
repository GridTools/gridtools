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
#define GT_VECTOR_LIMIT_SIZE 120
#include <stencil-composition/stencil-composition.hpp>

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;

struct copy_functor_extra_large {

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
    typedef accessor< 24, enumtype::in > in23;
    typedef accessor< 25, enumtype::in > in24;
    typedef accessor< 26, enumtype::in > in25;
    typedef accessor< 27, enumtype::in > in26;
    typedef accessor< 28, enumtype::in > in27;
    typedef accessor< 29, enumtype::in > in28;
    typedef accessor< 30, enumtype::in > in29;
    typedef accessor< 31, enumtype::in > in30;
    typedef accessor< 32, enumtype::in > in31;
    typedef accessor< 33, enumtype::in > in32;
    typedef accessor< 34, enumtype::in > in33;
    typedef accessor< 35, enumtype::in > in34;
    typedef accessor< 36, enumtype::in > in35;
    typedef accessor< 37, enumtype::in > in36;
    typedef accessor< 38, enumtype::in > in37;
    typedef accessor< 39, enumtype::in > in38;
    typedef accessor< 40, enumtype::in > in39;
    typedef accessor< 41, enumtype::in > in40;
    typedef accessor< 42, enumtype::in > in41;
    typedef accessor< 43, enumtype::in > in42;
    typedef accessor< 44, enumtype::in > in43;
    typedef accessor< 45, enumtype::in > in44;
    typedef accessor< 46, enumtype::in > in45;
    typedef accessor< 47, enumtype::in > in46;
    typedef accessor< 48, enumtype::in > in47;
    typedef accessor< 49, enumtype::in > in48;
    typedef accessor< 50, enumtype::in > in49;
    typedef accessor< 51, enumtype::in > in50;
    typedef accessor< 52, enumtype::in > in51;

    typedef boost::mpl::vector53< out,
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
        in22,
        in23,
        in24,
        in25,
        in26,
        in27,
        in28,
        in29,
        in30,
        in31,
        in32,
        in33,
        in34,
        in35,
        in36,
        in37,
        in38,
        in39,
        in40,
        in41,
        in42,
        in43,
        in44,
        in45,
        in46,
        in47,
        in48,
        in49,
        in50,
        in51 > arg_list;

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
    typedef arg< 24, data_store_t > p_in23;
    typedef arg< 25, data_store_t > p_in24;
    typedef arg< 26, data_store_t > p_in25;
    typedef arg< 27, data_store_t > p_in26;
    typedef arg< 28, data_store_t > p_in27;
    typedef arg< 29, data_store_t > p_in28;
    typedef arg< 30, data_store_t > p_in29;
    typedef arg< 31, data_store_t > p_in30;
    typedef arg< 32, data_store_t > p_in31;
    typedef arg< 33, data_store_t > p_in32;
    typedef arg< 34, data_store_t > p_in33;
    typedef arg< 35, data_store_t > p_in34;
    typedef arg< 36, data_store_t > p_in35;
    typedef arg< 37, data_store_t > p_in36;
    typedef arg< 38, data_store_t > p_in37;
    typedef arg< 39, data_store_t > p_in38;
    typedef arg< 40, data_store_t > p_in39;
    typedef arg< 41, data_store_t > p_in40;
    typedef arg< 42, data_store_t > p_in41;
    typedef arg< 43, data_store_t > p_in42;
    typedef arg< 44, data_store_t > p_in43;
    typedef arg< 45, data_store_t > p_in44;
    typedef arg< 46, data_store_t > p_in45;
    typedef arg< 47, data_store_t > p_in46;
    typedef arg< 48, data_store_t > p_in47;
    typedef arg< 49, data_store_t > p_in48;
    typedef arg< 50, data_store_t > p_in49;
    typedef arg< 51, data_store_t > p_in50;
    typedef arg< 52, data_store_t > p_in51;

    typedef boost::mpl::vector53< p_out,
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
        p_in22,
        p_in23,
        p_in24,
        p_in25,
        p_in26,
        p_in27,
        p_in28,
        p_in29,
        p_in30,
        p_in31,
        p_in32,
        p_in33,
        p_in34,
        p_in35,
        p_in36,
        p_in37,
        p_in38,
        p_in39,
        p_in40,
        p_in41,
        p_in42,
        p_in43,
        p_in44,
        p_in45,
        p_in46,
        p_in47,
        p_in48,
        p_in49,
        p_in50,
        p_in51 > accessor_list;
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
        (p_in22() = in),
        (p_in23() = in),
        (p_in24() = in),
        (p_in25() = in),
        (p_in26() = in),
        (p_in27() = in),
        (p_in28() = in),
        (p_in29() = in),
        (p_in30() = in),
        (p_in31() = in),
        (p_in32() = in),
        (p_in33() = in),
        (p_in34() = in),
        (p_in35() = in),
        (p_in36() = in),
        (p_in37() = in),
        (p_in38() = in),
        (p_in39() = in),
        (p_in40() = in),
        (p_in41() = in),
        (p_in42() = in),
        (p_in43() = in),
        (p_in44() = in),
        (p_in45() = in),
        (p_in46() = in),
        (p_in47() = in),
        (p_in48() = in),
        (p_in49() = in),
        (p_in50() = in),
        (p_in51() = in));

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
    uint_t di[5] = {0, 0, 0, d1 - 1, d1};
    uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    auto copy = gridtools::make_computation< gridtools::BACKEND >(domain,
        grid,
        gridtools::make_multistage // mss_descriptor
        (execute< forward >(),
                                                                      gridtools::make_stage< copy_functor >(p_out(),
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
                                                                          p_in22(),
                                                                          p_in23(),
                                                                          p_in24(),
                                                                          p_in25(),
                                                                          p_in26(),
                                                                          p_in27(),
                                                                          p_in28(),
                                                                          p_in29(),
                                                                          p_in30(),
                                                                          p_in31(),
                                                                          p_in32(),
                                                                          p_in33(),
                                                                          p_in34(),
                                                                          p_in35(),
                                                                          p_in36(),
                                                                          p_in37(),
                                                                          p_in38(),
                                                                          p_in39(),
                                                                          p_in40(),
                                                                          p_in41(),
                                                                          p_in42(),
                                                                          p_in43(),
                                                                          p_in44(),
                                                                          p_in45(),
                                                                          p_in46(),
                                                                          p_in47(),
                                                                          p_in48(),
                                                                          p_in49(),
                                                                          p_in50(),
                                                                          p_in51())));

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
