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

#ifdef USE_SERIALBOX

#include "gtest/gtest.h"
#include "stencil-composition/stencil-composition.hpp"

using namespace gridtools;
using namespace enumtype;

typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

/** Copy stencil */
struct test_functor {

    typedef accessor< 0, enumtype::in, extent<>, 3 > in;
    typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        eval(out()) = eval(in());
    }
};

TEST(StencilSerialization, Test) {
    uint_t d1 = 32;
    uint_t d2 = 32;
    uint_t d3 = 32;

    typedef layout_map< 0, 1, 2 > layout_t;
    typedef backend< Host, structured, Naive >::storage_info< 0, layout_t > meta_data_t;
    typedef backend< Host, structured, Naive >::storage_type< float_type, meta_data_t >::type storage_t;

    // Storages
    meta_data_t meta_data(d1, d2, d3);
    storage_t in(meta_data, "in");
    storage_t out(meta_data, "out");

    for (uint_t i = 0; i < d1; ++i)
        for (uint_t j = 0; j < d2; ++j)
            for (uint_t k = 0; k < d3; ++k)
                in(i, j, k) = i + j + k;

    // Domain
    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out));

    // Grid
    uint_t di[5] = {0, 0, 0, d1 - 1, d1};
    uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    // Computation
    auto copy = make_computation< backend< Host, structured, Naive > >(
        domain, grid, 
        make_multistage(execute< forward >(), make_stage< test_functor >(p_in(), p_out())));

    copy->ready();
    copy->steady();
    copy->run();

#ifdef __CUDACC__
    out.d2h_update();
    in.d2h_update();
#endif

    copy->finalize();

    bool success = true;
    for (uint_t i = 0; i < d1; ++i)
        for (uint_t j = 0; j < d2; ++j)
            for (uint_t k = 0; k < d3; ++k) {
                if (in(i, j, k) != out(i, j, k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "in = " << in(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                    success = false;
                }
            }
    ASSERT_TRUE(success);
}

#endif
