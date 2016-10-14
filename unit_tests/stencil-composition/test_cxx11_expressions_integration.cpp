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

/**
 * Integration test for testing expressions inside a computation.
 * The test setup is not very nice but it was designed that way to minimize compilation time, i.e. to test everything
 * within one make_computation call.
 */

#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

namespace {
    const double DEFAULT_VALUE = -999.;

    typedef interval< level< 0, -2 >, level< 1, 1 > > axis;
    typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;
}

class test_expressions : public testing::Test {
  protected:
#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    const uint_t d1 = 100;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 0;

    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    typedef gridtools::BACKEND::storage_info< 0, layout_t > meta_t;
    typedef gridtools::BACKEND::storage_type< double, meta_t >::type storage_type;

    meta_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid<::axis > grid;

    verifier verifier_;
    array< array< uint_t, 2 >, 3 > verifier_halos;

    storage_type val2;
    storage_type val3;
    storage_type out;
    storage_type reference;

    typedef arg< 0, storage_type > p_val2;
    typedef arg< 1, storage_type > p_val3;
    typedef arg< 2, storage_type > p_out;
    typedef boost::mpl::vector< p_val2, p_val3, p_out > accessor_list;

    aggregator_type< accessor_list > domain;

    test_expressions()
        : meta_(d1, d2, d3), di(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
          dj(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2), grid(di, dj),
#if FLOAT_PRECISION == 4
          verifier_(1e-6),
#else
          verifier_(1e-12),
#endif
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}},
          val2(meta_, 2., "val2"), val3(meta_, 3., "val3"), out(meta_, -555, "out"),
          reference(meta_, ::DEFAULT_VALUE, "reference"), domain(boost::fusion::make_vector(&val2, &val3, &out)) {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;
    }

    template < typename Computation >
    void execute_computation(Computation &comp) {
        comp->ready();
        comp->steady();
        comp->run();
#ifdef __CUDACC__
        out.d2h_update();
#endif
    }
};

#define EXPRESSION_TEST(INDEX, EXPR) \
    else if (eval.i() == INDEX && eval.j() == 0 && eval.k() == 0) eval(out()) = eval(EXPR);
#define EXPRESSION_TEST_RESULT(INDEX, RESULT) reference(INDEX, 0, 0) = RESULT;

#define EXPRESSION_TEST_DISABLED(INDEX, EXPR)
#define EXPRESSION_TEST_RESULT_DISABLED(INDEX, RESULT) reference(INDEX, 0, 0) = ::DEFAULT_VALUE;

namespace {
    struct test_functor {
        typedef in_accessor< 0, extent<>, 3 > val2;
        typedef in_accessor< 1, extent<>, 3 > val3;
        typedef inout_accessor< 2, extent<>, 3 > out;
        typedef boost::mpl::vector< val2, val3, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            constexpr gridtools::dimension< 1 > i{};
            constexpr gridtools::dimension< 2 > j{};
            constexpr gridtools::dimension< 3 > k{};

            if (false) // starts the cascade
                assert(false);

            EXPRESSION_TEST(0, val3() * val2())
            EXPRESSION_TEST(1, val3() + val2())
            EXPRESSION_TEST(2, val3() - val2())
            EXPRESSION_TEST(3, val3() / val2())

            EXPRESSION_TEST(4, val3(i, j, k) * val2(i, j, k))
            EXPRESSION_TEST(5, val3(i, j, k) + val2(i, j, k))
            EXPRESSION_TEST(6, val3(i, j, k) - val2(i, j, k))
            EXPRESSION_TEST(7, val3(i, j, k) / val2(i, j, k))

#ifdef CUDA8 // workaround for issue #342
            EXPRESSION_TEST(8, val3() * 3.)
#else
            EXPRESSION_TEST(8, val3(i, j, k) * 3.)
#endif
            EXPRESSION_TEST_DISABLED(9, 3. * val3())
            EXPRESSION_TEST_DISABLED(10, val3() * 3) // accessor<double> mult int
            EXPRESSION_TEST_DISABLED(11, 3 * val3()) // int mult accessor<double>

            EXPRESSION_TEST(12, val3() + 3.)
            EXPRESSION_TEST_DISABLED(13, 3. + val3())
            EXPRESSION_TEST_DISABLED(14, val3() + 3) // accessor<double> plus int
            EXPRESSION_TEST_DISABLED(15, 3 + val3()) // int plus accessor<double>

            EXPRESSION_TEST(16, val3() - 2.)
            EXPRESSION_TEST_DISABLED(17, 3. - val2())
            EXPRESSION_TEST_DISABLED(18, val3() - 2) // accessor<double> minus int
            EXPRESSION_TEST_DISABLED(19, 3 - val2()) // int minus accessor<double>

            EXPRESSION_TEST(20, val3() / 2.)
            EXPRESSION_TEST_DISABLED(21, 3. / val2())
            EXPRESSION_TEST_DISABLED(22, val3() / 2) // accessor<double> div int
            EXPRESSION_TEST_DISABLED(23, 3 / val2()) // int div accessor<double>

            EXPRESSION_TEST_DISABLED(24, -val2())
            EXPRESSION_TEST_DISABLED(25, +val2())

            EXPRESSION_TEST_DISABLED(26, val3() + 2. * val2())

            EXPRESSION_TEST(27, pow< 2 >(val3()))

            else eval(out()) = DEFAULT_VALUE;
        }
    };
}

TEST_F(test_expressions, test) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(
            execute< forward >(), gridtools::make_stage<::test_functor >(p_val2(), p_val3(), p_out())));

    EXPRESSION_TEST_RESULT(0, 6.);
    EXPRESSION_TEST_RESULT(1, 5.);
    EXPRESSION_TEST_RESULT(2, 1.);
    EXPRESSION_TEST_RESULT(3, 1.5);

    EXPRESSION_TEST_RESULT(4, 6.);
    EXPRESSION_TEST_RESULT(5, 5.);
    EXPRESSION_TEST_RESULT(6, 1.);
    EXPRESSION_TEST_RESULT(7, 1.5);

    EXPRESSION_TEST_RESULT(8, 9.);
    EXPRESSION_TEST_RESULT_DISABLED(9, 9.);
    EXPRESSION_TEST_RESULT_DISABLED(10, 9.);
    EXPRESSION_TEST_RESULT_DISABLED(11, 9.);

    EXPRESSION_TEST_RESULT(12, 6.);
    EXPRESSION_TEST_RESULT_DISABLED(13, 6.);
    EXPRESSION_TEST_RESULT_DISABLED(14, 6.);
    EXPRESSION_TEST_RESULT_DISABLED(15, 6.);

    EXPRESSION_TEST_RESULT(16, 1.);
    EXPRESSION_TEST_RESULT_DISABLED(17, 1.);
    EXPRESSION_TEST_RESULT_DISABLED(18, 1.);
    EXPRESSION_TEST_RESULT_DISABLED(19, 1.);

    EXPRESSION_TEST_RESULT(20, 1.5);
    EXPRESSION_TEST_RESULT_DISABLED(21, 1.5);
    EXPRESSION_TEST_RESULT_DISABLED(22, 1.5);
    EXPRESSION_TEST_RESULT_DISABLED(23, 1.5);

    EXPRESSION_TEST_RESULT_DISABLED(24, -2.);
    EXPRESSION_TEST_RESULT_DISABLED(25, +2.);

    EXPRESSION_TEST_RESULT_DISABLED(26, 7.);

    EXPRESSION_TEST_RESULT(27, 9.);

    execute_computation(comp);
    ASSERT_TRUE(verifier_.verify(grid, reference, out, verifier_halos));
}
