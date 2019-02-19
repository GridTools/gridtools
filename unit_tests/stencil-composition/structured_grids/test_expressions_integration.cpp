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

/**
 * Integration test for testing expressions inside a computation.
 * The test setup is not very nice but it was designed that way to minimize compilation time, i.e. to test everything
 * within one make_computation call.
 *
 *
 */

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/verifier.hpp>

using namespace gridtools;
using namespace gridtools::execute;
using namespace gridtools::expressions;

namespace {
    const double DEFAULT_VALUE = -999.;
}

class test_expressions : public testing::Test {
  protected:
    const uint_t d1 = 100;
    const uint_t d2 = 9;
    const uint_t d3 = 7;

    using storage_info_t = storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3>;
    using data_store_t = storage_traits<backend_t::backend_id_t>::data_store_t<float_type, storage_info_t>;

    storage_info_t storage_info_;

    gridtools::grid<axis<1>::axis_interval_t> grid;

    verifier verifier_;
    array<array<uint_t, 2>, 3> verifier_halos;

    data_store_t val2;
    data_store_t val3;
    data_store_t out;
    data_store_t reference;

    typedef arg<0, data_store_t> p_val2;
    typedef arg<1, data_store_t> p_val3;
    typedef arg<2, data_store_t> p_out;

    test_expressions()
        : storage_info_(d1, d2, d3), grid(make_grid(d1, d2, d3)),
#if GT_FLOAT_PRECISION == 4
          verifier_(1e-6),
#else
          verifier_(1e-12),
#endif
          verifier_halos{{{0, 0}, {0, 0}, {0, 0}}}, val2(storage_info_, 2., "val2"), val3(storage_info_, 3., "val3"),
          out(storage_info_, -555, "out"), reference(storage_info_, ::DEFAULT_VALUE, "reference") {
    }

    template <typename Computation>
    void execute_computation(Computation &comp) {
        comp.run(p_val2() = val2, p_val3() = val3, p_out() = out);
#ifdef __CUDACC__
        out.sync();
#endif
    }
};

#define EXPRESSION_TEST(EXPR)                                         \
    else if (eval.i() == index++ && eval.j() == 0 && eval.k() == 0) { \
        eval(out()) = eval(EXPR);                                     \
    }
#define EXPRESSION_TEST_RESULT(RESULT)               \
    make_host_view(reference)(index, 0, 0) = RESULT; \
    index++;

namespace {
    struct test_functor {
        typedef in_accessor<0, extent<>, 3> val2;
        typedef in_accessor<1, extent<>, 3> val3;
        typedef inout_accessor<2, extent<>, 3> out;
        typedef make_param_list<val2, val3, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            constexpr gridtools::dimension<1> i{};
            constexpr gridtools::dimension<2> j{};
            constexpr gridtools::dimension<3> k{};

            // starts the cascade
            int index = 0;
            if (false)
                assert(false);
            /*
             * Put expression test here in the form
             * EXPRESSION_TEST( <expr> ) where <expr> is the expression to test.
             * Then put the result below. The order of EXPRESSION_TESTs and EXPRESSION_TEST_RESULTs has to be preserved
             */
            EXPRESSION_TEST(val3() * val2())
            EXPRESSION_TEST(val3() + val2())
            EXPRESSION_TEST(val3() - val2())
            EXPRESSION_TEST(val3() / val2())

            EXPRESSION_TEST(val3(i, j, k) * val2(i, j, k))
            EXPRESSION_TEST(val3(i, j, k) + val2(i, j, k))
            EXPRESSION_TEST(val3(i, j, k) - val2(i, j, k))
            EXPRESSION_TEST(val3(i, j, k) / val2(i, j, k))

            EXPRESSION_TEST(val3() * 3.)
            EXPRESSION_TEST(3. * val3())
            EXPRESSION_TEST(val3() * 3) // accessor<double> mult int
            EXPRESSION_TEST(3 * val3()) // int mult accessor<double>

            EXPRESSION_TEST(val3() + 3.)
            EXPRESSION_TEST(3. + val3())
            EXPRESSION_TEST(val3() + 3) // accessor<double> plus int
            EXPRESSION_TEST(3 + val3()) // int plus accessor<double>

            EXPRESSION_TEST(val3() - 2.)
            EXPRESSION_TEST(3. - val2())
            EXPRESSION_TEST(val3() - 2) // accessor<double> minus int
            EXPRESSION_TEST(3 - val2()) // int minus accessor<double>
                                        //
            EXPRESSION_TEST(val3() / 2.)
            EXPRESSION_TEST(3. / val2())
            EXPRESSION_TEST(val3() / 2) // accessor<double> div int
            EXPRESSION_TEST(3 / val2()) // int div accessor<double>

            EXPRESSION_TEST(-val2())
            EXPRESSION_TEST(+val2())

            EXPRESSION_TEST(val3() + 2. * val2())

            EXPRESSION_TEST(pow<2>(val3()))

            else eval(out()) = DEFAULT_VALUE;
        }
    };
} // namespace

TEST_F(test_expressions, integration_test) {
    auto comp = gridtools::make_positional_computation<backend_t>(grid,
        gridtools::make_multistage(
            execute::forward(), gridtools::make_stage<::test_functor>(p_val2(), p_val3(), p_out())));

    int index = 0;

    /*
     * Put test result here in the same order as the expressions were given above.
     */
    EXPRESSION_TEST_RESULT(6.);
    EXPRESSION_TEST_RESULT(5.);
    EXPRESSION_TEST_RESULT(1.);
    EXPRESSION_TEST_RESULT(1.5);

    EXPRESSION_TEST_RESULT(6.);
    EXPRESSION_TEST_RESULT(5.);
    EXPRESSION_TEST_RESULT(1.);
    EXPRESSION_TEST_RESULT(1.5);

    EXPRESSION_TEST_RESULT(9.);
    EXPRESSION_TEST_RESULT(9.);
    EXPRESSION_TEST_RESULT(9.);
    EXPRESSION_TEST_RESULT(9.);

    EXPRESSION_TEST_RESULT(6.);
    EXPRESSION_TEST_RESULT(6.);
    EXPRESSION_TEST_RESULT(6.);
    EXPRESSION_TEST_RESULT(6.);

    EXPRESSION_TEST_RESULT(1.);
    EXPRESSION_TEST_RESULT(1.);
    EXPRESSION_TEST_RESULT(1.);
    EXPRESSION_TEST_RESULT(1.);

    EXPRESSION_TEST_RESULT(1.5);
    EXPRESSION_TEST_RESULT(1.5);
    EXPRESSION_TEST_RESULT(1.5);
    EXPRESSION_TEST_RESULT(1.5);

    EXPRESSION_TEST_RESULT(-2.);
    EXPRESSION_TEST_RESULT(+2.);

    EXPRESSION_TEST_RESULT(7.);

    EXPRESSION_TEST_RESULT(9.);

    execute_computation(comp);
    ASSERT_TRUE(verifier_.verify(grid, reference, out, verifier_halos));
}
