/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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
