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
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

// TODO this is actually not a proper unit test
// as it is not standalone testing expressions

namespace test_expressions_detail {
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

    const uint_t d1 = 13;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 0;

    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    typedef gridtools::BACKEND::storage_info< 0, layout_t > meta_t;
    typedef gridtools::BACKEND::storage_type< double, meta_t >::type storage_type;

    meta_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid< test_expressions_detail::axis > grid;

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
          reference(meta_, -999., "reference"), domain(boost::fusion::make_vector(&val2, &val3, &out)) {
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

/*
 * Macro for building an expression test:
 * - NAME test name
 * - EXPR expression to test, e.g. val2()*val3(), this is placed inside an eval(...)
 * - RESULT result of the expression assuming the same operation on all points
 *
 * Two accessors are defined:
 *  val2 initialized to 2.
 *  val3 initialized to 3.
 */
#define EXPRESSION_TEST(NAME, EXPR, RESULT)                                                         \
    struct NAME {                                                                                   \
        typedef in_accessor< 0, extent<>, 3 > val2;                                                 \
        typedef in_accessor< 1, extent<>, 3 > val3;                                                 \
        typedef inout_accessor< 2, extent<>, 3 > out;                                               \
        typedef boost::mpl::vector< val2, val3, out > arg_list;                                     \
        template < typename Evaluation >                                                            \
        GT_FUNCTION static void Do(Evaluation const &eval, test_expressions_detail::x_interval) {   \
            constexpr gridtools::dimension< 1 > i{};                                                \
            constexpr gridtools::dimension< 2 > j{};                                                \
            constexpr gridtools::dimension< 3 > k{};                                                \
            eval(out()) = eval(EXPR);                                                               \
        }                                                                                           \
    };                                                                                              \
    TEST_F(test_expressions, NAME) {                                                                \
        reference.initialize(RESULT);                                                               \
        auto comp = gridtools::make_computation< gridtools::BACKEND >(                              \
            domain,                                                                                 \
            grid,                                                                                   \
            gridtools::make_multistage(                                                             \
                execute< forward >(), gridtools::make_stage< NAME >(p_val2(), p_val3(), p_out()))); \
        execute_computation(comp);                                                                  \
        ASSERT_TRUE(verifier_.verify(grid, out, reference, verifier_halos));                        \
    }

#define EXPRESSION_TEST_DISABLED(NAME, EXPR, RESULT) \
    TEST_F(test_expressions, DISABLED_##NAME) {}

#ifdef CUDA8 // issue #342
EXPRESSION_TEST(accessor_mult_accessor, val3() * val2(), 6.)
EXPRESSION_TEST(accessor_plus_accessor, val3() + val2(), 5.)
EXPRESSION_TEST(accessor_minus_accessor, val3() - val2(), 1.)
#endif
EXPRESSION_TEST(accessor_div_accessor, val3() / val2(), 1.5)

EXPRESSION_TEST(accessor_mult_accessor_with_ijk_syntax, val3(i, j, k) * val2(i, j, k), 6.)
EXPRESSION_TEST(accessor_plus_accessor_with_ijk_syntax, val3(i, j, k) + val2(i, j, k), 5.)
EXPRESSION_TEST(accessor_minus_accessor_with_ijk_syntax, val3(i, j, k) - val2(i, j, k), 1.)
EXPRESSION_TEST(accessor_div_accessor_with_ijk_syntax, val3(i, j, k) / val2(i, j, k), 1.5)

#ifdef CUDA8 // issue #342
EXPRESSION_TEST(accessor_mult_double, val3() * 3., 9.)
#else
EXPRESSION_TEST(accessor_mult_double, val3(i, j, k) * 3., 9.)
#endif
EXPRESSION_TEST_DISABLED(double_mult_accessor, 3. * val3(), 9.)
EXPRESSION_TEST_DISABLED(accessor_mult_int, val3() * 3, 9.)
EXPRESSION_TEST_DISABLED(int_mult_accessor, 3 * val3(), 9.)

EXPRESSION_TEST(accessor_div_double, val3() / 3., 1.)
EXPRESSION_TEST_DISABLED(double_div_accessor, 3. / val3(), 1.)
EXPRESSION_TEST_DISABLED(accessor_div_int, val3() / 3, 1.)
EXPRESSION_TEST_DISABLED(int_div_accessor, 3 / val3(), 1.)

EXPRESSION_TEST(accessor_plus_double, val2() + 3., 5.)
EXPRESSION_TEST_DISABLED(double_plus_accessor, 3. + val2(), 5.)
EXPRESSION_TEST_DISABLED(accessor_plus_int, val2() + 3, 5.)
EXPRESSION_TEST_DISABLED(int_plus_accessor, 3 + val2(), 5.)

EXPRESSION_TEST(accessor_minus_double, val3() - 2., 1.)
EXPRESSION_TEST_DISABLED(double_minus_accessor, 3. - val2(), 1.)
EXPRESSION_TEST_DISABLED(accessor_minus_int, val3() - 2, 1.)
EXPRESSION_TEST_DISABLED(int_minus_accessor, 3 - val2(), 1.)

EXPRESSION_TEST_DISABLED(minus_sign, -val2(), -2.)
EXPRESSION_TEST_DISABLED(plus_sign, +val2(), 2.)

EXPRESSION_TEST_DISABLED(accessor_plus_double_mult_accessor, val3() + 2. * val2(), 7.)

EXPRESSION_TEST(pow_2_accessor, pow< 2 >(val3()), 9.)
