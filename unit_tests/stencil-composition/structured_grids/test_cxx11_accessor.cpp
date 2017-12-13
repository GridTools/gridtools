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

#include "gridtools.hpp"
#include "stencil-composition/accessor.hpp"
#include "stencil-composition/structured_grids/accessor_metafunctions.hpp"
#include "stencil-composition/expressions/expressions.hpp"
#include "stencil-composition/expandable_parameters/vector_accessor.hpp"

using namespace gridtools;

namespace interface {
    /** @brief simple interface
     */
    bool test_trivial() {
        accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > first(3, 2, -1);
        return first.get< 2 >() == 3 && first.get< 1 >() == 2 && first.get< 0 >() == -1;
    }

    bool test_array() {
#ifdef GT_NO_CONSTEXPR_OFFSET_TUPLE_CONSTR
        accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > first(array< int_t, 3 >{3, 2, -1});
#else
        constexpr accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > first(array< int_t, 3 >{3, 2, -1});
        GRIDTOOLS_STATIC_ASSERT((first.get< 2 >() == 3 && first.get< 1 >() == 2 && first.get< 0 >() == -1), "ERROR");
#endif
        return first.get< 2 >() == 3 && first.get< 1 >() == 2 && first.get< 0 >() == -1;
    }

    bool test_array_and_dim() {
#ifdef GT_NO_CONSTEXPR_OFFSET_TUPLE_CONSTR
        accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > first(
            array< int_t, 3 >{3, 2, -1}, dimension< 1 >(2));
#else
        constexpr accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > first(
            array< int_t, 3 >{3, 2, -1}, dimension< 1 >(2));
        GRIDTOOLS_STATIC_ASSERT((first.get< 2 >() == 3 && first.get< 1 >() == 4 && first.get< 0 >() == -1), "ERROR");
#endif
        return first.get< 2 >() == 3 && first.get< 1 >() == 4 && first.get< 0 >() == -1;
    }

    /** @brief interface with out-of-order optional arguments
     */
    bool test_alternative1() {
        accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 6 > first(dimension< 6 >(-6), dimension< 4 >(12));

        return first.get< 5 - 0 >() == 0 && first.get< 5 - 1 >() == 0 && first.get< 5 - 2 >() == 0 &&
               first.get< 5 - 3 >() == 12 && first.get< 5 - 4 >() == 0 && first.get< 5 - 5 >() == -6;
    }

    /** @brief interface with out-of-order optional arguments, represented as matlab indices
     */

    using namespace expressions;

    bool test_alternative2() {
#ifdef GT_NO_CONSTEXPR_OFFSET_TUPLE_CONSTR
        dimension< 1 > i;
        dimension< 2 > j;
        dimension< 3 > k;

        dimension< 4 > t;

        accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 4 > first(i - 5, j, dimension< 3 >(8), t + 2);
#else
        constexpr dimension< 1 > i;
        constexpr dimension< 2 > j;
        constexpr dimension< 3 > k;

        constexpr dimension< 4 > t;

        constexpr accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 4 > first(i - 5, j, dimension< 3 >(8), t + 2);

        GRIDTOOLS_STATIC_ASSERT(first.get< 3 - 0 >() == -5, "ERROR");
#endif
        return first.get< 3 - 0 >() == -5 && first.get< 3 - 1 >() == 0 && first.get< 3 - 2 >() == 8 &&
               first.get< 3 - 3 >() == 2;
    }

    /** @brief interface with aliases defined at compile-time

        allows to split a single field in its different components, assigning an offset to each component.
        The aforementioned offset is guaranteed to be treated as compile-time static constant value.
    */
    bool test_static_alias() {
#ifndef GT_NO_CONSTEXPR_OFFSET_TUPLE_CONSTR
        // mixing compile time and runtime values
        using t = dimension< 15 >;
        using arg_t = accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 15 >;
        using alias_t = alias< arg_t, t, dimension< 1 >, dimension< 7 > >::set< -3, 4, 2 >;

        alias_t first(dimension< 8 >(23), dimension< 3 >(-5));

        GRIDTOOLS_STATIC_ASSERT(alias_t::get_constexpr< 14 >() == 4, "ERROR");
        return first.get< 14 - 6 >() == 2 && first.get< 14 - 0 >() == 4 && first.get< 14 - 14 >() == -3 &&
               first.get< 14 - 7 >() == 23 && first.get< 14 - 2 >() == -5;
#else
        return true;
#endif
    }

    /** @brief interface with aliases defined at run-time

        allows to split a single field in its different components, assigning an offset to each component.
        The aforementioned offset can be a run-time value, or can be treated as static const when the instance has the
       constexpr specifier.
    */
    bool test_dynamic_alias() {

        // mixing caompile time and runtime values
        using t = dimension< 15 >;
        typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 15 > arg_t;
        alias< arg_t, t > field1(-3); // records the offset -3 as dynamic values
        alias< arg_t, t > field2(-1); // records the offset -1 as static const

        return field1(dimension< 3 >(-5), dimension< 1 >(1)).get< 14 - 0 >() == 1 &&
               field1(dimension< 3 >(-5), dimension< 1 >(1)).get< 14 - 2 >() == -5 &&
               field1(dimension< 3 >(-5), dimension< 1 >(1)).get< 14 - 14 >() == -3 &&
               field2(dimension< 3 >(-5), dimension< 1 >(1)).get< 14 - 0 >() == 1 &&
               field2(dimension< 3 >(-5), dimension< 1 >(1)).get< 14 - 2 >() == -5 &&
               field2(dimension< 3 >(-5), dimension< 1 >(1)).get< 14 - 14 >() == -1;
    }

} // namespace interface

using namespace interface;

TEST(Accessor, is_accessor) {
    GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor< 6, enumtype::inout, extent< 3, 4, 4, 5 > > >::value) == true, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor< 2, enumtype::in > >::value) == true, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< int >::value) == false, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< double & >::value) == false, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< double const & >::value) == false, "");
}

TEST(Accessor, Trivial) { EXPECT_TRUE(test_trivial()); }

TEST(Accessor, Array) { EXPECT_TRUE(test_array()); }

TEST(Accessor, Alternative) { EXPECT_TRUE(test_alternative1()); }

TEST(Accessor, Cxx11Alternative) { EXPECT_TRUE(test_alternative2()); }

TEST(Accessor, Cxx11DynamicAlias) { EXPECT_TRUE(test_dynamic_alias()); }

TEST(Accessor, Cxx11StaticAlias) { EXPECT_TRUE(test_static_alias()); }
