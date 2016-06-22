/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
/*@file */

#include <gridtools.hpp>
#include <stencil_composition/accessor.hpp>
#include <stencil_composition/expressions.hpp>

namespace interface {
    /** @brief simple interface
     */
    bool test_trivial() {
        accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > first(3, 2, -1);
        return first.get< 2 >() == 3 && first.get< 1 >() == 2 && first.get< 0 >() == -1;
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
#ifdef CXX11_ENABLED

    using namespace expressions;

    bool test_alternative2() {

        constexpr dimension< 1 >::Index i;
        constexpr dimension< 4 >::Index t;
        constexpr accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 4 > first(i - 5, t + 2, dimension< 3 >(8));

        GRIDTOOLS_STATIC_ASSERT(first.get< 3 - 0 >() == -5, "ERROR");
        return first.get< 3 - 0 >() == -5 && first.get< 3 - 1 >() == 0 && first.get< 3 - 2 >() == 8 &&
               first.get< 3 - 3 >() == 2;
    }

    /** @brief interface with aliases defined at compile-time

        allows to split a single field in its different components, assigning an offset to each component.
        The aforementioned offset is guaranteed to be treated as compile-time static constant value.
    */
    bool test_static_alias() {

        // mixing compile time and runtime values
        using t = dimension< 15 >;
        typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 15 > arg_t;
        using alias_t = alias< arg_t, t, dimension< 1 >, dimension< 7 > >::set< -3, 4, 2 >;

        alias_t first(dimension< 8 >(23), dimension< 3 >(-5));

        GRIDTOOLS_STATIC_ASSERT(alias_t::get_constexpr< 14 >() == 4, "ERROR");
        return first.get< 14 - 6 >() == 2 && first.get< 14 - 0 >() == 4 && first.get< 14 - 14 >() == -3 &&
               first.get< 14 - 7 >() == 23 && first.get< 14 - 2 >() == -5;
    }

    /** @brief interface with aliases defined at run-time

        allows to split a single field in its different components, assigning an offset to each component.
        The aforementioned offset can be a run-time value, or can be treated as static const when the instance has the
       constexpr specifier.
    */
    bool test_dynamic_alias() {

        // mixing caompile time and runtime values
        using t = dimension< 15 >;
        using x = dimension< 1 >;
        using z = dimension< 3 >;
        typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 15 > arg_t;
        alias< arg_t, t > field1(-3); // records the offset -3 as dynamic values
        alias< arg_t, t > field2(-1); // records the offset -1 as static const

        return field1(z(-5), x(1)).get< 14 - 0 >() == 1 && field1(z(-5), x(1)).get< 14 - 2 >() == -5 &&
               field1(z(-5), x(1)).get< 14 - 14 >() == -3 && field2(z(-5), x(1)).get< 14 - 0 >() == 1 &&
               field2(z(-5), x(1)).get< 14 - 2 >() == -5 && field2(z(-5), x(1)).get< 14 - 14 >() == -1;
    }

#endif
} // namespace interface
