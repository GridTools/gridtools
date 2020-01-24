/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/cartesian.hpp>

namespace gridtools {
    namespace cartesian {
        namespace {
            struct stage {
                using in = in_accessor<0, extent<-1, 1>>;
                using out = inout_accessor<1>;
                using param_list = make_param_list<in, out>;

                template <class Eval>
                GT_FUNCTION static void apply(Eval) {}
            };

            struct a {};
            struct b {};
            struct c {};
            struct d {};

            constexpr auto spec = execute_parallel()
                                      .stage_with_extent(extent<-5, 5>(), stage(), a(), b())
                                      .stage_with_extent(extent<-3, 3>(), stage(), b(), c())
                                      .stage(stage(), c(), d());

            template <class Arg, int_t... Is>
            constexpr bool testee = std::is_same<decltype(get_arg_extent(spec, Arg())), extent<Is...>>::value;

            static_assert(testee<a, -6, 6>, "");
            static_assert(testee<b, -5, 5>, "");
            static_assert(testee<c, -3, 3>, "");
            static_assert(testee<d>, "");
        } // namespace
    }     // namespace cartesian
} // namespace gridtools
