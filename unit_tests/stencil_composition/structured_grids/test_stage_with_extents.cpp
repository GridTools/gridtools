/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/computation_fixture.hpp>

namespace gridtools {
    namespace {
        struct stage {
            using in = in_accessor<0, extent<-1, 1>>;
            using out = inout_accessor<1>;
            using param_list = make_param_list<in, out>;

            template <class Eval>
            GT_FUNCTION static void apply(Eval) {}
        };

        struct stage_with_extents : computation_fixture<10> {
            stage_with_extents() : computation_fixture<10>{10, 10, 10} {}
        };

        TEST_F(stage_with_extents, smoke) {
            auto comp = make_computation(make_multistage(execute::forward(),
                make_stage_with_extent<stage, extent<-5, 5>>(p_0, p_1),
                make_stage_with_extent<stage, extent<-3, 3>>(p_1, p_2),
                make_stage<stage>(p_2, p_3)));

            using extent_0 = decltype(comp.get_arg_extent(p_0));
            using extent_1 = decltype(comp.get_arg_extent(p_1));
            using extent_2 = decltype(comp.get_arg_extent(p_2));
            using extent_3 = decltype(comp.get_arg_extent(p_3));

            static_assert(std::is_same<extent_0, extent<-6, 6>>(), "");
            static_assert(std::is_same<extent_1, extent<-4, 4>>(), "");
            static_assert(std::is_same<extent_2, extent<-1, 1>>(), "");
            static_assert(std::is_same<extent_3, extent<>>(), "");
        }
    } // namespace
} // namespace gridtools
