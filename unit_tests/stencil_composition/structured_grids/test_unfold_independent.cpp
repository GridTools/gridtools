/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/esf_metafunctions.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>

namespace gridtools {
    namespace {
        struct functor {
            GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(a0), GT_INOUT_ACCESSOR(a1));
        };

        struct fake_storage_type {
            using value_type = int;
        };

        constexpr arg<0, fake_storage_type> p0 = {};
        constexpr arg<1, fake_storage_type> p1 = {};
        constexpr auto stage = make_stage<functor>(p0, p1);

        using mss_type = decltype(make_multistage(
            execute::forward(), stage, stage, stage, make_independent(stage, stage, make_independent(stage, stage))));

        using testee_t = GT_META_CALL(unwrap_independent, mss_type::esf_sequence_t);

        static_assert(meta::length<testee_t>::value == 7, "");
        static_assert(meta::all_of<is_esf_descriptor, testee_t>::value, "");

        TEST(dummy, dumy) {}
    } // namespace
} // namespace gridtools
