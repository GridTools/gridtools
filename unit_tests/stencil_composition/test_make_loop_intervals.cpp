/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/make_loop_intervals.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/stencil_composition/interval.hpp>
#include <gridtools/stencil_composition/level.hpp>

namespace gridtools {
    namespace {
        using meta::list;

        template <uint_t Splitter, int_t Offset>
        using lev = level<Splitter, Offset, 3>;

        using from_t = lev<0, 1>;
        using to_t = lev<3, -1>;
        using axis_interval_t = interval<from_t, to_t>;

        struct stage1 {};
        struct stage2 {};

        template <template <class...> class StagesMaker>
        using testee = make_loop_intervals<StagesMaker, axis_interval_t>;

        namespace no_stages {
            using testee_t = testee<meta::always<list<>>::apply>;
            static_assert(std::is_same<testee_t, list<>>{}, "");
        } // namespace no_stages

        namespace simple {
            using testee_t = testee<meta::always<list<stage1>>::apply>;
            static_assert(std::is_same<testee_t, list<loop_interval<from_t, to_t, list<stage1>>>>{}, "");
        } // namespace simple

        namespace overlap {
            template <uint_t Splitter, int_t Offset>
            constexpr int_t idx() {
                return level_to_index<lev<Splitter, Offset>>::value;
            }
            constexpr bool has_stage1(int_t i) { return i >= idx<0, 2>() && i < idx<2, 1>(); }
            constexpr bool has_stage2(int_t i) { return i >= idx<1, 1>() && i < idx<3, -1>(); }

            template <class Index>
            using stages_maker = meta::filter<meta::not_<std::is_void>::apply,
                meta::list<std::conditional_t<has_stage1(Index::value), stage1, void>,
                    std::conditional_t<has_stage2(Index::value), stage2, void>>>;

            using testee_t = testee<stages_maker>;

            using expected_t = list<loop_interval<lev<0, 2>, lev<1, -1>, list<stage1>>,
                loop_interval<lev<1, 1>, lev<2, -1>, list<stage1, stage2>>,
                loop_interval<lev<2, 1>, lev<3, -2>, list<stage2>>>;

            static_assert(std::is_same<testee_t, expected_t>{}, "");
        } // namespace overlap

        TEST(dummy, dummy) {}
    } // namespace
} // namespace gridtools
