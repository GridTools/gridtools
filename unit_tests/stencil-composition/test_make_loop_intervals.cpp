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

#include <gridtools/stencil-composition/make_loop_intervals.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/generic_metafunctions/meta.hpp>
#include <gridtools/stencil-composition/interval.hpp>
#include <gridtools/stencil-composition/level.hpp>

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
        GT_META_DEFINE_ALIAS(testee, make_loop_intervals, (StagesMaker, axis_interval_t));

        namespace no_stages {
            using testee_t = GT_META_CALL(testee, meta::always<list<>>::apply);
            static_assert(std::is_same<testee_t, list<>>{}, "");
        } // namespace no_stages

        namespace simple {
            using testee_t = GT_META_CALL(testee, meta::always<list<stage1>>::apply);
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
            GT_META_DEFINE_ALIAS(stages_maker,
                meta::filter,
                (meta::not_<std::is_void>::apply,
                    meta::list<conditional_t<has_stage1(Index::value), stage1, void>,
                        conditional_t<has_stage2(Index::value), stage2, void>>));

            using testee_t = GT_META_CALL(testee, stages_maker);

            using expected_t = list<loop_interval<lev<0, 2>, lev<1, -1>, list<stage1>>,
                loop_interval<lev<1, 1>, lev<2, -1>, list<stage1, stage2>>,
                loop_interval<lev<2, 1>, lev<3, -2>, list<stage2>>>;

            static_assert(std::is_same<testee_t, expected_t>{}, "");
        } // namespace overlap

        TEST(dummy, dummy) {}
    } // namespace
} // namespace gridtools
