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
#pragma once

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "./interval.hpp"
#include "./level.hpp"
#include "./loop_interval.hpp"

namespace gridtools {
    namespace _impl {
        template <class Index, class Stages>
        struct loop_level {
            using type = loop_level;
        };

        template <template <class...> class StagesMaker>
        struct make_loop_level {
            template <class Index>
            GT_META_DEFINE_ALIAS(apply, loop_level, (Index, GT_META_CALL(StagesMaker, Index)));
        };

        template <class Acc, class Cur, class Prev = GT_META_CALL(meta::last, Acc)>
        GT_META_DEFINE_ALIAS(loop_level_inserter,
            meta::if_,
            (std::is_same<GT_META_CALL(meta::second, Cur), GT_META_CALL(meta::second, Prev)>,
                Acc,
                GT_META_CALL(meta::push_back, (Acc, Cur))));

        template <class LoopLevel, class NextLoopLevel, class FromIndex = GT_META_CALL(meta::first, LoopLevel)>
        GT_META_DEFINE_ALIAS(make_loop_interval,
            loop_interval,
            (GT_META_CALL(index_to_level, FromIndex),
                GT_META_CALL(index_to_level, typename GT_META_CALL(meta::first, NextLoopLevel)::prior),
                GT_META_CALL(meta::second, LoopLevel)));

        template <class LoopInterval>
        struct has_stages : std::false_type {};

        template <class From, class To, class Stages>
        struct has_stages<loop_interval<From, To, Stages>> : bool_constant<meta::length<Stages>::value != 0> {};
    } // namespace _impl

    GT_META_LAZY_NAMESPASE {
        template <template <class...> class StagesMaker, class Interval>
        struct make_loop_intervals {
            GRIDTOOLS_STATIC_ASSERT(is_interval<Interval>::value, GT_INTERNAL_ERROR);

            using from_index_t = GT_META_CALL(level_to_index, typename Interval::FromLevel);
            using to_index_t = GT_META_CALL(level_to_index, typename Interval::ToLevel);
            using indices_t = typename make_range<from_index_t, to_index_t>::type;
            using all_loop_levels_t = GT_META_CALL(
                meta::transform, (_impl::make_loop_level<StagesMaker>::template apply, indices_t));
            using first_loop_level_t = GT_META_CALL(meta::first, all_loop_levels_t);
            using rest_of_loop_levels_t = GT_META_CALL(meta::pop_front, all_loop_levels_t);
            using loop_levels_t = GT_META_CALL(
                meta::lfold, (_impl::loop_level_inserter, meta::list<first_loop_level_t>, rest_of_loop_levels_t));
            using next_loop_levels_t = GT_META_CALL(meta::push_back,
                (GT_META_CALL(meta::pop_front, loop_levels_t),
                    _impl::loop_level<typename to_index_t::next, meta::list<>>));
            using loop_intervals_t = GT_META_CALL(
                meta::transform, (_impl::make_loop_interval, loop_levels_t, next_loop_levels_t));
            using type = GT_META_CALL(meta::filter, (_impl::has_stages, loop_intervals_t));
        };
    }
    GT_META_DELEGATE_TO_LAZY(
        make_loop_intervals, (template <class...> class StagesMaker, class Interval), (StagesMaker, Interval));

} // namespace gridtools
