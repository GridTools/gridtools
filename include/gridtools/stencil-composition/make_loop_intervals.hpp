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
#include "./bind_functor_with_interval.hpp"
#include "./esf.hpp"
#include "./esf_metafunctions.hpp"
#include "./interval.hpp"
#include "./level.hpp"
#include "./loop_interval.hpp"

namespace gridtools {

    namespace _impl {

        template <class Index, class Esfs>
        struct loop_level {
            using type = loop_level;
        };

        template <class Index>
        struct make_loop_level_impl {
#if GT_BROKEN_TEMPLATE_ALIASES
            template <class Functor>
            struct bind_functor : bind_functor_with_interval<Functor, Index> {};
            template <class Esf>
            struct bind_esf : esf_transform_functor<bind_functor, Esf>;
#else
            template <class Functor>
            using bind_functor = typename bind_functor_with_interval<Functor, Index>::type;
            template <class Esf>
            using bind_esf = typename esf_transform_functor<bind_functor, Esf>::type;
#endif
            template <class Esfs>
            GT_META_DEFINE_ALIAS(bind_esfs, meta::transform, (bind_esf, Esfs));

            template <class Esfs>
            GT_META_DEFINE_ALIAS(apply,
                loop_level,
                (Index, GT_META_CALL(meta::filter, (is_esf_descriptor, GT_META_CALL(bind_esfs, Esfs)))));
        };

        template <class Esfs>
        struct make_loop_level {
            template <class Index>
#if GT_BROKEN_TEMPLATE_ALIASES
            struct apply : make_loop_level_impl<Index>::template apply<Esfs> {
            };
#else
            using apply = typename make_loop_level_impl<Index>::template apply<Esfs>;
#endif
        };

        template <class Acc,
            class Cur,
            class Prev = GT_META_CALL(meta::last, Acc),
            class Esfs = GT_META_CALL(meta::second, Cur),
            class PrevEsfs = GT_META_CALL(meta::second, Prev)>
        GT_META_DEFINE_ALIAS(loop_level_inserter,
            meta::if_,
            (std::is_same<Esfs, PrevEsfs>, Acc, GT_META_CALL(meta::push_back, (Acc, Cur))));

        template <class LoopLevel,
            class NextLoopLevel,
            class FromIndex = GT_META_CALL(meta::first, LoopLevel),
            class ToIndex = typename GT_META_CALL(meta::first, NextLoopLevel)::prior,
            class Esfs = GT_META_CALL(meta::second, LoopLevel)>
        GT_META_DEFINE_ALIAS(make_loop_interval,
            loop_interval,
            (GT_META_CALL(index_to_level, FromIndex), GT_META_CALL(index_to_level, ToIndex), Esfs));

        template <class LoopInterval>
        struct has_esfs : std::false_type {};

        template <class From, class To, class Esfs>
        struct has_esfs<loop_interval<From, To, Esfs>> : bool_constant<meta::length<Esfs>::value != 0> {};
    } // namespace _impl

#if GT_BROKEN_TEMPLATE_ALIASES
#define LAZY_MAKE_LOOP_INTERVALS make_loop_intervals
#else
#define LAZY_MAKE_LOOP_INTERVALS lazy_make_loop_intervals
#endif

    template <class Esfs, class AxisInterval>
    struct LAZY_MAKE_LOOP_INTERVALS {
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_esf_descriptor, Esfs>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_interval<AxisInterval>::value, GT_INTERNAL_ERROR);

        using from_index_t = GT_META_CALL(level_to_index, typename AxisInterval::FromLevel);
        using to_index_t = GT_META_CALL(level_to_index, typename AxisInterval::ToLevel);
        using indices_t = typename make_range<from_index_t, to_index_t>::type;
        using all_loop_levels_t = GT_META_CALL(
            meta::transform, (_impl::make_loop_level<Esfs>::template apply, indices_t));
        using first_loop_level_t = GT_META_CALL(meta::first, all_loop_levels_t);
        using rest_of_loop_levels_t = GT_META_CALL(meta::pop_front, all_loop_levels_t);
        using loop_levels_t = GT_META_CALL(
            meta::lfold, (_impl::loop_level_inserter, meta::list<first_loop_level_t>, rest_of_loop_levels_t));
        using next_loop_levels_t = GT_META_CALL(meta::push_back,
            (GT_META_CALL(meta::pop_front, loop_levels_t), _impl::loop_level<typename to_index_t::next, meta::list<>>));
        using loop_intervals_t = GT_META_CALL(
            meta::transform, (_impl::make_loop_interval, loop_levels_t, next_loop_levels_t));
        using type = GT_META_CALL(meta::filter, (_impl::has_esfs, loop_intervals_t));
    };

#undef LAZY_MAKE_LOOP_INTERVALS
#if !GT_BROKEN_TEMPLATE_ALIASES
    template <class Esfs, class AxisInterval>
    using make_loop_intervals = typename lazy_make_loop_intervals<Esfs, AxisInterval>::type;
#endif
} // namespace gridtools
