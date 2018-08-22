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

#include <boost/mpl/pair.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/generic_metafunctions/type_traits.hpp"
#include "../common/host_device.hpp"
#include "hasdo.hpp"
#include "interval.hpp"
#include "level.hpp"

namespace gridtools {

    namespace _impl {
        template <class ToIndex, class FromIndex>
        struct no_gap : bool_constant<ToIndex::value + 1 == FromIndex::value &&
                                      GT_META_CALL(index_to_level, ToIndex)::splitter ==
                                          GT_META_CALL(index_to_level, FromIndex)::splitter> {};
        template <class Functor>
        struct is_from_index {
            template <class Index>
            GT_META_DEFINE_ALIAS(apply, has_do, (Functor, GT_META_CALL(index_to_level, Index)));
        };

    } // namespace _impl

    /**
     * @struct compute_functor_do_methods
     * Meta function computing a vector containing all do method overloads inside the given axis interval
     * (note that the result vector contains pairs of from and to level indexes instead of intervals)
     */
    template <class Functor, class Axis>
    struct compute_functor_do_methods;

    template <class Functor, class AxisFromLevel, class AxisToLevel>
    struct compute_functor_do_methods<Functor, interval<AxisFromLevel, AxisToLevel>> {

        GRIDTOOLS_STATIC_ASSERT(!has_do<Functor>::value,
            "A functor's Do method is found to have only one argument, when it is supposed to have two");

        using from_index_t = GT_META_CALL(level_to_index, AxisFromLevel);
        using to_index_t = GT_META_CALL(level_to_index, AxisToLevel);
        using all_indices_t = typename make_range<from_index_t, to_index_t>::type;

        using from_indices_t = GT_META_CALL(
            meta::filter, (_impl::is_from_index<Functor>::template apply, all_indices_t));

        GRIDTOOLS_STATIC_ASSERT(meta::length<from_indices_t>::value, "no Do methods for an axis interval found");

#if GT_BROKEN_TEMPLATE_ALIASES
#define LAZY_DEDUCE_TO_INDEX deduce_to_index
#else
#define LAZY_DEDUCE_TO_INDEX lazy_deduce_to_index
#endif
        template <class FromIndex>
        struct LAZY_DEDUCE_TO_INDEX {
            template <class Index>
            GT_META_DEFINE_ALIAS(is_to_index, has_do, (Functor, GT_META_CALL(make_interval, (FromIndex, Index))));

            using indices_t = typename make_range<FromIndex, to_index_t>::type;
            using to_indices_t = GT_META_CALL(meta::filter, (is_to_index, indices_t));

            GRIDTOOLS_STATIC_ASSERT(
                meta::length<to_indices_t>::value, "no Do methods for an axis interval found with a given from level");

            GRIDTOOLS_STATIC_ASSERT(
                meta::length<to_indices_t>::value == 1, "ambiguous Do methods with a given from level");

            using type = GT_META_CALL(meta::first, to_indices_t);
        };
#undef LAZY_DEDUCE_TO_INDEX
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class FromIndex>
        using deduce_to_index = typename lazy_deduce_to_index<FromIndex>::type;

#endif

        using to_indices_t = GT_META_CALL(meta::transform, (deduce_to_index, from_indices_t));

        using no_gaps_t = GT_META_CALL(meta::transform,
            (_impl::no_gap, GT_META_CALL(meta::pop_back, to_indices_t), GT_META_CALL(meta::pop_front, from_indices_t)));

        GRIDTOOLS_STATIC_ASSERT(meta::all<no_gaps_t>::value, "Do methods are not continuous");

        using tuple_of_lists_t = GT_META_CALL(meta::zip, (from_indices_t, to_indices_t));

        using type = GT_META_CALL(
            meta::transform, (GT_META_CALL(meta::rename, boost::mpl::pair)::apply, tuple_of_lists_t));
    };
} // namespace gridtools
