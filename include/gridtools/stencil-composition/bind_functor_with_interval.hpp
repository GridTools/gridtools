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

#include <type_traits>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/generic_metafunctions/type_traits.hpp"
#include "../common/host_device.hpp"
#include "./hasdo.hpp"
#include "./interval.hpp"
#include "./level.hpp"

namespace gridtools {
    namespace _impl {
        template <class Functor, class Index, bool HasDo = has_do<Functor, GT_META_CALL(index_to_level, Index)>::value>
        struct find_from_index {
            using type = typename find_from_index<Functor, typename Index::prior>::type;
        };
        template <class Functor, class Index>
        struct find_from_index<Functor, Index, true> {
            using type = Index;
        };
        template <class Functor, int_t Lim>
        struct find_from_index<Functor, level_index<0, Lim>, false> {
            using type = void;
        };

        template <class Functor,
            class FromIndex,
            class ToIndex = FromIndex,
            bool HasDo = has_do<Functor, GT_META_CALL(make_interval, (FromIndex, ToIndex))>::value>
        struct find_to_index {
            using type = typename find_to_index<Functor, FromIndex, typename ToIndex::next>::type;
        };

        template <class Functor, class FromIndex, class ToIndex>
        struct find_to_index<Functor, FromIndex, ToIndex, true> {
            using type = ToIndex;
        };

        template <class Functor,
            class FromIndex,
            class ToIndex = FromIndex,
            class Interval = GT_META_CALL(make_interval, (FromIndex, ToIndex)),
            bool HasDo = has_do<Functor, Interval>::value>
        struct find_interval_impl {
            using type = typename find_interval_impl<Functor, FromIndex, typename ToIndex::next>::type;
        };

        template <class Functor, class FromIndex, class ToIndex, class Interval>
        struct find_interval_impl<Functor, FromIndex, ToIndex, Interval, true> {
            using type = Interval;
        };

        template <class Functor, class Index, class FromIndex = typename find_from_index<Functor, Index>::type>
        struct find_interval {
            GRIDTOOLS_STATIC_ASSERT(FromIndex::value <= Index::value, GT_INTERNAL_ERROR);
            using to_index_t = typename find_to_index<Functor, FromIndex>::type;
            GRIDTOOLS_STATIC_ASSERT(FromIndex::value <= to_index_t::value, GT_INTERNAL_ERROR);
            using type = conditional_t<(to_index_t::value < Index::value),
                void,
                GT_META_CALL(make_interval, (FromIndex, to_index_t))>;
        };

        template <class Functor, class Index>
        struct find_interval<Functor, Index, void> {
            using type = void;
        };

        template <class Functor, class Index>
        struct is_interval_overload_defined
            : bool_constant<!std::is_void<typename find_interval<Functor, Index>::type>::value> {};

        template <class Functor, class Interval>
        struct bound_functor {
            using arg_list = typename Functor::arg_list;

            template <class Eval>
            static GT_FUNCTION auto Do(Eval &eval) GT_AUTO_RETURN(Functor::Do(eval, Interval{}));
        };
    } // namespace _impl

    GT_META_LAZY_NAMESPASE {
        template <class Functor, class Index, class = void>
        struct bind_functor_with_interval {
            GRIDTOOLS_STATIC_ASSERT(is_level_index<Index>::value, GT_INTERNAL_ERROR);
            using type = void;
        };

        template <class Index>
        struct bind_functor_with_interval<void, Index, void> {
            GRIDTOOLS_STATIC_ASSERT(is_level_index<Index>::value, GT_INTERNAL_ERROR);
            using type = void;
        };

        template <class Functor, class Index>
        struct bind_functor_with_interval<Functor,
            Index,
            enable_if_t<_impl::is_interval_overload_defined<Functor, Index>::value>> {
            GRIDTOOLS_STATIC_ASSERT(is_level_index<Index>::value, GT_INTERNAL_ERROR);
            using type = _impl::bound_functor<Functor, typename _impl::find_interval<Functor, Index>::type>;
        };

        template <class Functor, class Index>
        struct bind_functor_with_interval<Functor,
            Index,
            enable_if_t<!_impl::is_interval_overload_defined<Functor, Index>::value && has_do<Functor>::value>> {
            GRIDTOOLS_STATIC_ASSERT(is_level_index<Index>::value, GT_INTERNAL_ERROR);
            using type = Functor;
        };
    }
    GT_META_DELEGATE_TO_LAZY(bind_functor_with_interval, (class Functor, class Index), (Functor, Index));
} // namespace gridtools
