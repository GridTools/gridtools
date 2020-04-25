/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/type_traits.hpp"
#include "has_apply.hpp"
#include "interval.hpp"
#include "level.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace _impl {
                template <class Functor, class Index, bool HasApply = has_apply<Functor, index_to_level<Index>>::value>
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
                    bool HasApply = has_apply<Functor, make_interval<FromIndex, ToIndex>>::value>
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
                    class Interval = make_interval<FromIndex, ToIndex>,
                    bool HasApply = has_apply<Functor, Interval>::value>
                struct find_interval_impl {
                    using type = typename find_interval_impl<Functor, FromIndex, typename ToIndex::next>::type;
                };

                template <class Functor, class FromIndex, class ToIndex, class Interval>
                struct find_interval_impl<Functor, FromIndex, ToIndex, Interval, true> {
                    using type = Interval;
                };

                template <class Functor, class Index, class FromIndex = typename find_from_index<Functor, Index>::type>
                struct find_interval {
                    static_assert(FromIndex::value <= Index::value, GT_INTERNAL_ERROR);
                    using to_index_t = typename find_to_index<Functor, FromIndex>::type;
                    static_assert(FromIndex::value <= to_index_t::value, GT_INTERNAL_ERROR);
                    using type = std::
                        conditional_t<(to_index_t::value < Index::value), void, make_interval<FromIndex, to_index_t>>;
                };

                template <class Functor, class Index>
                struct find_interval<Functor, Index, void> {
                    using type = void;
                };

                template <class Functor, class Index>
                struct is_interval_overload_defined
                    : bool_constant<!std::is_void<typename find_interval<Functor, Index>::type>::value> {};

                template <class Functor, class Interval>
                struct bound_functor : Functor {
                    template <class Eval>
                    static GT_FUNCTION void apply(Eval &eval) {
                        Functor::template apply<Eval &>(eval, Interval{});
                    }
                };
            } // namespace _impl

            namespace lazy {
                template <class Functor, class Index, class = void>
                struct bind_functor_with_interval {
                    static_assert(is_level_index<Index>::value, GT_INTERNAL_ERROR);
                    using type = void;
                };

                template <class Index>
                struct bind_functor_with_interval<void, Index, void> {
                    static_assert(is_level_index<Index>::value, GT_INTERNAL_ERROR);
                    using type = void;
                };

                template <class Functor, class Index>
                struct bind_functor_with_interval<Functor,
                    Index,
                    std::enable_if_t<_impl::is_interval_overload_defined<Functor, Index>::value>> {
                    static_assert(is_level_index<Index>::value, GT_INTERNAL_ERROR);
                    using type = _impl::bound_functor<Functor, typename _impl::find_interval<Functor, Index>::type>;
                };

                template <class Functor, class Index>
                struct bind_functor_with_interval<Functor,
                    Index,
                    std::enable_if_t<!_impl::is_interval_overload_defined<Functor, Index>::value &&
                                     has_apply<Functor>::value>> {
                    static_assert(is_level_index<Index>::value, GT_INTERNAL_ERROR);
                    using type = Functor;
                };
            } // namespace lazy
            /**
             *   Takes an elementary functor (Functor) and the level index (Index) as an input; deduces the interval
             * that should be used for the range from Index to Index::next and produces the functor where the deduced
             * interval is bound. I.e. the new functor has the apply method with a single argument and delegates to the
             * original one.
             *
             *   Corner cases:
             *     - if `void` is passed as a Functor the return will be also `void`
             *     - if there is no overload that includes that level and there is no overload with a single argument,
             * `void` is returned.
             *     - if there is no overload, but there is an overload with a single argument, original Functor is
             * returned.
             *
             *   The algorithm for deducing the interval is the following:
             *     - from the given Index we go left util we have found the level that matches as a lower bound. (Here
             * we use the fact that interval is convertible to its lower bound level). if we reach zero index, we fail.
             *     - next we search for the upper bound by going right from the lower bound that we just found.
             *     - if the interval contains the Index, we are done else we fail.
             *
             *   Note that the algorithm doesn't have assertions that there is no holes between intervals and there is
             * no overlaps. Moreover, it produces deterministic results even for insane setups.
             *
             *   TODO(anstaf): If we want to keep the strict requirements on interval overloads that we have now (no
             * holes, no overlaps), we need to add an additional predicate `is_valid_functor<Functor, AxisInterval>`
             * just for the user protection. The alternative is to clearly define overlap resolution rules for the user.
             */
            GT_META_DELEGATE_TO_LAZY(bind_functor_with_interval, (class Functor, class Index), (Functor, Index));
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
