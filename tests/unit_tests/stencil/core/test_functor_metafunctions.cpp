/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil/core/functor_metafunctions.hpp>
#include <gridtools/stencil/core/interval.hpp>
#include <gridtools/stencil/core/level.hpp>

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace {
                using interval_t = interval<level<0, 1, 3>, level<2, -1, 3>>;

                struct empty {};
                static_assert(!is_valid_functor<empty, interval_t>::value, "");

                struct simple {
                    template <class T>
                    static void apply(T);
                };
                static_assert(is_valid_functor<simple, interval_t>::value, "");

                using expected_simple_map_t = meta::list<meta::list<interval<level<0, 1, 3>, level<0, 1, 3>>, simple>,
                    meta::list<interval<level<0, 2, 3>, level<0, 2, 3>>, simple>,
                    meta::list<interval<level<0, 3, 3>, level<0, 3, 3>>, simple>,
                    meta::list<interval<level<1, -3, 3>, level<1, -3, 3>>, simple>,
                    meta::list<interval<level<1, -2, 3>, level<1, -2, 3>>, simple>,
                    meta::list<interval<level<1, -1, 3>, level<1, -1, 3>>, simple>,
                    meta::list<interval<level<1, 1, 3>, level<1, 1, 3>>, simple>,
                    meta::list<interval<level<1, 2, 3>, level<1, 2, 3>>, simple>,
                    meta::list<interval<level<1, 3, 3>, level<1, 3, 3>>, simple>,
                    meta::list<interval<level<2, -3, 3>, level<2, -3, 3>>, simple>,
                    meta::list<interval<level<2, -2, 3>, level<2, -2, 3>>, simple>,
                    meta::list<interval<level<2, -1, 3>, level<2, -1, 3>>, simple>>;

                static_assert(std::is_same<make_functor_map<simple, interval_t>, expected_simple_map_t>::value, "");

                struct good {
                    template <class T>
                    static void apply(T);
                    template <class T>
                    static void apply(T, interval<level<0, -2, 3>, level<0, 1, 3>>);
                    template <class T>
                    static void apply(T, interval<level<0, 3, 3>, level<1, 1, 3>>);
                    template <class T>
                    static void apply(T, interval<level<1, 2, 3>, level<2, -3, 3>>);
                };
                static_assert(is_valid_functor<good, interval_t>::value, "");

                using expected_good_map_t = meta::list<
                    meta::list<interval<level<0, 1, 3>, level<0, 1, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, -2, 3>, level<0, 1, 3>>>>,
                    meta::list<interval<level<0, 2, 3>, level<0, 2, 3>>, good>,
                    meta::list<interval<level<0, 3, 3>, level<0, 3, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, -3, 3>, level<1, -3, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, -2, 3>, level<1, -2, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, -1, 3>, level<1, -1, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, 1, 3>, level<1, 1, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<0, 3, 3>, level<1, 1, 3>>>>,
                    meta::list<interval<level<1, 2, 3>, level<1, 2, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<1, 2, 3>, level<2, -3, 3>>>>,
                    meta::list<interval<level<1, 3, 3>, level<1, 3, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<1, 2, 3>, level<2, -3, 3>>>>,
                    meta::list<interval<level<2, -3, 3>, level<2, -3, 3>>,
                        functor_metafunctions_impl_::bound_functor<good, interval<level<1, 2, 3>, level<2, -3, 3>>>>,
                    meta::list<interval<level<2, -2, 3>, level<2, -2, 3>>, good>,
                    meta::list<interval<level<2, -1, 3>, level<2, -1, 3>>, good>>;

                static_assert(std::is_same<make_functor_map<good, interval_t>, expected_good_map_t>::value, "");

                struct intersect {
                    template <class T>
                    static void apply(T, interval<level<0, -2, 3>, level<1, 2, 3>>);
                    template <class T>
                    static void apply(T, interval<level<1, -2, 3>, level<2, -3, 3>>);
                };
                static_assert(!is_valid_functor<intersect, interval_t>::value, "");

                struct gaps {
                    template <class T>
                    static void apply(T, interval<level<0, 3, 3>, level<1, -3, 3>>);
                    template <class T>
                    static void apply(T, interval<level<1, 3, 3>, level<2, -3, 3>>);
                };
                static_assert(is_valid_functor<gaps, interval_t>::value, "");

                using expected_gaps_map_t = meta::list<meta::list<interval<level<0, 1, 3>, level<0, 1, 3>>>,
                    meta::list<interval<level<0, 2, 3>, level<0, 2, 3>>>,
                    meta::list<interval<level<0, 3, 3>, level<0, 3, 3>>,
                        functor_metafunctions_impl_::bound_functor<gaps, interval<level<0, 3, 3>, level<1, -3, 3>>>>,
                    meta::list<interval<level<1, -3, 3>, level<1, -3, 3>>,
                        functor_metafunctions_impl_::bound_functor<gaps, interval<level<0, 3, 3>, level<1, -3, 3>>>>,
                    meta::list<interval<level<1, -2, 3>, level<1, -2, 3>>>,
                    meta::list<interval<level<1, -1, 3>, level<1, -1, 3>>>,
                    meta::list<interval<level<1, 1, 3>, level<1, 1, 3>>>,
                    meta::list<interval<level<1, 2, 3>, level<1, 2, 3>>>,
                    meta::list<interval<level<1, 3, 3>, level<1, 3, 3>>,
                        functor_metafunctions_impl_::bound_functor<gaps, interval<level<1, 3, 3>, level<2, -3, 3>>>>,
                    meta::list<interval<level<2, -3, 3>, level<2, -3, 3>>,
                        functor_metafunctions_impl_::bound_functor<gaps, interval<level<1, 3, 3>, level<2, -3, 3>>>>,
                    meta::list<interval<level<2, -2, 3>, level<2, -2, 3>>>,
                    meta::list<interval<level<2, -1, 3>, level<2, -1, 3>>>>;

                static_assert(std::is_same<make_functor_map<gaps, interval_t>, expected_gaps_map_t>::value, "");
            } // namespace
        }     // namespace core
    }         // namespace stencil
} // namespace gridtools
