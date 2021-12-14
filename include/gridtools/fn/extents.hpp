/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "../common/int_vector.hpp"
#include "../common/integral_constant.hpp"
#include "../common/utility.hpp"
#include "../meta.hpp"
#include "gridtools/meta/type_traits.hpp"

namespace gridtools {
    namespace fn {
        template <class Dim, ptrdiff_t L, ptrdiff_t U>
        struct extent {
            static_assert(L <= U);
            using dim_t = Dim;
            using lower_t = integral_constant<ptrdiff_t, L>;
            using upper_t = integral_constant<ptrdiff_t, U>;
            using size_t = integral_constant<std::size_t, U - L>;
            using list_t = meta::list<Dim, lower_t, upper_t>;
        };

        template <class>
        struct is_extent : std::false_type {};

        template <class Dim, ptrdiff_t L, ptrdiff_t U>
        struct is_extent<extent<Dim, L, U>> : bool_constant<L <= U> {};

#ifdef __cpp_concepts
        template <class T>
        concept extent_c = is_extent<T>::value;
#endif

        template <class... Ts>
        struct extents {
            static_assert(meta::all_of<is_extent, meta::list<Ts...>>::value);
            static_assert(meta::is_set<meta::list<typename Ts::dim_t...>>::value);
            using keys_t = hymap::keys<typename Ts::dim_t...>;
            static_assert(is_int_vector<typename keys_t::template values<typename Ts::lower_t...>>::value);
            static_assert(is_int_vector<typename keys_t::template values<typename Ts::upper_t...>>::value);
            static_assert(is_int_vector<typename keys_t::template values<typename Ts::size_t...>>::value);

            static GT_CONSTEVAL auto offsets() {
                return int_vector::prune_zeros(typename keys_t::template values<typename Ts::lower_t...>());
            }
            static GT_CONSTEVAL auto sizes() {
                return int_vector::prune_zeros(typename keys_t::template values<typename Ts::size_t...>());
            }
        };

        template <class>
        struct is_extents : std::false_type {};

        template <class... Ts>
        struct is_extents<extents<Ts...>> : bool_constant<meta::all_of<is_extent, meta::list<Ts...>>::value &&
                                                          meta::is_set<meta::list<typename Ts::dim_t...>>::value> {};

#ifdef __cpp_concepts
        template <class T>
        concept extents_c = meta::is_instantiation_of<extents, T>::value;
#endif

        template <class Extents, class Offsets>
        decltype(auto) GT_FUNCTION GT_CONSTEXPR extend_offsets(Offsets &&src) {
            static_assert(is_extents<Extents>::value);
            static_assert(is_int_vector<Offsets>::value);
            using namespace int_vector::arithmetic;
            return wstd::forward<Offsets>(src) + Extents::offsets();
        }

        template <class Extents, class Sizes>
        decltype(auto) GT_FUNCTION GT_CONSTEXPR extend_sizes(Sizes &&sizes) {
            static_assert(is_extents<Extents>::value);
            static_assert(is_int_vector<Sizes>::value);
            using namespace int_vector::arithmetic;
            return wstd::forward<Sizes>(sizes) + Extents::sizes();
        }

        namespace extent_impl_ {
            template <class...>
            struct merge_extents;

            template <class Dim, ptrdiff_t... L, ptrdiff_t... U>
            struct merge_extents<meta::list<Dim, extent<Dim, L, U>>...> {
                using type = meta::list<Dim, extent<Dim, std::min({L...}), std::max({U...})>>;
            };
        } // namespace extent_impl_

        // take any number of individual `extent`'s and produce the normalized `extents`
        // if some `extent`'s has the same dimension, they are merged
        template <class... Extents>
        using make_extents = meta::rename<extents,
            meta::transform<meta::second,
                meta::mp_make<meta::force<extent_impl_::merge_extents>::template apply,
                    meta::list<meta::list<typename Extents::dim_t, Extents>...>>>>;

        // merge several `extents` into a one
        template <class... Extentss>
        using enclosing_extents = meta::rename<make_extents, meta::concat<meta::rename<meta::list, Extentss>...>>;

    } // namespace fn
} // namespace gridtools
