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
#include <type_traits>
#include <utility>

#include "../common/hymap.hpp"
#include "../common/integral_constant.hpp"
#include "../meta.hpp"
#include "cartesian.hpp"
#include "connectivity.hpp"
#include "unstructured.hpp"

namespace gridtools::fn {
    namespace domain_extender_impl_ {
        template <class...>
        struct get_offset;
        template <class Dim>
        struct get_offset<Dim, meta::val<>> : std::integral_constant<int, 0> {};
        template <class Dim, auto D, auto V, auto... Vs>
        struct get_offset<Dim, meta::val<D, V, Vs...>> : get_offset<Dim, meta::val<Vs...>> {};
        template <class Dim, Dim D, auto V, auto... Vs>
        struct get_offset<Dim, meta::val<D, V, Vs...>>
            : std::integral_constant<int, V + get_offset<Dim, meta::val<Vs...>>::value> {};

        template <class Dim, class OffsetsList>
        using get_offsets =
            meta::transform<meta::curry<meta::force<get_offset>::apply, Dim>::template apply, OffsetsList>;

        template <template <class...> class L, class... Ts>
        constexpr auto get_min(L<Ts...>) {
            if constexpr (sizeof...(Ts))
                return std::min({Ts::value...});
            else
                return 0;
        }

        template <template <class...> class L, class... Ts>
        constexpr auto get_max(L<Ts...>) {
            if constexpr (sizeof...(Ts))
                return std::max({Ts::value...});
            else
                return 0;
        }

        template <class OffsetsList>
        struct extend_size_f {
            template <class Dim, class Val>
            auto operator()(Val val) const {
                using offsets_t = get_offsets<Dim, OffsetsList>;
                using min_t = integral_constant<int, get_min(offsets_t())>;
                using max_t = integral_constant<int, get_max(offsets_t())>;
                return val + max_t() - min_t();
            }
        };

        template <class OffsetsList>
        struct shift_offsets_f {
            template <class Dim, class Val>
            auto operator()(Val val) const {
                using offsets_t = get_offsets<Dim, OffsetsList>;
                using min_t = integral_constant<int, get_min(offsets_t())>;
                return val + min_t();
            }
        };

        template <class OffsetsList, class Sizes, class Offsets>
        constexpr auto extend(OffsetsList, cartesian<Sizes, Offsets> const &domain) {
            return cartesian(hymap::transform(extend_size_f<OffsetsList>(), domain.sizes),
                hymap::transform(shift_offsets_f<OffsetsList>(), domain.offsets));
        }

        template <class Range>
        Range hor_extend(meta::val<>, Range range) {
            return range;
        }

        template <auto Conn, auto Off, auto... Vs, class Range>
        Range hor_extend(meta::val<Conn, Off, Vs...>, Range range) {
            if constexpr (Connectivity<decltype(Conn)>)
                return hor_extend(meta::constant<Vs...>, get_output_range(Conn, meta::list<meta::val<Off>>(), range));
            else
                return hor_extend(meta::constant<Vs...>, range);
        }

        template <class Range>
        constexpr bool is_empty_range(Range const &range) {
            return tuple_util::get<0>(range) == tuple_util::get<1>(range);
        }

        constexpr auto enclosing_range = []<class Range>(Range const &lhs, Range const &rhs) {
            return is_empty_range(lhs)
                       ? rhs
                       : is_empty_range(rhs) ? lhs
                                             : Range{std::min(tuple_util::get<0>(lhs), tuple_util::get<0>(rhs)),
                                                   std::max(tuple_util::get<1>(lhs), tuple_util::get<1>(rhs))};
        };

        template <class OffsetsList, class Horizontal, class Domain>
        constexpr auto hor_output_range(Domain const &domain) {
            int from = at_key_with_default<Horizontal, int>(domain.offsets);
            int to = from + at_key<Horizontal>(domain.sizes);
            return tuple_util::fold(enclosing_range,
                tuple_util::transform(
                    [&](auto offsets) {
                        return hor_extend(offsets, std::array{from, to});
                    },
                    meta::rename<std::tuple, OffsetsList>()));
        }

        template <class Horizontal>
        struct change_horizontal_f {
            int m_val;
            template <class Dim, class Val>
            auto operator()(Val val) const {
                if constexpr (std::is_same_v<Dim, Horizontal>)
                    return m_val;
                else
                    return val;
            }
        };

        template <class OffsetsList, class Sizes, class Offsets, class Horizontal>
        constexpr auto extend(OffsetsList, unstructured<Sizes, Offsets, Horizontal> const &domain) {
            static_assert(has_key<Sizes, Horizontal>());
            auto cart_dom = extend(OffsetsList(), cartesian(domain.sizes, domain.offsets));
            auto [hor_from, hor_to] = hor_output_range<OffsetsList, Horizontal>(domain);
            assert(hor_to > hor_from);
            return unstructured(
                hymap::transform(change_horizontal_f<Horizontal>{hor_to - hor_from}, std::move(cart_dom.sizes)),
                hymap::merge(
                    typename hymap::keys<Horizontal>::template values<int>(hor_from), std::move(cart_dom.offsets)),
                Horizontal());
        }

        template <class OffsetsList>
        constexpr auto domain_extender = []<class D>(D &&domain) -> decltype(auto) {
            return extend(OffsetsList(), std::forward<D>(domain));
        };
    } // namespace domain_extender_impl_
    using domain_extender_impl_::domain_extender;
} // namespace gridtools::fn