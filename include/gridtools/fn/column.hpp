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

#include <bits/ranges_base.h>
#include <ranges>

#include "../sid/concept.hpp"

namespace gridtools::fn {
    namespace column_impl_ {
        template <class Ptr, class Stride, class PtrDiff>
        struct iter {
            Ptr m_ptr;
            Stride m_stride;

            using difference_type = PtrDiff;
            using value_type = typename std::indirectly_readable_traits<Ptr>::value_type;

            friend constexpr auto operator<=>(iter const &, iter const &) = default;

            //            constexpr friend auto operator-(iter const &l, iter const &r) -> decltype(
            //                (std::declval<Ptr const &>() - std::declval<Ptr const &>()) / std::declval<Stride const
            //                &>()) { return (l.m_ptr - r.m_ptr) / l.m_stride;
            //            }
            //
            //            constexpr friend iter operator+(iter i, difference_type n) { return i += n; }
            //            constexpr friend iter operator+(difference_type n, iter i) { return i += n; }
            //            constexpr friend iter operator-(iter i, difference_type n) { return i -= n; }
            //
            //            decltype(auto) operator[](auto n) const { return *sid::shifted(m_ptr, m_stride, n); }
            //
            //            constexpr iter &operator+=(auto n) {
            //                sid::shift(m_ptr, m_stride, n);
            //                return *this;
            //            }
            //
            //            constexpr iter &operator-=(auto n) {
            //                sid::shift(m_ptr, m_stride, -n);
            //                return *this;
            //            }

            constexpr iter &operator++() {
                sid::shift(m_ptr, m_stride, integral_constant<int, 1>());
                return *this;
            }
            constexpr iter operator++(int) {
                iter res = *this;
                ++*this;
                return res;
            }
            //            constexpr iter &operator--() {
            //                sid::shift(m_ptr, m_stride, integral_constant<int, -1>());
            //                return *this;
            //            }
            //            constexpr iter operator--(int) {
            //                iter res = *this;
            //                --*this;
            //                return res;
            //            }
            decltype(auto) operator*() const { return *m_ptr; }
        };

        using probe_t = iter<double *, std::ptrdiff_t, std::ptrdiff_t>;

        static_assert(std::input_or_output_iterator<probe_t>);
        static_assert(std::input_iterator<probe_t>);
        //        static_assert(std::output_iterator<probe_t, double>);
        //        static_assert(std::sentinel_for<probe_t, probe_t>);
        //        static_assert(std::forward_iterator<probe_t>);
        //        static_assert(std::bidirectional_iterator<probe_t>);
        //        static_assert(std::totally_ordered<probe_t>);
        //        static_assert(std::sized_sentinel_for<probe_t, probe_t>);
        //        static_assert(std::derived_from<std::__detail::__iter_concept<probe_t>,
        //        std::random_access_iterator_tag>); static_assert(std::random_access_iterator<probe_t>);

        template <class Ptr, class Stride>
        struct column {
            using difference_type = std::ptrdiff_t;
            using value_type = typename std::indirectly_readable_traits<Ptr>::value_type;

            Stride m_stride;
            Ptr m_ptr;

            constexpr friend bool operator==(column l, column r) { return l.m_ptr == r.m_ptr; }

            constexpr friend bool operator!=(column l, column r) { return !(l == r); }

            constexpr column &operator++() {
                sid::shift(m_ptr, m_stride, integral_constant<int, 1>());
                return *this;
            }
            constexpr void operator++(int) { *this ++; }
            decltype(auto) operator*() const { return *m_ptr; }

            constexpr column begin() && { return std::move(*this); }
            constexpr column &begin() & { return *this; }
            constexpr column begin() const & { return *this; }
            constexpr std::unreachable_sentinel_t end() const { return {}; }
        };

        using simple_column_t = column<double *, int>;
        using diff = std::iter_difference_t<simple_column_t>;
        using ddd = std::iter_value_t<simple_column_t>;

        //        static_assert(std::indirectly_readable<simple_column_t>);
        static_assert(std::ranges::input_range<simple_column_t>);
    } // namespace column_impl_
} // namespace gridtools::fn
