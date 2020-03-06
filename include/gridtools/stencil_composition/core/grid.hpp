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

#include <cassert>
#include <initializer_list>
#include <iterator>
#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../common/dim.hpp"
#include "../common/extent.hpp"
#include "execution_types.hpp"
#include "interval.hpp"
#include "level.hpp"

namespace gridtools {
    namespace core {
        template <class Interval>
        class grid {
            static_assert(is_interval<Interval>::value, GT_INTERNAL_ERROR);
            static_assert(Interval::FromLevel::splitter == 0, GT_INTERNAL_ERROR);

            static constexpr int_t offset_limit = Interval::FromLevel::offset_limit;
            static constexpr int_t start_offset = Interval::FromLevel::offset;

            int_t m_i_start;
            int_t m_i_size;
            int_t m_j_start;
            int_t m_j_size;
            int_t m_k_values[Interval::ToLevel::splitter];

            template <uint_t Splitter = 0, int_t Offset = start_offset>
            static integral_constant<int_t, (Offset > 0) ? Offset - 1 : Offset> offset(
                level<Splitter, Offset, offset_limit> = {}) {
                return {};
            }

            template <class Level, std::enable_if_t<Level::splitter != 0, int> = 0>
            int_t splitter_value(Level) const {
                return m_k_values[Level::splitter - 1];
            }

            template <int_t Offset>
            static integral_constant<int_t, 0> splitter_value(level<0, Offset, offset_limit>) {
                return {};
            }

            template <class T, std::enable_if_t<T::FromLevel::splitter != T::ToLevel::splitter, int> = 0>
            auto splitter_size(T) const {
                return splitter_value(typename T::ToLevel()) - splitter_value(typename T::FromLevel());
            }

            template <uint_t Splitter, int_t FromOffset, int_t ToOffset>
            static integral_constant<int_t, 0> splitter_size(
                interval<level<Splitter, FromOffset, offset_limit>, level<Splitter, ToOffset, offset_limit>>) {
                return {};
            }

            template <class Level>
            auto value_at(Level x) const {
                return splitter_value(x) + offset(x) - offset();
            }

          public:
            using interval_t = Interval;

            template <class Intervals = std::initializer_list<int_t>>
            grid(int_t i_start, int_t i_size, int_t j_start, int_t j_size, Intervals const &intervals)
                : m_i_start(i_start), m_i_size(i_size), m_j_start(j_start), m_j_size(j_size) {
                int_t acc = 0;
                auto src = std::begin(intervals);
                for (auto &dst : m_k_values) {
                    assert(src != std::end(intervals));
                    dst = acc + *(src++);
                    acc = dst;
                }
            }

            auto origin() const {
                return tuple_util::make<hymap::keys<dim::i, dim::j, dim::k>::values>(m_i_start, m_j_start, offset());
            }

            template <class Extent = extent<>>
            int_t i_size(Extent extent = {}) const {
                return extent.extend(dim::i(), m_i_size);
            }

            template <class Extent = extent<>>
            int_t j_size(Extent extent = {}) const {
                return extent.extend(dim::j(), m_j_size);
            }

            template <class From = typename Interval::FromLevel,
                class To = typename Interval::ToLevel,
                class Execution = forward>
            auto k_start(interval<From, To> = {}, Execution = {}) const {
                return value_at(From());
            }

            template <class From, class To>
            auto k_start(interval<From, To>, backward) const {
                return value_at(To());
            }

            template <class From = typename Interval::FromLevel,
                class To = typename Interval::ToLevel,
                class Extent = extent<>>
            auto k_size(interval<From, To> x = {}, Extent extent = {}) const {
                using namespace literals;
                return extent.extend(dim::k(), 1_c + splitter_size(x) + offset(To()) - offset(From()));
            }

            auto size() const {
                return tuple_util::make<hymap::keys<dim::i, dim::j, dim::k>::values>(i_size(), j_size(), k_size());
            }
        };

        template <class T>
        using is_grid = meta::is_instantiation_of<grid, T>;
    } // namespace core
} // namespace gridtools
