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

#include "../common/defs.hpp"
#include "../common/halo_descriptor.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "axis.hpp"
#include "dim.hpp"
#include "execution_types.hpp"
#include "extent.hpp"
#include "interval.hpp"
#include "level.hpp"

namespace gridtools {
    template <class Interval>
    class grid {
        GT_STATIC_ASSERT(is_interval<Interval>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(Interval::FromLevel::splitter == 0, GT_INTERNAL_ERROR);

        static constexpr int_t offset_limit = Interval::FromLevel::offset_limit;
        static constexpr uint_t start_offset = Interval::FromLevel::offset;

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
        grid(halo_descriptor const &direction_i, halo_descriptor const &direction_j, Intervals const &intervals)
            : m_i_start((int_t)direction_i.begin()),
              m_i_size((int_t)direction_i.end() + 1 - (int_t)direction_i.begin()),
              m_j_start((int_t)direction_j.begin()),
              m_j_size((int_t)direction_j.end() + 1 - (int_t)direction_j.begin()) {
            int_t acc = 0;
            auto src = std::begin(intervals);
            for (auto &dst : m_k_values) {
                assert(src != std::end(intervals));
                dst = acc + *(src++);
                acc = dst;
            }
        }

        int_t i_start() const { return m_i_start; }

        int_t j_start() const { return m_j_start; }

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
            class Execution = execute::forward>
        auto k_start(interval<From, To> = {}, Execution = {}) const {
            return value_at(From());
        }

        template <class From, class To>
        auto k_start(interval<From, To>, execute::backward) const {
            return value_at(To());
        }

        template <class From = typename Interval::FromLevel, class To = typename Interval::ToLevel>
        auto k_size(interval<From, To> x = {}) const {
            using namespace literals;
            return 1_c + splitter_size(x) + offset(To()) - offset(From());
        }
    };

    template <class T>
    using is_grid = meta::is_instantiation_of<grid, T>;

    template <class Axis>
    grid<typename Axis::axis_interval_t> make_grid(
        halo_descriptor const &direction_i, halo_descriptor const &direction_j, Axis const &axis) {
        return {direction_i, direction_j, axis.interval_sizes()};
    }
    inline grid<axis<1>::axis_interval_t> make_grid(uint_t di, uint_t dj, int_t dk) {
        return make_grid(halo_descriptor(di), halo_descriptor(dj), axis<1>(dk));
    }
    template <class Axis>
    grid<typename Axis::axis_interval_t> make_grid(uint_t di, uint_t dj, Axis const &axis) {
        return {halo_descriptor(di), halo_descriptor(dj), axis.interval_sizes()};
    }
    inline grid<axis<1>::axis_interval_t> make_grid(
        halo_descriptor const &direction_i, halo_descriptor const &direction_j, int_t dk) {
        return make_grid(direction_i, direction_j, axis<1>(dk));
    }
} // namespace gridtools
