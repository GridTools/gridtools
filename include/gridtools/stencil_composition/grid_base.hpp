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

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/halo_descriptor.hpp"
#include "../common/integral_constant.hpp"
#include "axis.hpp"
#include "interval.hpp"

namespace gridtools {
    namespace _impl {
        /*
         * @brief convert an array of intervals in an array of indices of splitters
         */
        template <size_t NIntervals>
        array<int_t, NIntervals + 1> intervals_to_indices(const array<int_t, NIntervals> &intervals) {
            array<int_t, NIntervals + 1> indices;
            indices[0] = 0;
            indices[1] = intervals[0];
            for (size_t i = 2; i < NIntervals + 1; ++i) {
                indices[i] = indices[i - 1] + intervals[i - 1];
            }
            return indices;
        }
    } // namespace _impl

    namespace grid_base_impl_ {
        constexpr int_t real_offset(int_t val) { return val > 0 ? val - 1 : val; }
    } // namespace grid_base_impl_

    template <typename Axis>
    struct grid_base {
        GT_STATIC_ASSERT(is_interval<Axis>::value, GT_INTERNAL_ERROR);
        typedef Axis axis_type;

        static constexpr int_t size = Axis::ToLevel::splitter - Axis::FromLevel::splitter + 1;

        array<int_t, size> value_list;

      private:
        halo_descriptor m_direction_i;
        halo_descriptor m_direction_j;

      public:
        /**
         * @brief standard ctor
         * @param direction_i halo_descriptor in i direction
         * @param direction_j halo_descriptor in j direction
         * @param value_list gridtools::array with splitter positions
         */
        GT_FUNCTION
        explicit grid_base(halo_descriptor const &direction_i,
            halo_descriptor const &direction_j,
            const array<int_t, size> &value_list)
            : value_list(value_list), m_direction_i(direction_i), m_direction_j(direction_j) {}

        GT_FUNCTION int_t i_low_bound() const { return m_direction_i.begin(); }

        GT_FUNCTION int_t i_high_bound() const { return m_direction_i.end(); }

        GT_FUNCTION int_t j_low_bound() const { return m_direction_j.begin(); }

        GT_FUNCTION int_t j_high_bound() const { return m_direction_j.end(); }

        template <class Level, int_t Offset = grid_base_impl_::real_offset(Level::offset)>
        GT_FUNCTION int_t value_at() const {
            GT_STATIC_ASSERT(is_level<Level>::value, GT_INTERNAL_ERROR);
            return value_list[Level::splitter] + Offset;
        }

        template <uint_t FromSplitter,
            int_t FromOffset,
            uint_t ToSplitter,
            int_t ToOffset,
            int_t OffsetLimit,
            int_t Extra = grid_base_impl_::real_offset(ToOffset) - grid_base_impl_::real_offset(FromOffset),
            enable_if_t<(FromSplitter < ToSplitter), int> = 0>
        GT_FUNCTION int_t count(
            level<FromSplitter, FromOffset, OffsetLimit>, level<ToSplitter, ToOffset, OffsetLimit>) const {
            return value_list[ToSplitter] - value_list[FromSplitter] + Extra + 1;
        }

        template <uint_t FromSplitter,
            int_t FromOffset,
            uint_t ToSplitter,
            int_t ToOffset,
            int_t OffsetLimit,
            int_t Extra = grid_base_impl_::real_offset(FromOffset) - grid_base_impl_::real_offset(ToOffset),
            enable_if_t<(FromSplitter >= ToSplitter), int> = 0>
        GT_FUNCTION int_t count(
            level<FromSplitter, FromOffset, OffsetLimit>, level<ToSplitter, ToOffset, OffsetLimit>) const {
            return value_list[FromSplitter] - value_list[ToSplitter] + Extra + 1;
        }

        template <uint_t Splitter,
            int_t FromOffset,
            int_t ToOffset,
            int_t OffsetLimit,
            int_t Delta = grid_base_impl_::real_offset(ToOffset) - grid_base_impl_::real_offset(FromOffset),
            int_t Val = (Delta > 0) ? 1 + Delta : 1 - Delta>
        GT_FUNCTION integral_constant<int_t, Val> count(
            level<Splitter, FromOffset, OffsetLimit>, level<Splitter, ToOffset, OffsetLimit>) const {
            return {};
        }

        GT_FUNCTION int_t k_min() const { return value_at<typename Axis::FromLevel>(); }

        GT_FUNCTION int_t k_max() const {
            // -1 because the axis has to be one level bigger than the largest k interval
            return value_at<typename Axis::ToLevel>() - 1;
        }

        /**
         * The total length of the k dimension as defined by the axis.
         */
        GT_FUNCTION int_t k_total_length() const { return k_max() - k_min() + 1; }

        GT_FUNCTION halo_descriptor const &direction_i() const { return m_direction_i; }

        GT_FUNCTION halo_descriptor const &direction_j() const { return m_direction_j; }
    };

} // namespace gridtools
