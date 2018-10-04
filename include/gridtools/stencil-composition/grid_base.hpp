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
#include "../common/array.hpp"
#include "../common/halo_descriptor.hpp"
#include "axis.hpp"
#include "interval.hpp"

namespace gridtools {
    namespace _impl {
        /*
         * @brief convert an array of intervals in an array of indices of splitters
         */
        template <size_t NIntervals>
        array<uint_t, NIntervals + 1> intervals_to_indices(const array<uint_t, NIntervals> &intervals) {
            array<uint_t, NIntervals + 1> indices;
            indices[0] = 0;
            indices[1] = intervals[0];
            for (size_t i = 2; i < NIntervals + 1; ++i) {
                indices[i] = indices[i - 1] + intervals[i - 1];
            }
            return indices;
        }
    } // namespace _impl

    // TODO should be removed once we removed all ctor(array) calls
    namespace enumtype_axis {
        enum coordinate_argument { minus, plus, begin, end, length };
    } // namespace enumtype_axis

    using namespace enumtype_axis;

    template <typename Axis>
    struct grid_base {
        GRIDTOOLS_STATIC_ASSERT((is_interval<Axis>::value), GT_INTERNAL_ERROR);
        typedef Axis axis_type;

        static constexpr int_t size = Axis::ToLevel::splitter - Axis::FromLevel::splitter + 1;

        array<uint_t, size> value_list;

      private:
        halo_descriptor m_direction_i;
        halo_descriptor m_direction_j;

      public:
        GT_DEPRECATED("This constructor does not initialize the vertical axis, use the constructor with 3 "
                      "arguments. (deprecated after 1.05.02)")
        GT_FUNCTION explicit grid_base(halo_descriptor const &direction_i, halo_descriptor const &direction_j)
            : m_direction_i(direction_i), m_direction_j(direction_j) {}

        /**
         * @brief standard ctor
         * @param direction_i halo_descriptor in i direction
         * @param direction_j halo_descriptor in j direction
         * @param value_list gridtools::array with splitter positions
         */
        GT_FUNCTION
        explicit grid_base(halo_descriptor const &direction_i,
            halo_descriptor const &direction_j,
            const array<uint_t, size> &value_list)
            : m_direction_i(direction_i), m_direction_j(direction_j), value_list(value_list) {}

        GT_DEPRECATED("Use constructor with halo_descriptors (deprecated after 1.05.02)")
        GT_FUNCTION explicit grid_base(uint_t *i, uint_t *j /*, uint_t* k*/)
            : m_direction_i(i[minus], i[plus], i[begin], i[end], i[length]),
              m_direction_j(j[minus], j[plus], j[begin], j[end], j[length]) {}

        GT_FUNCTION
        uint_t i_low_bound() const { return m_direction_i.begin(); }

        GT_FUNCTION
        uint_t i_high_bound() const { return m_direction_i.end(); }

        GT_FUNCTION
        uint_t j_low_bound() const { return m_direction_j.begin(); }

        GT_FUNCTION
        uint_t j_high_bound() const { return m_direction_j.end(); }

        template <class Level, int_t Offset = Level::offset>
        GT_FUNCTION enable_if_t<(Offset > 0), uint_t> value_at() const {
            GRIDTOOLS_STATIC_ASSERT((is_level<Level>::value), GT_INTERNAL_ERROR);
            return value_list[Level::splitter] + Offset - 1;
        }

        template <class Level, int_t Offset = Level::offset>
        GT_FUNCTION enable_if_t<(Offset <= 0), uint_t> value_at() const {
            GRIDTOOLS_STATIC_ASSERT((is_level<Level>::value), GT_INTERNAL_ERROR);
            return value_list[Level::splitter] - static_cast<uint_t>(-Offset);
        }

        GT_FUNCTION uint_t k_min() const { return value_at<typename Axis::FromLevel>(); }

        GT_FUNCTION uint_t k_max() const {
            // -1 because the axis has to be one level bigger than the largest k interval
            return value_at<typename Axis::ToLevel>() - 1;
        }

        /**
         * The total length of the k dimension as defined by the axis.
         */
        GT_FUNCTION
        uint_t k_total_length() const { return k_max() - k_min() + 1; }

        GT_FUNCTION halo_descriptor const &direction_i() const { return m_direction_i; }

        GT_FUNCTION halo_descriptor const &direction_j() const { return m_direction_j; }
    };

} // namespace gridtools
