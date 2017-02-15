/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "../common/halo_descriptor.hpp"
#include "../common/array.hpp"
#include "../storage/partitioner.hpp"
#include "interval.hpp"

namespace gridtools {

    // TODO should be removed once we removed all ctor(array) calls
    namespace enumtype_axis {
        enum coordinate_argument { minus, plus, begin, end, length };
    } // namespace enumtype_axis

    using namespace enumtype_axis;

    template < typename AxisInterval, typename Partitioner = partitioner_dummy >
    struct grid_base {
        GRIDTOOLS_STATIC_ASSERT((is_interval< AxisInterval >::value), "Internal Error: wrong type");
        typedef AxisInterval axis_type;
        typedef Partitioner partitioner_t;

        typedef typename boost::mpl::plus<
            boost::mpl::minus< typename AxisInterval::ToLevel::Splitter, typename AxisInterval::FromLevel::Splitter >,
            static_int< 1 > >::type size_type;

        array< uint_t, size_type::value > value_list;

        GT_FUNCTION grid_base(const grid_base< AxisInterval, Partitioner > &other)
            : m_partitioner(other.m_partitioner), m_direction_i(other.m_direction_i),
              m_direction_j(other.m_direction_j) {
            value_list = other.value_list;
        }

        GT_FUNCTION
        explicit grid_base(halo_descriptor const &direction_i, halo_descriptor const &direction_j)
            :
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdangling-field"
#endif
              m_partitioner(partitioner_dummy())
#ifdef __clang__
#pragma clang diagnostic pop
#endif
              ,
              m_direction_i(direction_i), m_direction_j(direction_j) {
            GRIDTOOLS_STATIC_ASSERT(is_partitioner_dummy< partitioner_t >::value,
                "you have to construct the grid with a valid partitioner, or with no partitioner at all.");
        }

        template < typename ParallelStorage >
        GT_FUNCTION explicit grid_base(const Partitioner &part_, ParallelStorage const &storage_)
            : m_partitioner(part_), m_direction_i(storage_.template get_halo_descriptor< 0 >()) // copy
              ,
              m_direction_j(storage_.template get_halo_descriptor< 1 >()) // copy
        {
            GRIDTOOLS_STATIC_ASSERT(!is_partitioner_dummy< Partitioner >::value,
                "you have to add the partitioner to the grid template parameters");
        }

        GT_FUNCTION
        explicit grid_base(uint_t *i, uint_t *j /*, uint_t* k*/)
            :
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdangling-field"
#endif
              m_partitioner(partitioner_dummy())
#ifdef __clang__
#pragma clang diagnostic pop
#endif
              ,
              m_direction_i(i[minus], i[plus], i[begin], i[end], i[length]),
              m_direction_j(j[minus], j[plus], j[begin], j[end], j[length]) {
            GRIDTOOLS_STATIC_ASSERT(is_partitioner_dummy< partitioner_t >::value,
                "you have to construct the grid with a valid partitioner, or with no partitioner at all.");
        }

        GT_FUNCTION
        uint_t i_low_bound() const { return m_direction_i.begin(); }

        GT_FUNCTION
        uint_t i_high_bound() const { return m_direction_i.end(); }

        GT_FUNCTION
        uint_t j_low_bound() const { return m_direction_j.begin(); }

        GT_FUNCTION
        uint_t j_high_bound() const { return m_direction_j.end(); }

        template < typename Level >
        GT_FUNCTION uint_t value_at() const {
            GRIDTOOLS_STATIC_ASSERT((is_level< Level >::value), "Internal Error: wrong type");
            int_t offs = Level::Offset::value;
            if (offs < 0)
                offs += 1;
            return value_list[Level::Splitter::value] + offs;
        }

        GT_FUNCTION uint_t k_min() const { return value_at< typename AxisInterval::FromLevel >(); }

        GT_FUNCTION uint_t k_max() const {
            // -1 because the axis has to be one level bigger than the largest k interval
            return value_at< typename AxisInterval::ToLevel >() - 1;
        }

        /**
         * The total length of the k dimension as defined by the axis.
         */
        GT_FUNCTION
        uint_t k_total_length() const {
            const uint_t begin_of_k = value_at< typename AxisInterval::FromLevel >();
            const uint_t end_of_k = value_at< typename AxisInterval::ToLevel >() - 1;
            return k_max() - k_min() + 1;
        }

        halo_descriptor const &direction_i() const { return m_direction_i; }

        halo_descriptor const &direction_j() const { return m_direction_j; }

        const Partitioner &partitioner() const {
            // the partitioner must be set
            return m_partitioner;
        }

        template < typename Flag >
        bool at_boundary(ushort_t const &coordinate_, Flag const &flag_) const {
            return m_partitioner.at_boundary(coordinate_, flag_);
        }

      private:
        Partitioner const &m_partitioner;
        halo_descriptor m_direction_i;
        halo_descriptor m_direction_j;
    };

} // namespace gridtools
