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

namespace gridtools {

    template < typename Axis >
    struct grid_cg {
        GRIDTOOLS_STATIC_ASSERT((is_interval< Axis >::value), "Internal Error: wrong type");
        typedef Axis axis_type;

        typedef typename boost::mpl::plus<
            boost::mpl::minus< typename Axis::ToLevel::Splitter, typename Axis::FromLevel::Splitter >,
            static_int< 1 > >::type size_type;

        array< uint_t, size_type::value > value_list;

        GT_FUNCTION
        explicit grid_cg(halo_descriptor const &direction_i, halo_descriptor const &direction_j)
            : m_direction_i(direction_i), m_direction_j(direction_j) {}

        GT_FUNCTION
        explicit grid_cg(uint_t *i, uint_t *j /*, uint_t* k*/)
            : m_direction_i(i[minus], i[plus], i[begin], i[end], i[length]),
              m_direction_j(j[minus], j[plus], j[begin], j[end], j[length]) {}

        GT_FUNCTION
        explicit grid_cg(array< uint_t, 5 > const &i, array< uint_t, 5 > const &j)
            : m_direction_i(i[minus], i[plus], i[begin], i[end], i[length]),
              m_direction_j(j[minus], j[plus], j[begin], j[end], j[length]) {}

        __device__ grid_cg(grid_cg< Axis > const &other)
            : m_direction_i(other.m_direction_i), m_direction_j(other.m_direction_j), value_list(other.value_list) {}

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

        GT_FUNCTION
        uint_t value_at_top() const {
            return value_list[size_type::value - 1];
            // return m_k_high_bound;
        }

        GT_FUNCTION
        uint_t value_at_bottom() const {
            return value_list[0];
            // return m_k_low_bound;
        }

        halo_descriptor const &direction_i() const { return m_direction_i; }

        halo_descriptor const &direction_j() const { return m_direction_j; }

      private:
        halo_descriptor m_direction_i;
        halo_descriptor m_direction_j;
    };

} // namespace gridtools
