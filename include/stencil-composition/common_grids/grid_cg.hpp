/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
