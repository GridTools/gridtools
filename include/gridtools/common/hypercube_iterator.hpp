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

#include "array.hpp"
#include "array_addons.hpp"
#include "generic_metafunctions/gt_integer_sequence.hpp"
#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    namespace impl_ {

        template < size_t D >
        class hypercube_view {
          private:
            using point_t = array< size_t, D >;
            struct grid_iterator {
                point_t m_pos;
                const point_t &m_begin;
                const point_t &m_end;

                GT_FUNCTION grid_iterator &operator++() {
                    for (size_t i = 0; i < D; ++i) {
                        size_t index = D - i - 1;
                        if (m_pos[index] + 1 < m_end[index]) {
                            m_pos[index]++;
                            return *this;
                        } else {
                            m_pos[index] = m_begin[index];
                        }
                    }
                    // we reached the end
                    for (size_t i = 0; i < D; ++i)
                        m_pos[i] = m_end[i];
                    return *this;
                }

                GT_FUNCTION grid_iterator operator++(int) {
                    grid_iterator tmp(*this);
                    operator++();
                    return tmp;
                }

                GT_FUNCTION point_t const &operator*() const { return m_pos; }

                GT_FUNCTION bool operator!=(const grid_iterator &other) const { return m_pos != other.m_pos; }
            };

          public:
            GT_FUNCTION hypercube_view(const point_t &begin, const point_t &end) : m_begin(begin), m_end(end) {}
            GT_FUNCTION hypercube_view(const point_t &end) : m_end(end) {}

            GT_FUNCTION grid_iterator begin() const { return grid_iterator{m_begin, m_begin, m_end}; }
            GT_FUNCTION grid_iterator end() const { return grid_iterator{m_end, m_begin, m_end}; }

          private:
            point_t m_begin = {};
            point_t m_end;
        };
    }

    /**
     * @brief constructs a view on a hypercube from an array of ranges (e.g. pairs); the end of the range is exclusive.
     */
    template < typename Container,
        typename Decayed = typename std::decay< Container >::type,
        size_t OuterD = tuple_size< Decayed >::value,
        size_t InnerD = tuple_size< typename tuple_element< 0, Decayed >::type >::value,
        typename std::enable_if< OuterD != 0 && InnerD == 2, int >::type = 0 >
    GT_FUNCTION impl_::hypercube_view< OuterD > make_hypercube_view(Container &&cube) {
        auto &&transposed = transpose(std::forward< Container >(cube));
        return {convert_to_array< size_t >(transposed[0]), convert_to_array< size_t >(transposed[1])};
    }

    /**
     * @brief short-circuit for zero dimensional hypercube (transpose cannot work)
     */
    template < typename Container,
        size_t D = tuple_size< typename std::decay< Container >::type >::value,
        typename std::enable_if< D == 0, int >::type = 0 >
    GT_FUNCTION array< array< size_t, 0 >, 0 > make_hypercube_view(Container &&) {
        return {};
    }

    /**
     * @brief constructs a view on a hypercube from an array of integers (size of the loop in each dimension, ranges
     * start from 0); the end of the range is exclusive.
     */
    template < typename Container,
        typename Decayed = typename std::decay< Container >::type,
        size_t D = tuple_size< Decayed >::value,
        typename std::enable_if< D != 0 &&
                                     std::is_convertible< size_t, typename tuple_element< 0, Decayed >::type >::value,
            int >::type = 0 >
    GT_FUNCTION impl_::hypercube_view< D > make_hypercube_view(Container &&sizes) {
        return {convert_to_array< size_t >(std::forward< Container >(sizes))};
    }
}
