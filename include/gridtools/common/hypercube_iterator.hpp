/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "array.hpp"
#include "defs.hpp"
#include "host_device.hpp"
#include "tuple_util.hpp"

namespace gridtools {
    namespace impl_ {

        template <size_t D>
        class hypercube_view {
          private:
            using point_t = array<size_t, D>;
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
    } // namespace impl_

    /**
     * @brief constructs a view on a hypercube from an array of ranges (e.g. pairs); the end of the range is exclusive.
     */
    template <typename Container,
        typename Decayed = typename std::decay<Container>::type,
        size_t OuterD = tuple_size<Decayed>::value,
        size_t InnerD = tuple_size<typename tuple_element<0, Decayed>::type>::value,
        typename std::enable_if<OuterD != 0 && InnerD == 2, int>::type = 0>
    GT_FUNCTION impl_::hypercube_view<OuterD> make_hypercube_view(Container &&cube) {
        auto &&transposed = tuple_util::host_device::transpose(std::forward<Container>(cube));
        return {tuple_util::host_device::convert_to<array, size_t>(tuple_util::host_device::get<0>(transposed)),
            tuple_util::host_device::convert_to<array, size_t>(tuple_util::host_device::get<1>(transposed))};
    }

    /**
     * @brief short-circuit for zero dimensional hypercube (transpose cannot work)
     */
    template <typename Container,
        size_t D = tuple_size<typename std::decay<Container>::type>::value,
        typename std::enable_if<D == 0, int>::type = 0>
    GT_FUNCTION array<array<size_t, 0>, 0> make_hypercube_view(Container &&) {
        return {};
    }

    /**
     * @brief constructs a view on a hypercube from an array of integers (size of the loop in each dimension, ranges
     * start from 0); the end of the range is exclusive.
     */
    template <typename Container,
        typename Decayed = typename std::decay<Container>::type,
        size_t D = tuple_size<Decayed>::value,
        typename std::enable_if<D != 0 && std::is_convertible<size_t, typename tuple_element<0, Decayed>::type>::value,
            int>::type = 0>
    GT_FUNCTION impl_::hypercube_view<D> make_hypercube_view(Container &&sizes) {
        return {tuple_util::host_device::convert_to<array, size_t>(std::forward<Container>(sizes))};
    }
} // namespace gridtools
