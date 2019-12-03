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
#include <cstddef>
#include <type_traits>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/generic_metafunctions/accumulate.hpp"
#include "../common/host_device.hpp"
#include "../common/layout_map.hpp"

namespace gridtools {
    namespace storage {
        namespace info_impl_ {
            template <class Layout>
            GT_CONSTEXPR uint_t make_padded_length(Layout, int layout_arg, uint_t align, uint_t length) {
                if (layout_arg == -1)
                    return 1;
                if (layout_arg == Layout::max_arg && layout_arg != 0)
                    return (length + align - 1) / align * align;
                return length;
            }

            template <class Layout, class Array>
            GT_CONSTEXPR uint_t make_stride(Layout, int layout_arg, Array const &padded_lengths) {
                if (layout_arg == -1)
                    return 0;
                uint_t res = 1;
                for (int i = Layout::max_arg; i != layout_arg; --i)
                    res *= padded_lengths[Layout::find(i)];
                return res;
            }

            template <int... Dims, int... LayoutArgs, class Array>
            GT_CONSTEXPR Array make_strides(layout_map<LayoutArgs...> layout, uint_t align, Array const &lengths) {
                assert(align > 0);
                Array padded_lengths = {make_padded_length(layout, LayoutArgs, align, lengths[Dims])...};
                return {make_stride(layout, LayoutArgs, padded_lengths)...};
            }

            template <class>
            struct base;

            template <size_t... Dims>
            struct base<std::index_sequence<Dims...>> {
                static constexpr size_t ndims = sizeof...(Dims);

              private:
                using array_t = array<uint_t, ndims>;
                template <size_t>
                using index_type = uint_t;

                array_t m_lengths;
                array_t m_strides;
                uint_t m_length;

              public:
                constexpr base() : m_lengths{}, m_strides{}, m_length(0) {}

                template <int... LayoutArgs>
                GT_CONSTEXPR base(layout_map<LayoutArgs...> layout, uint_t align, array_t const &lengths)
                    : m_lengths(lengths), m_strides(make_strides<Dims...>(layout, align, m_lengths)),
                      m_length(accumulate(logical_and(), true, m_lengths[Dims]...) ? index((m_lengths[Dims] - 1)...) + 1
                                                                                   : 0) {}

                GT_FUNCTION GT_CONSTEXPR auto const &lengths() const { return m_lengths; }
                GT_FUNCTION GT_CONSTEXPR auto const &strides() const { return m_strides; }

                GT_FUNCTION GT_CONSTEXPR auto index(index_type<Dims>... indices) const {
                    assert(accumulate(logical_and(), true, (indices < m_lengths[Dims])...));
                    return accumulate(plus_functor(), 0, indices * m_strides[Dims]...);
                }
                GT_FUNCTION GT_CONSTEXPR auto index(array<int, ndims> const &indices) const {
                    return index(indices[Dims]...);
                }

                template <int... LayoutArgs>
                GT_FUNCTION GT_CONSTEXPR array_t indices(layout_map<LayoutArgs...>, uint_t index) const {
                    return {(LayoutArgs == -1
                                 ? m_lengths[Dims] - 1
                                 : LayoutArgs ? index % m_strides[layout_map<LayoutArgs...>::find(LayoutArgs - 1)] /
                                                    m_strides[Dims]
                                              : index / m_strides[Dims])...};
                }

                GT_FUNCTION GT_CONSTEXPR auto length() const { return m_length; }
            };
        } // namespace info_impl_

        template <size_t N>
        struct info : info_impl_::base<std::make_index_sequence<N>> {
            using info_impl_::base<std::make_index_sequence<N>>::base;
        };

        template <size_t N>
        GT_FUNCTION GT_CONSTEXPR bool operator==(info<N> const &lhs, info<N> const &rhs) {
            return lhs.lengths() == rhs.lengths() && lhs.strides() == rhs.strides();
        }

        template <size_t N>
        GT_FUNCTION GT_CONSTEXPR bool operator!=(info<N> const &lhs, info<N> const &rhs) {
            return !(lhs == rhs);
        }
    } // namespace storage
} // namespace gridtools
