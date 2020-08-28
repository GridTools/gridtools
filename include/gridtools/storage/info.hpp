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
#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"

namespace gridtools {
    namespace storage {
        namespace info_impl_ {

            template <class>
            struct layout_tuple;

            template <int... Is>
            struct layout_tuple<layout_map<Is...>> {
                using type = tuple<integral_constant<int, Is>...>;
            };

            template <class Layout, class Align>
            struct make_padded_length_f {
                Align m_align;
                template <class Length>
                integral_constant<int, 1> operator()(integral_constant<int, -1>, Length) const {
                    return {};
                }
                template <int Max = Layout::max_arg, class Length, std::enable_if_t<Max != 0 && Max != -1, int> = 0>
                auto operator()(integral_constant<int, Layout::max_arg>, Length length) const {
                    return (length + m_align - integral_constant<int, 1>()) / m_align * m_align;
                }
                template <int I, class Length>
                auto operator()(integral_constant<int, I>, Length length) const {
                    return length;
                }
            };

            template <class Layout, class Align, class Lengths>
            auto make_padded_lengths(Align align, Lengths const &lengths) {
                return tuple_util::transform(
                    make_padded_length_f<Layout, Align>{align}, typename layout_tuple<Layout>::type(), lengths);
            }

            template <class Layout, class Lengths>
            struct make_stride_f {
                Lengths const &m_lengths;
                integral_constant<int, 0> operator()(integral_constant<int, -1>) const { return {}; }
                template <int Max = Layout::max_arg, std::enable_if_t<Max != -1, int> = 0>
                integral_constant<int, 1> operator()(integral_constant<int, Layout::max_arg>) const {
                    return {};
                }
                template <int I>
                auto operator()(integral_constant<int, I>) const {
                    static constexpr size_t next = Layout::find(I + 1);
                    return tuple_util::get<next>(m_lengths) * (*this)(integral_constant<int, I + 1>());
                }
            };

            template <class Layout, class Lengths>
            auto make_strides_helper(Lengths const &lengths) {
                return tuple_util::transform(
                    make_stride_f<Layout, Lengths>{lengths}, typename layout_tuple<Layout>::type());
            }

            template <class Layout, class Align, class Lengths>
            auto make_strides(Align align, Lengths const &lengths) {
                return make_strides_helper<Layout>(make_padded_lengths<Layout>(align, lengths));
            }

            template <class Lengths, class Strides, class = std::make_index_sequence<tuple_util::size<Lengths>::value>>
            class info;

            template <class Lengths, class Strides, size_t... Dims>
            class info<Lengths, Strides, std::index_sequence<Dims...>> : tuple<Lengths, Strides> {
                static_assert(tuple_util::size<Lengths>::value == tuple_util::size<Strides>::value, GT_INTERNAL_ERROR);

                GT_FUNCTION tuple<Lengths, Strides> const &base() const { return *this; }

              public:
                static constexpr size_t ndims = tuple_util::size<Lengths>::value;

                info(Lengths lengths, Strides strides)
                    : tuple<Lengths, Strides>{std::move(lengths), std::move(strides)} {}

                GT_FUNCTION auto const &native_lengths() const { return tuple_util::host_device::get<0>(base()); }
                GT_FUNCTION auto const &native_strides() const { return tuple_util::host_device::get<1>(base()); }
                GT_FUNCTION int length() const {
                    return accumulate(logical_and(), true, lengths()[Dims]...) ? index((lengths()[Dims] - 1)...) + 1
                                                                               : 0;
                }
                GT_FUNCTION array<int_t, ndims> lengths() const {
                    return {(int_t)tuple_util::host_device::get<Dims>(native_lengths())...};
                }
                GT_FUNCTION array<int_t, ndims> strides() const {
                    return {(int_t)tuple_util::host_device::get<Dims>(native_strides())...};
                }

                template <class... Is>
                GT_FUNCTION auto index(Is... indices) const {
                    static_assert(sizeof...(Is) == ndims, GT_INTERNAL_ERROR);
                    assert(accumulate(
                        logical_and(), true, (indices < tuple_util::host_device::get<Dims>(native_lengths()))...));
                    return accumulate(plus_functor(),
                        integral_constant<int, 0>(),
                        indices * tuple_util::host_device::get<Dims>(native_strides())...);
                    //                    return tuple_util::host_device::fold([](auto l, auto r) { return l + r; },
                    //                        tuple_util::host_device::transform([](auto l, auto r) { return l * r; },
                    //                            tuple_util::host_device::make<tuple>(indices...),
                    //                            native_strides()));
                }

                template <class Indices>
                GT_FUNCTION auto index_from_tuple(Indices const &indices) const {
                    return index(tuple_util::host_device::get<Dims>(indices)...);
                }

                template <int... LayoutArgs>
                GT_FUNCTION array<int, ndims> indices(layout_map<LayoutArgs...>, int_t index) const {
                    return {(LayoutArgs == -1
                                 ? lengths()[Dims] - 1
                                 : LayoutArgs ? index % strides()[layout_map<LayoutArgs...>::find(LayoutArgs - 1)] /
                                                    strides()[Dims]
                                              : index / strides()[Dims])...};
                }
            };

            template <class Lengths, class Strides>
            info<Lengths, Strides> make_info_helper(Lengths lengths, Strides strides) {
                return {std::move(lengths), std::move(strides)};
            }

            template <class Layout, class Align, class Lengths>
            auto make_info(Align align, Lengths const &lengths) {
                return make_info_helper(lengths, make_strides<Layout>(align, lengths));
            }

            template <class Layout, class Align, class Length>
            uint_t make_padded_length(Layout, int layout_arg, Align align, Length length) {
                return layout_arg == -1
                           ? 1
                           : layout_arg == Layout::max_arg && layout_arg != 0 ? (length + align - 1) / align * align
                                                                              : length;
            }

            template <class Layout, class Array>
            uint_t make_stride(Layout, int layout_arg, Array const &padded_lengths) {
                if (layout_arg == -1)
                    return 0;
                uint_t res = 1;
                for (int i = Layout::max_arg; i != layout_arg; --i)
                    res *= padded_lengths[Layout::find(i)];
                return res;
            }

            template <int... Dims, int... LayoutArgs, class Align, class Lengths>
            Lengths make_strides_old(layout_map<LayoutArgs...> layout, Align align, Lengths const &lengths) {
                assert(align > 0);
                Lengths padded_lengths = {
                    make_padded_length(layout, LayoutArgs, align, tuple_util::get<Dims>(lengths))...};
                return {make_stride(layout, LayoutArgs, padded_lengths)...};
            }

            template <class, class>
            struct base;

            template <class Lengths, size_t... Dims>
            struct base<Lengths, std::index_sequence<Dims...>> {
                static constexpr size_t ndims = sizeof...(Dims);

              private:
                using array_t = array<uint_t, ndims>;
                template <size_t>
                using index_type = uint_t;

                Lengths m_lengths;
                array_t m_strides;
                uint_t m_length;

              public:
                template <int... LayoutArgs, class Align, class Lens>
                base(layout_map<LayoutArgs...> layout, Align align, Lens const &lengths)
                    : m_lengths{tuple_util::get<Dims>(lengths)...},
                      m_strides(make_strides_old<Dims...>(layout, align, m_lengths)),
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

        using info_impl_::make_info;

        template <size_t N>
        struct info : info_impl_::base<array<uint_t, N>, std::make_index_sequence<N>> {
            using info_impl_::base<array<uint_t, N>, std::make_index_sequence<N>>::base;
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
