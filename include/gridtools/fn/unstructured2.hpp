/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "../common/hymap.hpp"
#include "../sid/concept.hpp"
#include "../stencil/positional.hpp"
#include "./backend2/common.hpp"
#include "./executor.hpp"
#include "./neighbor_table.hpp"

namespace gridtools::fn {
    namespace unstructured::dim {
        using horizontal = integral_constant<int, 0>;
        using vertical = integral_constant<int, 1>;
        using k = vertical;
    } // namespace unstructured::dim

    namespace unstructured_impl_ {
        using gridtools::stencil::positional;
        namespace dim = unstructured::dim;
        using backend::data_type;

        template <class Tag, class From, class To, class NeighborTable>
        struct connectivity_table {
            NeighborTable const &m_neighbor_table;
            using from_t = From;
            using to_t = To;
            using tag_t = Tag;
        };

        template <class Tables, class Sizes>
        struct domain {
            Tables m_tables;
            Sizes m_sizes;
        };

        template <class Tag, class From, class To, class NeighborTable>
        auto connectivity(NeighborTable const &nt) {
            return connectivity_table<Tag, From, To, NeighborTable>{nt};
        }

        template <class... Connectivities>
        auto unstructured_domain(
            std::size_t horizontal_size, std::size_t vertical_size, Connectivities const &...conns) {
            auto table_map = hymap::keys<typename Connectivities::tag_t...>::make_values(conns...);
            auto sizes = hymap::keys<dim::horizontal, dim::vertical>::make_values(horizontal_size, vertical_size);
            return domain<decltype(table_map), decltype(sizes)>{std::move(table_map), std::move(sizes)};
        };

        template <class Tag, class Ptr, class Strides, class Domain>
        struct iterator {
            Ptr m_ptr;
            Strides const &m_strides;
            Domain const &m_domain;
            int m_index;
        };

        template <class Tag, class Ptr, class Strides, class Domain>
        GT_FUNCTION constexpr bool can_deref(iterator<Tag, Ptr, Strides, Domain> const &it) {
            return it.m_index != -1;
        }

        template <class Tag, class Ptr, class Strides, class Domain>
        GT_FUNCTION constexpr auto deref(iterator<Tag, Ptr, Strides, Domain> const &it) {
            assert(m_index != -1);
            decltype(auto) stride = at_key<Tag>(sid::get_stride<dim::horizontal>(it.m_strides));
            return *sid::shifted(it.m_ptr, stride, it.m_index);
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Conn, class Offset>
        GT_FUNCTION constexpr auto horizontal_shift(iterator<Tag, Ptr, Strides, Domain> const &it, Conn, Offset) {
            auto const &table = at_key<Conn>(it.m_domain.m_tables);
            auto new_index = get<Offset::value>(neighbor_table::neighbors(table.m_neighbor_table, it.m_index));
            auto shifted = it;
            shifted.m_index = new_index;
            return shifted;
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Dim, class Offset>
        GT_FUNCTION constexpr auto non_horizontal_shift(
            iterator<Tag, Ptr, Strides, Domain> const &it, Dim, Offset offset) {
            auto shifted = it;
            sid::shift(shifted.m_ptr, at_key<Tag>(sid::get_stride<Dim>(shifted.m_strides)), offset);
            return shifted;
        }

        template <class Stride, class Tag, class = void>
        struct has_horizontal_stride : std::false_type {};

        template <class Stride, class Tag>
        struct has_horizontal_stride<Stride,
            Tag,
            std::enable_if_t<
                !is_integral_constant_of<std::decay_t<decltype(at_key<Tag>(std::declval<Stride>()))>, 0>()>>
            : std::true_type {};

        template <class Tag, class Ptr, class Strides, class Domain, class Dim, class Offset, class... Offsets>
        GT_FUNCTION constexpr auto shift(
            iterator<Tag, Ptr, Strides, Domain> const &it, Dim, Offset offset, Offsets... offsets) {
            if (it.m_index == -1)
                return it;

            using stride_t = std::decay_t<decltype(sid::get_stride<Dim>(std::declval<Strides>()))>;
            if constexpr (has_key<decltype(it.m_domain.m_tables), Dim>() && !has_horizontal_stride<stride_t, Tag>()) {
                return shift(horizontal_shift(it, Dim(), offset), offsets...);
            } else {
                return shift(non_horizontal_shift(it, Dim(), offset), offsets...);
            }
        }
        template <class Tag, class Ptr, class Strides, class Domain>
        GT_FUNCTION constexpr auto shift(iterator<Tag, Ptr, Strides, Domain> const &it) {
            return it;
        }

        template <class Conn, class F, class Init, class... Tags, class... Ptrs, class... Strides, class... Domains>
        GT_FUNCTION constexpr auto reduce(Conn, F f, Init init, iterator<Tags, Ptrs, Strides, Domains> const &...its) {
            auto res = std::move(init);
            tuple_util::for_each(
                [&](auto offset) {
                    if ((... && can_deref(shift(its, Conn(), offset))))
                        res = f(res, deref(shift(its, Conn(), offset))...);
                },
                meta::rename<tuple, meta::make_indices_c<Conn::max_neighbors>>());
            return res;
        }

        template <class Domain>
        struct make_iterator {
            Domain m_domain;

            GT_FUNCTION auto operator()() const {
                return [&](auto tag, auto const &ptr, auto const &strides) {
                    auto tptr = host_device::at_key<decltype(tag)>(ptr);
                    int index = *host_device::at_key<integral_constant<int, 0>>(ptr);
                    decltype(auto) stride = at_key<decltype(tag)>(sid::get_stride<dim::horizontal>(strides));
                    sid::shift(tptr, stride, -index);
                    return iterator<decltype(tag), decltype(tptr), decltype(strides), Domain>{
                        std::move(tptr), strides, m_domain, index};
                };
            }
        };

        template <class Backend, class Domain>
        using stencil_exec_t = stencil_executor<Backend, make_iterator<Domain>, decltype(Domain::m_sizes), 1>;

        template <class Backend, class Domain, class TmpAllocator>
        struct backend {
            Domain m_domain;
            TmpAllocator m_allocator;

            template <class Sid>
            auto make_tmp_like(Sid const &) {
                using element_t = sid::element_type<Sid>;
                return allocate_global_tmp(m_allocator, m_domain.m_sizes, data_type<element_t>());
            }

            auto stencil_executor() const {
                using horizontal_t = meta::at_c<get_keys<std::remove_reference_t<decltype(m_domain.m_sizes)>>, 0>;
                static const auto index = positional<horizontal_t>();
                return [&] {
                    auto exec = stencil_exec_t<Backend, Domain>{m_domain.m_sizes, make_iterator<Domain>{m_domain}};
                    return std::move(exec).arg(index);
                };
            }
        };

        template <class Backend, class Tables, class Sizes>
        auto make_backend(Backend, domain<Tables, Sizes> const &d) {
            auto allocator = tmp_allocator(Backend());
            return backend<Backend, domain<Tables, Sizes>, decltype(allocator)>{d, std::move(allocator)};
        }
    } // namespace unstructured_impl_

    using unstructured_impl_::connectivity;
    using unstructured_impl_::unstructured_domain;
} // namespace gridtools::fn
