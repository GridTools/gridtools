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
#include "./executor.hpp"

namespace gridtools::fn {
    namespace unstructured::dim {
        struct neighbor {};
        struct k {};
    } // namespace unstructured::dim

    namespace unstructured_impl_ {
        using gridtools::stencil::positional;
        namespace dim = unstructured::dim;

        template <class Tag, class From, class To, class PtrHolder, class Strides, int MaxNeighbors>
        struct connectivity_table {
            PtrHolder m_ptr_holder;
            Strides m_strides;
            using from_t = From;
            using to_t = To;
            using tag_t = Tag;
            using max_neighbors_t = integral_constant<int, MaxNeighbors>;
        };

        template <class Tables, class Sizes>
        struct domain {
            Tables m_tables;
            Sizes m_sizes;
        };

        template <class Tag, class From, class To, class Sid, int I>
        auto connectivity(Sid const &s, integral_constant<int, I>) {
            auto ptr_holder = sid::get_origin(s);
            auto strides = sid::get_strides(s);
            return connectivity_table<Tag, From, To, decltype(ptr_holder), decltype(strides), I>{
                std::move(ptr_holder), std::move(strides)};
        }

        template <class Horizontal, class... Connectivities>
        auto unstructured_domain(
            std::size_t horizontal_size, std::size_t vertical_size, Connectivities const &...conns) {
            auto table_map = hymap::keys<typename Connectivities::tag_t...>::make_values(conns...);
            auto sizes = hymap::keys<Horizontal, dim::k>::make_values(horizontal_size, vertical_size);
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
        bool can_deref(iterator<Tag, Ptr, Strides, Domain> const &it) {
            return it.m_index != -1;
        }

        template <class Tag, class Ptr, class Strides, class Domain>
        auto deref(iterator<Tag, Ptr, Strides, Domain> const &it) {
            return *it.m_ptr;
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Conn, class Offset>
        GT_FUNCTION auto horizontal_shift(iterator<Tag, Ptr, Strides, Domain> const &it, Conn, Offset offset) {
            auto const &table = at_key<Conn>(it.m_domain.m_tables);
            using from_t = typename std::remove_reference_t<decltype(table)>::from_t;
            using to_t = typename std::remove_reference_t<decltype(table)>::to_t;
            auto table_ptr = table.m_ptr_holder();
            auto hori_stride = sid::get_stride<from_t>(table.m_strides);
            auto nb_stride = sid::get_stride<dim::neighbor>(table.m_strides);
            sid::shift(table_ptr, hori_stride, it.m_index);
            sid::shift(table_ptr, nb_stride, offset);
            auto table_value = *table_ptr;
            auto shifted = it;
            auto from_stride = at_key<Tag>(sid::get_stride<from_t>(shifted.m_strides));
            sid::shift(shifted.m_ptr, from_stride, -shifted.m_index);
            auto to_stride = at_key<Tag>(sid::get_stride<to_t>(shifted.m_strides));
            sid::shift(shifted.m_ptr, to_stride, table_value);
            shifted.m_index = table_value;
            return shifted;
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Dim, class Offset>
        GT_FUNCTION auto non_horizontal_shift(iterator<Tag, Ptr, Strides, Domain> const &it, Dim, Offset offset) {
            auto shifted = it;
            sid::shift(shifted.m_ptr, at_key<Tag>(sid::get_stride<Dim>(shifted.m_strides)), offset);
            return shifted;
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Dim, class Offset, class... Offsets>
        GT_FUNCTION auto shift(iterator<Tag, Ptr, Strides, Domain> const &it, Dim, Offset offset, Offsets... offsets) {
            if constexpr (has_key<decltype(it.m_domain.m_tables), Dim>()) {
                return shift(horizontal_shift(it, Dim(), offset), offsets...);
            } else {
                return shift(non_horizontal_shift(it, Dim(), offset), offsets...);
            }
        }
        template <class Tag, class Ptr, class Strides, class Domain>
        GT_FUNCTION auto shift(iterator<Tag, Ptr, Strides, Domain> const &it) {
            return it;
        }

        template <class Domain>
        struct make_iterator {
            Domain m_domain;

            GT_FUNCTION auto operator()() const {
                return [&](auto tag, auto const &ptr, auto const &strides) {
                    auto tptr = host_device::at_key<decltype(tag)>(ptr);
                    int index = *host_device::at_key<integral_constant<int, 0>>(ptr);
                    return iterator<decltype(tag), decltype(tptr), decltype(strides), Domain>{
                        std::move(tptr), strides, m_domain, index};
                };
            }
        };

        template <class Backend, class Domain>
        using stencil_exec_t = stencil_executor<Backend, make_iterator<Domain>, decltype(Domain::m_sizes), 1>;

        template <class Backend, class Domain>
        struct backend {
            Domain m_domain;

            auto stencil_executor() const {
                using horizontal_t = meta::at_c<get_keys<std::remove_reference_t<decltype(m_domain.m_sizes)>>, 0>;
                return [&] {
                    auto exec = stencil_exec_t<Backend, Domain>{m_domain.m_sizes, make_iterator<Domain>{m_domain}};
                    return std::move(exec).arg(positional<horizontal_t>());
                };
            }
        };

        template <class Backend, class Tables, class Sizes>
        auto make_backend(Backend, domain<Tables, Sizes> const &d) {
            return backend<Backend, domain<Tables, Sizes>>{d};
        }
    } // namespace unstructured_impl_

    using unstructured_impl_::connectivity;
    using unstructured_impl_::unstructured_domain;
} // namespace gridtools::fn
