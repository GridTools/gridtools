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
    } // namespace unstructured::dim

    namespace unstructured_impl_ {
        using gridtools::stencil::positional;
        namespace dim = unstructured::dim;
        using backend::data_type;

        template <class Tables, class Sizes>
        struct domain {
            Tables m_tables;
            Sizes m_sizes;
        };

        template <class Tables, class Sizes, class Offsets>
        struct domain_with_offsets {
            domain<Tables, Sizes> m_domain;
            Offsets m_offsets;

            domain_with_offsets(Tables const &tables, Sizes const &sizes, Offsets const &offsets)
                : m_domain{tables, sizes}, m_offsets(offsets) {}
        };

        template <class Tag, class NeighborTable>
        typename hymap::keys<Tag>::template values<NeighborTable> connectivity(NeighborTable const &nt) {
            return {nt};
        }

        template <class Sizes,
            class Offsets,
            class... Connectivities,
            std::enable_if_t<!std::is_integral_v<Sizes>, int> = 0>
        auto unstructured_domain(Sizes const &sizes, Offsets const &offsets, Connectivities const &...conns) {

            return domain_with_offsets(hymap::concat(conns...), sizes, offsets);
        };

        template <class... Connectivities>
        auto unstructured_domain(int horizontal_size, int vertical_size, Connectivities const &...conns) {
            return domain_with_offsets(hymap::concat(conns...),
                hymap::keys<dim::horizontal, dim::vertical>::make_values(horizontal_size, vertical_size),
                std::tuple());
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
            assert(can_deref(it));
            decltype(auto) stride = host_device::at_key<Tag>(sid::get_stride<dim::horizontal>(it.m_strides));
            return *sid::shifted(it.m_ptr, stride, it.m_index);
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Conn, class Offset>
        GT_FUNCTION constexpr auto horizontal_shift(iterator<Tag, Ptr, Strides, Domain> const &it, Conn, Offset) {
            auto const &table = host_device::at_key<Conn>(it.m_domain.m_tables);
            auto new_index = get<Offset::value>(neighbor_table::neighbors(table, it.m_index));
            auto shifted = it;
            shifted.m_index = new_index;
            return shifted;
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Dim, class Offset>
        GT_FUNCTION constexpr auto non_horizontal_shift(
            iterator<Tag, Ptr, Strides, Domain> const &it, Dim, Offset offset) {
            auto shifted = it;
            sid::shift(shifted.m_ptr, host_device::at_key<Tag>(sid::get_stride<Dim>(shifted.m_strides)), offset);
            return shifted;
        }

        template <class Tag, class Ptr, class Strides, class Domain, class Dim, class Offset, class... Offsets>
        GT_FUNCTION constexpr auto shift(
            iterator<Tag, Ptr, Strides, Domain> const &it, Dim, Offset offset, Offsets... offsets) {
            if (it.m_index == -1)
                return it;

            if constexpr (has_key<decltype(it.m_domain.m_tables), Dim>()) {
                return shift(horizontal_shift(it, Dim(), offset), offsets...);
            } else {
                return shift(non_horizontal_shift(it, Dim(), offset), offsets...);
            }
        }
        template <class Tag, class Ptr, class Strides, class Domain>
        GT_FUNCTION constexpr auto shift(iterator<Tag, Ptr, Strides, Domain> const &it) {
            return it;
        }

        template <class Domain>
        struct make_iterator {
            Domain m_domain;

            GT_FUNCTION auto operator()() const {
                return [&](auto tag, auto const &ptr, auto const &strides) {
                    auto tptr = host_device::at_key<decltype(tag)>(ptr);
                    int index = *host_device::at_key<integral_constant<int, 0>>(ptr);
                    decltype(auto) stride =
                        host_device::at_key<decltype(tag)>(sid::get_stride<dim::horizontal>(strides));
                    sid::shift(tptr, stride, -index);
                    return iterator<decltype(tag), decltype(tptr), decltype(strides), Domain>{
                        std::move(tptr), strides, m_domain, index};
                };
            }
        };

        template <class Backend, class Domain, class Offsets>
        using stencil_exec_t = stencil_executor<Backend, make_iterator<Domain>, decltype(Domain::m_sizes), Offsets, 1>;

        template <class Backend, class Domain, class Offsets>
        using vertical_exec_t =
            vertical_executor<Backend, make_iterator<Domain>, dim::vertical, decltype(Domain::m_sizes), Offsets, 1>;

        template <class Backend, class Domain, class TmpAllocator>
        struct backend {
            Domain m_domain;
            TmpAllocator m_allocator;

            static constexpr auto index = positional<dim::horizontal>();

            template <class T>
            auto make_tmp() {
                return allocate_global_tmp(m_allocator, m_domain.m_domain.m_sizes, data_type<T>());
            }

            auto stencil_executor() const {
                using domain_t = decltype(Domain::m_domain);
                using offsets_t = decltype(Domain::m_offsets);
                return [&] {
                    auto exec = stencil_exec_t<Backend, domain_t, offsets_t>{
                        m_domain.m_domain.m_sizes, m_domain.m_offsets, make_iterator<domain_t>{m_domain.m_domain}};
                    return std::move(exec).arg(index);
                };
            }

            auto vertical_executor() const {
                using domain_t = decltype(Domain::m_domain);
                using offsets_t = decltype(Domain::m_offsets);
                return [&] {
                    auto exec = vertical_exec_t<Backend, domain_t, offsets_t>{
                        m_domain.m_domain.m_sizes, m_domain.m_offsets, make_iterator<domain_t>{m_domain.m_domain}};
                    return std::move(exec).arg(index);
                };
            }
        };

        template <class Backend, class Tables, class Sizes, class Offsets>
        auto make_backend(Backend, domain_with_offsets<Tables, Sizes, Offsets> const &d) {
            auto allocator = tmp_allocator(Backend());
            return backend<Backend, domain_with_offsets<Tables, Sizes, Offsets>, decltype(allocator)>{
                d, std::move(allocator)};
        }
    } // namespace unstructured_impl_

    using unstructured_impl_::connectivity;
    using unstructured_impl_::unstructured_domain;
} // namespace gridtools::fn
