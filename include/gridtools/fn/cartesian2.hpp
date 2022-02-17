/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../common/tuple_util.hpp"
#include "../sid/concept.hpp"
#include "./backend2/common.hpp"
#include "./executor.hpp"

namespace gridtools::fn {
    namespace cartesian::dim {
        using i = integral_constant<int, 0>;
        using j = integral_constant<int, 1>;
        using k = integral_constant<int, 2>;
    } // namespace cartesian::dim

    namespace cartesian_impl_ {
        namespace dim = cartesian::dim;
        using backend::data_type_from_sid;

        using domain_t = hymap::keys<dim::i, dim::j, dim::k>::values<int, int, int>;

        inline domain_t cartesian_domain(int i, int j, int k) { return {i, j, k}; }

        template <class Sizes>
        domain_t cartesian_domain(Sizes const &sizes) {
            return {tuple_util::get<0>(sizes), tuple_util::get<1>(sizes), tuple_util::get<2>(sizes)};
        }

        template <class Tag, class Ptr, class Strides>
        struct iterator {
            Ptr m_ptr;
            Strides const &m_strides;
        };

        template <class Tag, class Ptr, class Strides>
        GT_FUNCTION auto deref(iterator<Tag, Ptr, Strides> const &it) {
            return *it.m_ptr;
        }

        template <class Tag, class Ptr, class Strides, class Dim, class Offset, class... Offsets>
        GT_FUNCTION void shift_impl(iterator<Tag, Ptr, Strides> &it, Dim, Offset offset, Offsets... offsets) {
            sid::shift(it.m_ptr, at_key<Tag>(sid::get_stride<Dim>(it.m_strides)), offset);
            shift_impl(it, offsets...);
        }
        template <class Tag, class Ptr, class Strides>
        GT_FUNCTION void shift_impl(iterator<Tag, Ptr, Strides> &) {}

        template <class Tag, class Ptr, class Strides, class... Offsets>
        GT_FUNCTION auto shift(iterator<Tag, Ptr, Strides> const &it, Offsets... offsets) {
            auto shifted = it;
            shift_impl(shifted, offsets...);
            return shifted;
        }

        struct make_iterator {
            GT_FUNCTION auto operator()() const {
                return [](auto tag, auto const &ptr, auto const &strides) {
                    auto tptr = host_device::at_key<decltype(tag)>(ptr);
                    return iterator<decltype(tag), decltype(tptr), decltype(strides)>{std::move(tptr), strides};
                };
            }
        };

        template <class Backend>
        using stencil_exec_t = stencil_executor<Backend, make_iterator, domain_t>;
        template <class Backend>
        using vertical_exec_t = vertical_executor<Backend, make_iterator, dim::k, domain_t>;

        template <class Backend, class TmpAllocator>
        struct backend {
            domain_t m_domain;
            TmpAllocator m_allocator;

            template <class Sid>
            auto make_tmp_like(Sid const &s) {
                return allocate_global_tmp(m_allocator, m_domain, data_type_from_sid(s));
            }

            auto stencil_executor() const {
                return [&] { return stencil_exec_t<Backend>{m_domain, make_iterator{}}; };
            }

            auto vertical_executor() const {
                return [&] { return vertical_exec_t<Backend>{m_domain, make_iterator{}}; };
            }
        };

        template <class Backend>
        auto make_backend(Backend, domain_t const &domain) {
            auto allocator = tmp_allocator(Backend());
            return backend<Backend, decltype(allocator)>{domain, std::move(allocator)};
        }
    } // namespace cartesian_impl_
    using cartesian_impl_::cartesian_domain;
    using cartesian_impl_::make_backend;
} // namespace gridtools::fn
