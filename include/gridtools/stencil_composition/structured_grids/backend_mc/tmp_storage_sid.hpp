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

#include <memory>
#include <vector>

#include <omp.h>

#include "../../../common/functional.hpp"
#include "../../../common/hymap.hpp"
#include "../../dim.hpp"
#include "../../pos3.hpp"
#include "../../sid/synthetic.hpp"

namespace gridtools {

    struct thread_dim;

    namespace _impl_tmp_mc {

        template <class T>
        struct ptr_holder {
            T *m_ptr;

            GT_FORCE_INLINE constexpr T *operator()() const { return m_ptr; }

            ptr_holder operator+(std::size_t offset) const { return {m_ptr + offset}; }
        };

        template <class T, class Extent>
        pos3<std::size_t> full_block_size(pos3<std::size_t> const &block_size) {
            static constexpr std::size_t alignment = 64 / sizeof(T);
            const std::size_t size_i =
                (block_size.i - Extent::iminus::value + Extent::iplus::value + alignment - 1) / alignment * alignment;
            const std::size_t size_j = block_size.j - Extent::jminus::value + Extent::jplus::value;
            const std::size_t size_k = block_size.k - Extent::kminus::value + Extent::kplus::value;
            return {size_i, size_j, size_k};
        }

        template <class T, class Extent>
        std::size_t storage_size(pos3<std::size_t> const &block_size) {
            auto bs = full_block_size<T, Extent>(block_size);
            return bs.i * bs.j * bs.k * omp_get_max_threads();
        }

        using strides_t =
            hymap::keys<dim::i, dim::k, dim::j, thread_dim>::values<integral_constant<int_t, 1>, int_t, int_t, int_t>;

        template <class T, class Extent>
        strides_t strides(pos3<std::size_t> const &block_size) {
            auto bs = full_block_size<T, Extent>(block_size);
            return {integral_constant<int, 1>{}, bs.i, bs.i * bs.k, bs.i * bs.k * bs.j};
        }

        template <class T, class Extent>
        std::size_t origin_offset(pos3<std::size_t> const &block_size) {
            auto st = strides<T, Extent>(block_size);
            return at_key<dim::i>(st) * -Extent::iminus::value + at_key<dim::j>(st) * -Extent::jminus::value +
                   at_key<dim::k>(st) * -Extent::kminus::value;
        }

    } // namespace _impl_tmp_mc

    class tmp_allocator_mc {
      public:
        template <class T>
        _impl_tmp_mc::ptr_holder<T> allocate(std::size_t n) const {
            static std::vector<std::unique_ptr<T[]>> m_ptrs;
            // TODO: align and shift
            m_ptrs.emplace_back(new T[n]);
            return {m_ptrs.back().get()};
        };
    };

    template <class T, class Extent, class Allocator>
    auto make_tmp_storage_mc(Allocator &allocator, pos3<std::size_t> const &block_size) GT_AUTO_RETURN((
        sid::synthetic()
            .set<sid::property::origin>(allocator.template allocate<T>(
                _impl_tmp_mc::storage_size<T, Extent>(block_size) + _impl_tmp_mc::origin_offset<T, Extent>(block_size)))
            .template set<sid::property::strides_kind, _impl_tmp_mc::strides_t>()
            .template set<sid::property::strides>(_impl_tmp_mc::strides<T, Extent>(block_size))));

} // namespace gridtools
