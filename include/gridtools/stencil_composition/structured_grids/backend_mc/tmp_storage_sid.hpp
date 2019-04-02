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

#include "../../../common/hugepage_alloc.hpp"
#include "../../../common/hymap.hpp"
#include "../../dim.hpp"
#include "../../pos3.hpp"
#include "../../sid/simple_ptr_holder.hpp"
#include "../../sid/synthetic.hpp"

namespace gridtools {

    struct thread_dim_mc;

    namespace _impl_tmp_mc {

        static constexpr std::size_t byte_alignment = 64;

        /**
         * @brief Block size including extents and padding.
         */
        template <class T, class Extent>
        pos3<std::size_t> full_block_size(pos3<std::size_t> const &block_size) {
            static constexpr std::size_t alignment = byte_alignment / sizeof(T);
            const std::size_t size_i =
                (block_size.i - Extent::iminus::value + Extent::iplus::value + alignment - 1) / alignment * alignment;
            const std::size_t size_j = block_size.j - Extent::jminus::value + Extent::jplus::value;
            const std::size_t size_k = block_size.k - Extent::kminus::value + Extent::kplus::value;
            return {size_i, size_j, size_k};
        }

        /**
         * @brief Size of the full allocation of a temporary buffer (in number of elements).
         */
        template <class T, class Extent>
        std::size_t storage_size(pos3<std::size_t> const &block_size) {
            auto bs = full_block_size<T, Extent>(block_size);
            return bs.i * bs.j * bs.k * omp_get_max_threads() + byte_alignment / sizeof(T);
        }

        /**
         * @brief Strides kind tag. Strides depend on data type (due to cache-line alignment) and extent.
         */
        template <class T, class Extent>
        struct strides_kind;

        /**
         * @brief Strides, depending on data type due to padding to cache-line size.
         */
        template <class T, class Extent>
        hymap::keys<dim::i, dim::j, dim::k, thread_dim_mc>::values<integral_constant<int_t, 1>, int_t, int_t, int_t>
        strides(pos3<std::size_t> const &block_size) {
            auto bs = full_block_size<T, Extent>(block_size);
            return {integral_constant<int, 1>{}, bs.i * bs.k, bs.i, bs.i * bs.j * bs.k};
        }

        /**
         * @brief Offset from allocation start to first element inside compute domain.
         */
        template <class T, class Extent>
        std::size_t origin_offset(pos3<std::size_t> const &block_size) {
            auto st = strides<T, Extent>(block_size);
            std::size_t offset = at_key<dim::i>(st) * -Extent::iminus::value +
                                 at_key<dim::j>(st) * -Extent::jminus::value +
                                 at_key<dim::k>(st) * -Extent::kminus::value;
            static constexpr std::size_t alignment = byte_alignment / sizeof(T);
            return (offset + alignment - 1) / alignment * alignment;
        }

    } // namespace _impl_tmp_mc

    /**
     * @brief Simple allocator for temporaries.
     */
    class tmp_allocator_mc {
        using deleter_t = std::integral_constant<decltype(&hugepage_free), &hugepage_free>;
        std::vector<std::unique_ptr<void, deleter_t>> m_ptrs;

      public:
        template <class T>
        sid::host::simple_ptr_holder<T *> allocate(std::size_t n) {
            m_ptrs.emplace_back(hugepage_alloc(n * sizeof(T)));
            return {static_cast<T *>(m_ptrs.back().get())};
        };
    };

    template <class T, class Extent, class Allocator>
    auto make_tmp_storage_mc(Allocator &allocator, pos3<std::size_t> const &block_size) GT_AUTO_RETURN((
        sid::synthetic()
            .set<sid::property::origin>(allocator.template allocate<T>(
                _impl_tmp_mc::storage_size<T, Extent>(block_size) + _impl_tmp_mc::origin_offset<T, Extent>(block_size)))
            .template set<sid::property::strides>(_impl_tmp_mc::strides<T, Extent>(block_size))
            .template set<sid::property::strides_kind, _impl_tmp_mc::strides_kind<T, Extent>>()));

} // namespace gridtools
