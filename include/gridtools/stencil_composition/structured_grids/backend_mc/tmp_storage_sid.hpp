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
#include "../../sid/concept.hpp"
#include "../../sid/simple_ptr_holder.hpp"
#include "../../sid/synthetic.hpp"

namespace gridtools {

    struct thread_dim_mc;

    namespace _impl_tmp_mc {

        using byte_alignment = std::integral_constant<std::size_t, 64>;

        /**
         * @brief Pads a size to multiple of cache line size. If this is not possible without using additional cache
         * lines, the original size is returned.
         */
        template <class T>
        std::size_t pad(std::size_t size) {
            std::size_t padded_bytesize =
                (size * sizeof(T) + byte_alignment::value - 1) / byte_alignment::value * byte_alignment::value;
            return padded_bytesize % sizeof(T) == 0 ? padded_bytesize / sizeof(T) : size;
        }

        /**
         * @brief Block size including extents and padding.
         */
        template <class T, class Extent>
        pos3<std::size_t> full_block_size(pos3<std::size_t> const &block_size) {
            // add padding to the i dimension to align all first elements along the i dimension
            const std::size_t size_i = pad<T>(block_size.i - Extent::iminus::value + Extent::iplus::value);
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
            // allocate one extra cache line to allow for offsetting the initial allocation
            // to guarantee alignment of first element inside domain
            constexpr std::size_t extra = (byte_alignment::value + sizeof(T) - 1) / sizeof(T);
            return bs.i * bs.j * bs.k * omp_get_max_threads() + extra;
        }

        template <std::size_t, class>
        struct strides_kind_impl;

        /**
         * @brief Strides kind tag. Strides depend on data type size (due to cache-line alignment) and extent.
         */
        template <class T, class Extent>
        using strides_kind = strides_kind_impl<sizeof(T), Extent>;

        /**
         * @brief Strides, depending on data type due to padding to cache-line size. Specialization for non-zero extents
         * along k-dimension.
         */
        template <class T, class Extent, enable_if_t<Extent::kminus::value != 0 || Extent::kplus::value != 0, int> = 0>
        hymap::keys<dim::i, dim::j, dim::k, thread_dim_mc>::values<integral_constant<int_t, 1>, int_t, int_t, int_t>
        strides(pos3<std::size_t> const &block_size) {
            auto bs = full_block_size<T, Extent>(block_size);
            return {integral_constant<int, 1>{}, bs.i * bs.k, bs.i, bs.i * bs.j * bs.k};
        }

        /**
         * @brief Strides, depending on data type due to padding to cache-line size. Specialization for zero extents
         * along k-dimension.
         */
        template <class T, class Extent, enable_if_t<Extent::kminus::value == 0 && Extent::kplus::value == 0, int> = 0>
        hymap::keys<dim::i, dim::j, thread_dim_mc>::values<integral_constant<int_t, 1>, int_t, int_t> strides(
            pos3<std::size_t> const &block_size) {
            auto bs = full_block_size<T, Extent>(block_size);
            assert(bs.k == 1);
            return {integral_constant<int, 1>{}, bs.i, bs.i * bs.j};
        }

        /**
         * @brief Offset from allocation start to first element inside compute domain.
         */
        template <class T, class Extent>
        std::size_t origin_offset(pos3<std::size_t> const &block_size) {
            auto st = strides<T, Extent>(block_size);
            std::size_t offset = sid::get_stride<dim::i>(st) * -Extent::iminus::value +
                                 sid::get_stride<dim::j>(st) * -Extent::jminus::value +
                                 sid::get_stride<dim::k>(st) * -Extent::kminus::value;
            // Add padding at the start of the allocation to align the first element inside the domain
            return pad<T>(offset);
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
    auto make_tmp_storage_mc(Allocator &allocator, pos3<std::size_t> const &block_size)
        GT_AUTO_RETURN((sid::synthetic()
                            .set<sid::property::origin>(
                                allocator.template allocate<T>(_impl_tmp_mc::storage_size<T, Extent>(block_size)) +
                                _impl_tmp_mc::origin_offset<T, Extent>(block_size))
                            .template set<sid::property::strides>(_impl_tmp_mc::strides<T, Extent>(block_size))
                            .template set<sid::property::strides_kind, _impl_tmp_mc::strides_kind<T, Extent>>()
                            .template set<sid::property::ptr_diff, int_t>()));

} // namespace gridtools
