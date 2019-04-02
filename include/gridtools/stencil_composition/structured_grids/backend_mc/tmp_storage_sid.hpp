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

        template <class Extent>
        pos3<std::size_t> full_block_size(pos3<std::size_t> const &block_size) {
            // TODO: alignment
            return {block_size.i - Extent::iminus::value + Extent::iplus::value,
                block_size.j - Extent::jminus::value + Extent::jplus::value,
                block_size.k};
        }

        template <class Extent>
        std::size_t block_storage_size(pos3<std::size_t> const &block_size) {
            auto bs = full_block_size<Extent>(block_size);
            return bs.i * bs.j * bs.k;
        }

        template <class Extent>
        std::size_t storage_size(pos3<std::size_t> const &block_size) {
            return block_storage_size<Extent>(block_size) * omp_get_max_threads();
        }

        using strides_t =
            hymap::keys<dim::i, dim::k, dim::j, thread_dim>::values<integral_constant<int_t, 1>, int_t, int_t, int_t>;

        template <class Extent>
        strides_t strides(pos3<std::size_t> const &block_size) {
            // auto bs = full_block_size<Extent>(block_size);
            // return {{}, bs.i, bs.i * bs.k, bs.i * bs.k * bs.j};
            return {};
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
    auto make_tmp_storage_mc(Allocator &allocator, pos3<std::size_t> const &block_size) GT_AUTO_RETURN(
        sid::synthetic().set<sid::property::origin>(allocator.template allocate<T>(_impl_tmp_mc::storage_size<Extent>(
            block_size))) /*.set<sid::property::strides>(_impl_tmp_mc::strides<Extent>(block_size))*/);

} // namespace gridtools
