/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../dim.hpp"
#include "../sid/concept.hpp"
#include "shared_allocator.hpp"

#include <iostream>

namespace gridtools {

    template <class T>
    class ptr_holder {
      private:
        int_t m_offset; // in bytes

      public:
        ptr_holder(int_t offset) : m_offset(offset) {}
        GT_DEVICE T *operator()() const {
            extern __shared__ char ij_cache_shm[];
            return reinterpret_cast<T *>(ij_cache_shm + m_offset);
        }
    };

#ifndef GT_ICOSAHEDRAL_GRIDS
    template <class T, int_t ISize, int_t JSize, int_t IZero, int_t JZero>
    class sid_ij_cache {
      private:
        int_t m_allocation; // in bytes

        static constexpr int_t i_stride = 1;
        static constexpr int_t j_stride = i_stride * ISize;
        static constexpr int_t size = j_stride * JSize;

        using stride_map_t =
            hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, i_stride>, integral_constant<int_t, j_stride>>;

      public:
        template <typename Allocator>
        sid_ij_cache(Allocator &allocator) : m_allocation(allocator.template allocate<sizeof(T)>(size * sizeof(T))) {}

        friend ptr_holder<T> sid_get_origin(sid_ij_cache &cache) {
            constexpr int_t offset = IZero * i_stride + JZero * j_stride;
            return cache.m_allocation + offset * sizeof(T);
        }
        friend stride_map_t sid_get_strides(sid_ij_cache const &) { return {}; }
    };

#else
    template <class T, int_t ISize, int_t NumColors, int_t JSize, int_t IZero, int_t JZero>
    class sid_ij_cache {
      private:
        int_t m_allocation; // in bytes

        static constexpr int_t c_stride = i_stride * NumColors;
        static constexpr int_t j_stride = c_stride * JSize;
        static constexpr int_t size = j_stride * JSize;

        using StrideMap = hymap::keys<dim::i, dim::c, dim::j>::values<integral_constant<int_t, i_stride>,
            integral_constant<int_t, c_stride>,
            integral_constant<int_t, j_stride>>;

      public:
        template <typename Allocator>
        sid_ij_cache(Allocator &allocator) : m_allocation(allocator.template allocate<sizeof(T)>(size * sizeof(T))) {}

        friend ptr_holder<T> sid_get_origin(sid_ij_cache &cache) {
            constexpr int_t offset = IZero * i_stride + JZero * j_stride;
            return cache.m_allocation + offset * sizeof(T);
        }
        friend stride_map_t sid_get_strides(sid_ij_cache const &) { return {}; }
    };
#endif

} // namespace gridtools
