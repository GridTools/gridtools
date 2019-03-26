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

namespace gridtools {

    template <class T>
    class PtrHolder {
      private:
        int_t m_offset;

      public:
        GT_DEVICE T *operator()() const {
            extern __shared__ T shm[];
            return shm + m_offset;
        }
    };

#ifndef GT_ICOSAHEDRAL_GRIDS
    template <class T, int_t ISize, int_t JSize, int_t IZero, int_t JZero>
    class sid_ij_cache {
      private:
        int_t m_allocation;

        static constexpr int_t i_stride = 1;
        static constexpr int_t j_stride = i_stride * ISize;
        static constexpr int_t k_stride = j_stride * JSize;

        using StrideMap = hymap::keys<dim::i, dim::j, dim::k>::values<integral_constant<int_t, i_stride>,
            integral_constant<int_t, j_stride>,
            integral_constant<int_t, k_stride>>;

      public:
        template <typename Allocator>
        sid_ij_cache(Allocator &&allocator) : m_allocation(allocator.template allocate<T>(k_stride)) {}

        friend PtrHolder<T> sid_get_origin(sid_ij_cache const &cache) {
            constexpr int_t offset = JZero * i_stride + IZero;
            return cache.m_allocation + offset;
        }
        friend StrideMap sid_get_strides(sid_ij_cache const &) { return {}; }
    };

#else
    template <class T, int_t NumColors, int_t ISize, int_t JSize, int_t IZero, int_t JZero>
    class sid_ij_cache {
      private:
        int_t m_allocation;

        static constexpr int_t i_stride = 1;
        static constexpr int_t c_stride = i_stride * NumColors;
        static constexpr int_t j_stride = c_stride * JSize;
        static constexpr int_t k_stride = j_stride * KSize;

        using StrideMap = hymap::keys<dim::i, dim::c, dim::j, dim::k>::values<integral_constant<int_t, i_stride>,
            integral_constant<int_t, c_stride>,
            integral_constant<int_t, j_stride>,
            integral_constant<int_t, k_stride>>;

      public:
        template <typename Allocator>
        sid_ij_cache(Allocator &&allocator) : m_allocation(allocator.allocate<T>(k_stride)) {}

        friend PtrHolder<T> sid_get_origin(sid_ij_cache const &cache) {
            constexpr int_t offset = JZero * stride_i + IZero;
            return cache.m_allocation + offset;
        }
        friend StrideMap sid_get_strides(sid_ij_cache const &) { return {}; }
    };
#endif

} // namespace gridtools
