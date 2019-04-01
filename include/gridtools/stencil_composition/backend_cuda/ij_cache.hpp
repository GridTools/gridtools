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
#include "../sid/synthetic.hpp"
#include "shared_allocator.hpp"

#include <iostream>

namespace gridtools {
#ifndef GT_ICOSAHEDRAL_GRIDS

    namespace ij_cache_impl_ {
        template <int_t IStride, int_t JStride>
        using strides_map_t =
            hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, IStride>, integral_constant<int_t, JStride>>;

        template <class T,
            int_t ISize,
            int_t JSize,
            int_t IZero,
            int_t JZero,
            int_t IStride = 1,
            int_t JStride = IStride *ISize,
            int_t Size = ISize *JSize,
            int_t Offset = IStride *IZero + JStride *JZero>
        auto make_ij_cache_helper(shared_allocator &allocator) GT_AUTO_RETURN(
            sid::synthetic()
                .template set<sid::property::origin>(allocator.template allocate<T>(Size) + Offset)
                .template set<sid::property::strides>(ij_cache_impl_::strides_map_t<IStride, JStride>{}));

    } // namespace ij_cache_impl_

    template <class T, int_t ISize, int_t JSize, int_t IZero, int_t JZero>
    auto make_ij_cache(shared_allocator &allocator)
        GT_AUTO_RETURN((ij_cache_impl_::make_ij_cache_helper<T, ISize, JSize, IZero, JZero>(allocator)));
#else
#endif

#ifdef asdfweroji
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
