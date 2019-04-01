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
            int_t Size = JStride *JSize,
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
    namespace ij_cache_impl_ {
        template <int_t IStride, int_t CStride, int_t JStride>
        using strides_map_t = hymap::keys<dim::i, dim::c, dim::j>::values<integral_constant<int_t, IStride>,
            integral_constant<int_t, CStride>,
            integral_constant<int_t, JStride>>;

        template <class T,
            int_t ISize,
            int_t NumColors,
            int_t JSize,
            int_t IZero,
            int_t JZero,
            int_t IStride = 1,
            int_t CStride = IStride *NumColors,
            int_t JStride = CStride *JSize,
            int_t Size = JStride *JSize,
            int_t Offset = IStride *IZero + JStride *JZero>
        auto make_ij_cache_helper(shared_allocator &allocator) GT_AUTO_RETURN(
            sid::synthetic()
                .template set<sid::property::origin>(allocator.template allocate<T>(Size) + Offset)
                .template set<sid::property::strides>(ij_cache_impl_::strides_map_t<IStride, CStride, JStride>{}));

    } // namespace ij_cache_impl_

    template <class T, int_t ISize, int_t NumColors, int_t JSize, int_t IZero, int_t JZero>
    auto make_ij_cache(shared_allocator &allocator)
        GT_AUTO_RETURN((ij_cache_impl_::make_ij_cache_helper<T, ISize, NumColors, JSize, IZero, JZero>(allocator)));
#endif

} // namespace gridtools
