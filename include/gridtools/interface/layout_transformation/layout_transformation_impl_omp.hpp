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

#include <cassert>
#include <cstddef>

namespace gridtools {
    namespace impl {
        template <class T, class Dims, class DstStrides, class SrcSrides>
        void transform_openmp_loop(T *dst,
            T const *__restrict__ src,
            Dims const &dims,
            DstStrides const &dst_strides,
            SrcSrides const &src_strides) {

            size_t dims_size = dims.size();

            auto nth_dim = [&dims](int n) -> int { return n < dims.size() ? dims[n] : 1; };
            auto nth_stride = [](int n, auto const &strides) -> int { return n < strides.size() ? strides[n] : 0; };

            auto omp_loop = [size_i = nth_dim(0),
                                size_j = nth_dim(1),
                                size_k = nth_dim(2),
                                src_stride_i = nth_stride(0, src_strides),
                                src_stride_j = nth_stride(1, src_strides),
                                src_stride_k = nth_stride(2, src_strides),
                                dst_stride_i = nth_stride(0, dst_strides),
                                dst_stride_j = nth_stride(1, dst_strides),
                                dst_stride_k = nth_stride(2, dst_strides)](T *dst, T const *__restrict__ src) {
#pragma omp parallel for collapse(3)
                for (int i = 0; i < size_i; ++i)
                    for (int j = 0; j < size_j; ++j)
                        for (int k = 0; k < size_k; ++k)
                            dst[dst_stride_i * i + dst_stride_j * j + dst_stride_k * k] =
                                src[src_stride_i * i + src_stride_j * j + src_stride_k * k];
            };

            size_t outer_total_size = 1;
            for (size_t d = 3; d < dims_size; ++d)
                outer_total_size *= dims[d];

            auto offset = [dims_size, &dims](size_t index, auto const &strides) {
                size_t res = 0;
                for (size_t d = 3; d < dims_size; ++d) {
                    res += index % dims[d] * strides[d];
                    index /= dims[d];
                }
                return res;
            };

            for (size_t index = 0; index != outer_total_size; ++index)
                omp_loop(dst + offset(index, dst_strides), src + offset(index, src_strides));
        }
    } // namespace impl
} // namespace gridtools
