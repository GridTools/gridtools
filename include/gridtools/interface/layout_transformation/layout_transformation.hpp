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

#include "layout_transformation_impl_omp.hpp"

#ifdef __CUDACC__

#include "../../common/cuda_is_ptr.hpp"
#include "layout_transformation_impl_cuda.hpp"

namespace gridtools {
    namespace interface {
        template <class T, class Dims, class DstStrides, class SrcSrides>
        void transform(T *dst,
            T const *__restrict__ src,
            Dims const &dims,
            DstStrides const &dst_strides,
            SrcSrides const &src_strides) {
            assert(dst);
            assert(src);
            assert(dims.size() == dst_strides.size());
            assert(src_strides.size() == dst_strides.size());
            assert(is_gpu_ptr(dst) == is_gpu_ptr(src));
            if (is_gpu_ptr(dst))
                impl::transform_cuda_loop(dst, src, dims, dst_strides, src_strides);
            else
                impl::transform_openmp_loop(dst, src, dims, dst_strides, src_strides);
        }
    } // namespace interface
} // namespace gridtools

#else

namespace gridtools {
    namespace interface {
        template <class T, class Dims, class DstStrides, class SrcSrides>
        void transform(T *dst,
            T const *__restrict__ src,
            Dims const &dims,
            DstStrides const &dst_strides,
            SrcSrides const &src_strides) {
            assert(dst);
            assert(src);
            assert(dims.size() == dst_strides.size());
            assert(src_strides.size() == dst_strides.size());
            impl::transform_openmp_loop(dst, src, dims, dst_strides, src_strides);
        }
    } // namespace interface
} // namespace gridtools

#endif
