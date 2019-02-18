/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <vector>

#include "../../common/cuda_is_ptr.hpp"
#include "../../common/defs.hpp"
#include "layout_transformation_impl_omp.hpp"

#ifdef __CUDACC__

#include "layout_transformation_impl_cuda.hpp"

namespace gridtools {
    namespace interface {
        template <typename DataType>
        void transform(DataType *dst,
            DataType *src,
            const std::vector<uint_t> &dims,
            const std::vector<uint_t> &dst_strides,
            const std::vector<uint_t> &src_strides) {
            bool use_cuda = is_gpu_ptr(dst);
            if (use_cuda != is_gpu_ptr(src))
                throw std::runtime_error("transform(): source and destination pointers need to be from the same memory "
                                         "space (both host or both gpu pointers)");
            if (use_cuda)
                impl::transform_cuda_loop(dst, src, dims, dst_strides, src_strides);
            else
                impl::transform_openmp_loop(dst, src, dims, dst_strides, src_strides);
        }
    } // namespace interface
} // namespace gridtools

#else

namespace gridtools {
    namespace interface {
        template <typename DataType>
        void transform(DataType *dst,
            DataType *src,
            const std::vector<uint_t> &dims,
            const std::vector<uint_t> &dst_strides,
            const std::vector<uint_t> &src_strides) {
            if (is_gpu_ptr(dst) || is_gpu_ptr(src))
                throw std::runtime_error("transform(): source and destination pointers need to be in the host space");
            impl::transform_openmp_loop(dst, src, dims, dst_strides, src_strides);
        }
    } // namespace interface
} // namespace gridtools

#endif
