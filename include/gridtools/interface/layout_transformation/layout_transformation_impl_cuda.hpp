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

#include <algorithm>

#include "../../common/array.hpp"
#include "../../common/cuda_util.hpp"
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/accumulate.hpp"
#include "../../common/hypercube_iterator.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple_util.hpp"
#include "layout_transformation_config.hpp"

namespace gridtools {
    namespace impl {
        /**
         * @brief copy std::vector to (potentially bigger) gridtools::array
         */
        template <class Src>
        auto vector_to_array(Src const &src, uint_t init_value) {
            assert(GT_TRANSFORM_MAX_DIM >= src.size() && "array too small");

            array<uint_t, GT_TRANSFORM_MAX_DIM> res;
            std::fill(res.begin(), res.end(), init_value);
            std::copy(src.begin(), src.end(), res.begin());
            return res;
        }

        // compile-time block size due to HIP-Clang bug https://github.com/ROCm-Developer-Tools/HIP/issues/1283
        using block_size_1d_t = integral_constant<int_t, 8>;

        template <class Strides, class Indices, size_t... Dims>
        __device__ auto index_impl(Strides const &strides, Indices const &indices, std::index_sequence<Dims...>) {
            return accumulate(plus_functor(), (strides[Dims + 3] * indices[Dims])...);
        }

        template <class Strides, class Indices>
        __device__ auto index(Strides const &strides, Indices const &indices) {
            return index_impl(strides, indices, std::make_index_sequence<GT_TRANSFORM_MAX_DIM - 3>());
        }

        template <class T, class Dims, class DstStrides, class SrcSrtides, class OuterDims>
        __global__ void transform_cuda_loop_kernel(T *raw_dst,
            T const *raw_src,
            Dims dims,
            DstStrides dst_strides,
            SrcSrtides src_strides,
            OuterDims outer_dims) {

            int i = blockIdx.x * block_size_1d_t::value + threadIdx.x;
            if (i >= dims[0])
                return;
            int j = blockIdx.y * block_size_1d_t::value + threadIdx.y;
            if (j >= dims[1])
                return;
            int k = blockIdx.z * block_size_1d_t::value + threadIdx.z;
            if (k >= dims[2])
                return;

            T *dst = raw_dst + i * dst_strides[0] + j * dst_strides[1] + k * dst_strides[2];
            T const *src = raw_src + i * src_strides[0] + j * src_strides[1] + k * src_strides[2];

            // this can be optimized but it is not as bad as it looks as one of the memories is coalescing (assuming one
            // of the layouts is a suitable gpu layout...)

            // TODO this range-based loop does not work on daint in release mode
            // for (auto &&outer : make_hypercube_view(outer_dims)) {
            auto &&hyper = make_hypercube_view(outer_dims);
            for (auto &&outer = hyper.begin(); outer != hyper.end(); ++outer)
                dst[index(dst_strides, *outer)] = src[index(src_strides, *outer)];
        }

        template <class T, class Dims, class DstStrides, class SrcSrides>
        void transform_cuda_loop(T *dst,
            T const *__restrict__ src,
            Dims const &dims,
            DstStrides const &dst_strides,
            SrcSrides const &src_strides) {
            auto a_dims = vector_to_array(dims, 1);
            array<size_t, GT_TRANSFORM_MAX_DIM - 3> outer_dims;
            std::copy(a_dims.begin() + 3, a_dims.end(), outer_dims.begin());

            dim3 grid_size((a_dims[0] + block_size_1d_t::value - 1) / block_size_1d_t::value,
                (a_dims[1] + block_size_1d_t::value - 1) / block_size_1d_t::value,
                (a_dims[2] + block_size_1d_t::value - 1) / block_size_1d_t::value);
            dim3 block_size(block_size_1d_t::value, block_size_1d_t::value, block_size_1d_t::value);

            transform_cuda_loop_kernel<<<grid_size, block_size>>>(
                dst, src, a_dims, vector_to_array(dst_strides, 0), vector_to_array(src_strides, 0), outer_dims);

#ifndef NDEBUG
            GT_CUDA_CHECK(cudaDeviceSynchronize());
#else
            GT_CUDA_CHECK(cudaGetLastError());
#endif
        }
    } // namespace impl
} // namespace gridtools
