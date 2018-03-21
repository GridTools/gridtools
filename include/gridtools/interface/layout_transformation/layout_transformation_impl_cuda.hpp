/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

#include "../../common/defs.hpp"
#include "../../common/layout_map_metafunctions.hpp"
#include "../../common/multi_iterator.hpp"
#include "../../storage/storage-facility.hpp"
#include "call_cuda_kernel.hpp"
#include "layout_transformation_config.hpp"
#include "layout_transformation_helper.hpp"

#include <vector>

namespace gridtools {
    namespace impl {
#ifdef __CUDACC__
        template < typename DataType, typename StorageInfo >
        struct transform_cuda_loop_kernel_functor {
            GT_FUNCTION transform_cuda_loop_kernel_functor(
                DataType *dst, DataType *src, const StorageInfo &si_dst, const StorageInfo &si_src, int i, int j, int k)
                : dst(dst), src(src), si_dst(si_dst), si_src(si_src), i(i), j(j), k(k) {}

            template < typename... OuterIndices >
            GT_FUNCTION typename std::enable_if< is_all_integral< OuterIndices... >::value >::type operator()(
                OuterIndices... outer_indices) {
                dst[si_dst.index(i, j, k, outer_indices...)] = src[si_src.index(i, j, k, outer_indices...)];
            }

          private:
            DataType *dst;
            DataType *src;
            const StorageInfo &si_dst;
            const StorageInfo &si_src;
            const int i;
            const int j;
            const int k;
        };

        template < typename DataType >
        __global__ void transform_cuda_loop_kernel(DataType *dst,
            DataType *src,
            gridtools::array< gridtools::uint_t, GT_TRANSFORM_MAX_DIM > dims,
            gridtools::array< gridtools::uint_t, GT_TRANSFORM_MAX_DIM > dst_strides,
            gridtools::array< gridtools::uint_t, GT_TRANSFORM_MAX_DIM > src_strides,
            gridtools::array< size_t, GT_TRANSFORM_MAX_DIM - 3 > outer_dims) {

            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= dims[0])
                return;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            if (j >= dims[1])
                return;
            int k = blockIdx.z * blockDim.z + threadIdx.z;
            if (k >= dims[2])
                return;

            using dummy_layout_map =
                gridtools::default_layout_map_t< GT_TRANSFORM_MAX_DIM >; // not used since we pass strides directly
            using storage_info = gridtools::storage_info_interface< 0, dummy_layout_map >;
            storage_info si_dst(dims, dst_strides);
            storage_info si_src(dims, src_strides);

            // this can be optimized but it is not as bad as it looks as one of the memories is coalescing (assuming one
            // of the layouts is a suitable gpu layout...)

            for (auto outer : make_hypercube_view(outer_dims)) {
                transform_cuda_loop_kernel_functor< DataType, storage_info > f(dst, src, si_dst, si_src, i, j, k);
                f(outer[0], outer[1]); // TODO make generic
            }

            //            make_multi_iterator(outer_dims)
            //                .iterate(
            //                    transform_cuda_loop_kernel_functor< DataType, storage_info >(dst, src, si_dst, si_src,
            //                    i, j, k));
        }
#endif

        template < typename DataType >
        void transform_cuda_loop(DataType *dst,
            DataType *src,
            const std::vector< uint_t > &dims,
            const std::vector< uint_t > &dst_strides,
            const std::vector< uint_t > &src_strides) {
#ifdef __CUDACC__
            int block_size_1d = 8;

            auto a_dims = impl::vector_to_dims_array< GT_TRANSFORM_MAX_DIM >(dims);
            gridtools::array< size_t, GT_TRANSFORM_MAX_DIM - 3 > outer_dims;
            std::copy(a_dims.begin() + 3, a_dims.end(), outer_dims.begin());

            dim3 grid_size((a_dims[0] + block_size_1d - 1) / block_size_1d,
                (a_dims[1] + block_size_1d - 1) / block_size_1d,
                (a_dims[2] + block_size_1d - 1) / block_size_1d);
            dim3 block_size(block_size_1d, block_size_1d, block_size_1d);

            CALL_CUDA_KERNEL(transform_cuda_loop_kernel,
                grid_size,
                block_size,
                dst,
                src,
                a_dims,
                impl::vector_to_strides_array< GT_TRANSFORM_MAX_DIM >(dst_strides),
                impl::vector_to_strides_array< GT_TRANSFORM_MAX_DIM >(src_strides),
                outer_dims);
#ifndef NDEBUG
            {
                cudaDeviceSynchronize();
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    fprintf(stderr, "CUDA ERROR: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
                    exit(-1);
                }
            }
#endif

#else
            throw std::runtime_error("calling CUDA transformation, but not compiled with CUDA support");
#endif
        }
    }
}
