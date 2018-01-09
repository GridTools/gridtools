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
#include "multi_iterator.hpp"
#include "layout_transformation_config.hpp"
#include "layout_transformation_helper.hpp"
#include "../../storage/storage-facility.hpp"

#include <vector>

namespace gridtools {
    namespace impl {
        template < typename DataType, typename StorageInfo, typename... HigherIndices >
        GT_FUNCTION_HOST void transform_openmp_loop_impl(DataType *dst,
            DataType *src,
            const StorageInfo &si_dst,
            const StorageInfo &si_src,
            const gridtools::array< gridtools::uint_t, GT_TRANSFORM_MAX_DIM > &a_dims,
            HigherIndices... higher_indices) // TODO protect ints
        {
            const uint_t n_i = a_dims[0];
            const uint_t n_j = a_dims[1];
            const uint_t n_k = a_dims[2];
#pragma omp parallel for collapse(3)
            for (int i = 0; i < n_i; ++i)
                for (int j = 0; j < n_j; ++j)
                    for (int k = 0; k < n_k; ++k) {
                        dst[si_dst.index(i, j, k, higher_indices...)] = src[si_src.index(i, j, k, higher_indices...)];
                    }
        }

        // c++11 helper (c++14 would use a lambda with "auto...")
        template < typename DataType, typename StorageInfo >
        struct transform_openmp_loop_impl_functor {
            transform_openmp_loop_impl_functor(DataType *dst,
                DataType *src,
                const StorageInfo &si_dst,
                const StorageInfo &si_src,
                const gridtools::array< gridtools::uint_t, GT_TRANSFORM_MAX_DIM > &a_dims)
                : dst(dst), src(src), si_dst(si_dst), si_src(si_src), a_dims(a_dims) {}

            template < typename... OuterIndices >
            void operator()(OuterIndices... outer_indices) {
                transform_openmp_loop_impl(dst, src, si_dst, si_src, a_dims, outer_indices...);
            }

          private:
            DataType *dst;
            DataType *src;
            const StorageInfo &si_dst;
            const StorageInfo &si_src;
            const gridtools::array< gridtools::uint_t, GT_TRANSFORM_MAX_DIM > &a_dims;
        };

        template < typename DataType >
        void transform_openmp_loop(DataType *dst,
            DataType *src,
            const std::vector< uint_t > &dims,
            const std::vector< uint_t > &dst_strides,
            const std::vector< uint_t > &src_strides) {

            if (dims.size() > GT_TRANSFORM_MAX_DIM)
                throw std::runtime_error("Reached compile time GT_TRANSFORM_MAX_DIM in layout transformation. Increase "
                                         "the value for higher dimensional transformations.");

            using dummy_layout_map =
                default_layout_map_t< GT_TRANSFORM_MAX_DIM >; // not used since we pass strides directly

            using storage_info = gridtools::storage_info_interface< 0, dummy_layout_map >;
            auto a_dims = impl::vector_to_dims_array< GT_TRANSFORM_MAX_DIM >(dims);
            auto a_dst_strides = impl::vector_to_strides_array< GT_TRANSFORM_MAX_DIM >(dst_strides);
            auto a_src_strides = impl::vector_to_strides_array< GT_TRANSFORM_MAX_DIM >(src_strides);

            storage_info si_dst(a_dims, a_dst_strides);
            storage_info si_src(a_dims, a_src_strides);

            gridtools::array< gridtools::uint_t, GT_TRANSFORM_MAX_DIM - 3 > outer_dims;
            std::copy(a_dims.begin() + 3, a_dims.end(), outer_dims.begin());

            // c++14 version
            //            iterate(outer_dims,
            //                [&](auto... outer_indices) {
            //                    transform_openmp_loop_impl(dst, src, si_dst, si_src, a_dims, outer_indices...);
            //                });

            iterate(outer_dims,
                transform_openmp_loop_impl_functor< DataType, storage_info >(dst, src, si_dst, si_src, a_dims));
        }
    }
}
