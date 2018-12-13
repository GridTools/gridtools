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
