/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

#include <assert.h>

#include "../common/storage_info_interface.hpp"

namespace gridtools {

    /*
     * @brief The cuda storage info implementation.
     * @tparam Id unique ID that should be shared among all storage infos with the same dimensionality.
     * @tparam Layout information about the memory layout
     * @tparam Halo information about the halo sizes (by default no halo is set)
     * @tparam Alignment information about the alignment (cuda_storage_info is aligned to 32 by default)
     */
    template < unsigned Id,
        typename Layout,
        typename Halo = zero_halo< Layout::masked_length >,
        typename Alignment = alignment< 32 > >
    struct cuda_storage_info : storage_info_interface< Id, Layout, Halo, Alignment > {
      private:
        mutable cuda_storage_info< Id, Layout, Halo, Alignment > *m_gpu_ptr;

      public:
        /*
         * @brief cuda_storage_info constructor.
         * @param dims_ the dimensionality (e.g., 128x128x80)
         */
        template < typename... Dims >
        explicit constexpr cuda_storage_info(Dims... dims_)
            : storage_info_interface< Id, Layout, Halo, Alignment >(dims_...), m_gpu_ptr(nullptr) {
            static_assert(is_halo< Halo >::value, "Given type is not a halo type.");
            static_assert(is_alignment< Alignment >::value, "Given type is not an alignment type.");
            static_assert(boost::mpl::and_< boost::mpl::bool_< (sizeof...(Dims) > 0) >,
                              typename is_all_integral< Dims... >::type >::value,
                "Dimensions have to be integral types.");
        }

        /*
         * @brief cuda_storage_info destructor.
         */
        ~cuda_storage_info() = default;

        /*
         * @brief retrieve the device pointer. This information is needed when the storage information should be passed
         * to a kernel.
         * @return a storage info device pointer
         */
        cuda_storage_info< Id, Layout, Halo, Alignment > *get_gpu_ptr() const {
            if (!m_gpu_ptr) {
                cudaError_t err = cudaMalloc(&m_gpu_ptr, sizeof(cuda_storage_info< Id, Layout, Halo, Alignment >));
                assert((err == cudaSuccess) && "failed to allocate GPU memory.");
                err = cudaMemcpy((void *)m_gpu_ptr,
                    (void *)this,
                    sizeof(cuda_storage_info< Id, Layout, Halo, Alignment >),
                    cudaMemcpyHostToDevice);
                assert((err == cudaSuccess) && "failed to clone storage_info to the device.");
            }
            return m_gpu_ptr;
        }
    };

    template < typename T >
    struct is_cuda_storage_info : boost::mpl::false_ {};

    template < unsigned Id, typename Layout, typename Halo, typename Alignment >
    struct is_cuda_storage_info< cuda_storage_info< Id, Layout, Halo, Alignment > > : boost::mpl::true_ {};
}
