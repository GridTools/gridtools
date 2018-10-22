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

#include <type_traits>

#include "../../common/gt_assert.hpp"
#include "../common/storage_info_interface.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    /*
     * @brief The host storage info implementation.
     * @tparam Id unique ID that should be shared among all storage infos with the same dimensionality.
     * @tparam Layout information about the memory layout
     * @tparam Halo information about the halo sizes (by default no halo is set)
     * @tparam Alignment information about the alignment (host_storage_info is not aligned by default)
     */
    template <uint_t Id,
        typename Layout,
        typename Halo = zero_halo<Layout::masked_length>,
        typename Alignment = alignment<1>>
    struct host_storage_info : storage_info_interface<Id, Layout, Halo, Alignment> {
        GRIDTOOLS_STATIC_ASSERT((is_halo<Halo>::value), "Given type is not a halo type.");
        GRIDTOOLS_STATIC_ASSERT((is_alignment<Alignment>::value), "Given type is not an alignment type.");

      public:
        static constexpr uint_t ndims = storage_info_interface<Id, Layout, Halo, Alignment>::ndims;

        /*
         * @brief host_storage_info constructor.
         * @param dims_ the dimensionality (e.g., 128x128x80)
         */
        template <typename... Dims,
            typename std::enable_if<sizeof...(Dims) == ndims && is_all_integral_or_enum<Dims...>::value, int>::type = 0>
        explicit constexpr host_storage_info(Dims... dims)
            : storage_info_interface<Id, Layout, Halo, Alignment>(dims...) {}

        /*
         * @brief host_storage_info constructor.
         * @param dims the dimensionality (e.g., 128x128x80)
         * @param strides the strides used to describe a layout of the data in memory
         */
        constexpr host_storage_info(array<uint_t, ndims> dims, array<uint_t, ndims> strides)
            : storage_info_interface<Id, Layout, Halo, Alignment>(dims, strides) {}
    };

    template <typename T>
    struct is_host_storage_info : std::false_type {};

    template <uint_t Id, typename Layout, typename Halo, typename Alignment>
    struct is_host_storage_info<host_storage_info<Id, Layout, Halo, Alignment>> : std::true_type {};

    /**
     * @}
     */
} // namespace gridtools
