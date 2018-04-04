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

#include "../../common/array.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../../common/variadic_pack_metafunctions.hpp"
#include "../../common/layout_map.hpp"
#include "storage_info_interface.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    /*
     * @brief A storage info runtime object, providing dimensions and strides but without type information of the
     * storage info. Can be used in interfaces where no strict type information is available, i.e. in the interface to
     * Fortran.
     */
    class storage_info_rt {
      private:
        std::vector< uint_t > dims_;
        std::vector< uint_t > unaligned_dims_;
        std::vector< uint_t > strides_;

      public:
        storage_info_rt(std::vector< uint_t > dims, std::vector< uint_t > unaligned_dims, std::vector< uint_t > strides)
            : dims_(dims), unaligned_dims_(unaligned_dims), strides_(strides) {}

        const std::vector< uint_t > &dims() const { return dims_; }
        const std::vector< uint_t > &unaligned_dims() const { return unaligned_dims_; }
        const std::vector< uint_t > &strides() const { return strides_; }
    };

    namespace {
        template < int Idx >
        struct unaligned_dim_getter {
            constexpr unaligned_dim_getter() {}

            template < typename StorageInfo >
            GT_FUNCTION static constexpr uint_t apply(const StorageInfo &storage_info_) {
                return storage_info_.template total_length< Idx >();
            }
        };

        template < template < int Idx > class Getter, typename StorageInfo >
        gridtools::array< uint_t, StorageInfo::layout_t::masked_length > make_array_from(
            const StorageInfo &storage_info) {
            using seq = gridtools::apply_gt_integer_sequence<
                typename gridtools::make_gt_integer_sequence< int, StorageInfo::layout_t::masked_length >::type >;
            return gridtools::array< uint_t, StorageInfo::layout_t::masked_length >(
                seq::template apply< gridtools::array< uint_t, StorageInfo::layout_t::masked_length >, Getter >(
                    storage_info));
        }
    }

    /*
     * @brief Construct a storage_info_rt from a storage_info
     */
    template < typename StorageInfo >
    storage_info_rt make_storage_info_rt(const StorageInfo &storage_info) {
        return storage_info_rt( //
            to_vector(storage_info.dims()),
            to_vector(make_unaligned_dims_array(storage_info)),
            to_vector(storage_info.strides()));
    }

    /*
     * @brief Constructs gridtools::array of unaligned_dims.
     */
    template < typename StorageInfo >
    gridtools::array< uint_t, StorageInfo::layout_t::masked_length > make_unaligned_dims_array(
        const StorageInfo &storage_info) {
        GRIDTOOLS_STATIC_ASSERT((gridtools::is_storage_info< StorageInfo >::value), "Expected a StorageInfo");
        return make_array_from< unaligned_dim_getter >(storage_info);
    }

    /**
     * @}
     */
}
