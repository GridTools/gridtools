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

#include <tuple>

#include "../level.hpp"
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include <gridtools.hpp>

#include "../mss_functor.hpp"
#include "../sfinae.hpp"
#include "../tile.hpp"
#include "../../common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "execute_kernel_functor_cuda.hpp"

namespace gridtools {

    template < enumtype::strategy >
    struct strategy_from_id_cuda;

    /**
       @brief specialization for the \ref enumtype::Naive strategy
    */
    template <>
    struct strategy_from_id_cuda< enumtype::Naive > {};

    /**
     * @brief struct holding backend-specific runtime information about stencil execution.
     * Empty for the CUDA backend.
     */
    struct execution_info_cuda {};

    /**
       @brief specialization for the \ref enumtype::Block strategy
       Empty as not used in the CUDA backend
    */
    template <>
    struct strategy_from_id_cuda< enumtype::Block > {
        // default block size for Block strategy
        typedef block_size< GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J, 1 > block_size_t;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendIds backend ids type
         */
        template < typename MssComponents, typename BackendIds, typename ReductionData >
        struct fused_mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssComponents, is_mss_components >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((meta::is_instantiation_of< std::tuple, MssComponents >::value), GT_INTERNAL_ERROR);

            template < typename LocalDomainListArray, typename Grid >
            static void run(
                LocalDomainListArray const &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);

                host_for_each< GT_META_CALL(meta::make_indices_for, MssComponents) >(mss_functor< MssComponents,
                    Grid,
                    LocalDomainListArray,
                    BackendIds,
                    ReductionData,
                    execution_info_cuda >(local_domain_lists, grid, reduction_data, {}));
            }
        };
    };

} // namespace gridtools
