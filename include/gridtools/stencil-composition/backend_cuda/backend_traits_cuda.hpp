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

#include <cuda_runtime.h>

#include "../../common/defs.hpp"
#include "../../storage/data_store.hpp"

#include "../backend_traits_fwd.hpp"
#include "../grid_traits_fwd.hpp"
#include "execute_kernel_functor_cuda.hpp"
#include "strategy_cuda.hpp"

#ifdef ENABLE_METERS
#include "timer_cuda.hpp"
#else
#include "../timer_dummy.hpp"
#endif

/**@file
@brief type definitions and structures specific for the CUDA backend*/
namespace gridtools {

    /** @brief traits struct defining the types which are specific to the CUDA backend*/
    template <>
    struct backend_traits_from_id<target::cuda> {

        /** This is the functor used to generate view instances. According to the given storage
           an appropriate view is returned. When using the CUDA backend we return device view instances.
        */
        struct make_view_f {
            template <typename S, typename SI>
            auto operator()(data_store<S, SI> const &src) const GT_AUTO_RETURN(make_device_view(src));
        };

        /**
           @brief assigns the two given values using the given thread Id whithin the block
        */
        template <uint_t Id>
        struct once_per_block {
            template <typename Left, typename Right>
            GT_FUNCTION static void assign(Left &l, Right const &r) {
                assert(blockDim.z == 1);
                if (Id % (blockDim.x * blockDim.y) == threadIdx.y * blockDim.x + threadIdx.x)
                    l = r;
            }
        };

        /**
         * @brief main execution of a mss.
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            template <typename LocalDomain, typename Grid, typename ExecutionInfo>
            static void run(LocalDomain &local_domain, const Grid &grid, ExecutionInfo &&) {
                typedef typename kernel_functor_executor<backend_ids_t, RunFunctorArgs>::type kernel_functor_executor_t;
                kernel_functor_executor_t(local_domain, grid)();
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        typedef std::true_type mss_fuse_esfs_strategy;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template <typename BackendIds>
        struct select_strategy {
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
            typedef strategy_from_id_cuda<typename BackendIds::strategy_id_t> type;
        };

#ifdef ENABLE_METERS
        typedef timer_cuda performance_meter_t;
#else
        typedef timer_dummy performance_meter_t;
#endif
    };

} // namespace gridtools
