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

#include <utility>

#include "../../common/functional.hpp"
#include "../backend_traits_fwd.hpp"
#include "../empty_iterate_domain_cache.hpp"

#include "../structured_grids/backend_mic/execute_kernel_functor_mic.hpp"
#include "../structured_grids/backend_mic/strategy_mic.hpp"

#ifdef ENABLE_METERS
#include "timer_mic.hpp"
#else
#include "../timer_dummy.hpp"
#endif

/**@file
@brief type definitions and structures specific for the Mic backend
*/
namespace gridtools {
    /**Traits struct, containing the types which are specific for the mic backend*/
    template <>
    struct backend_traits_from_id<platform::mc> {

        /** This is the functor used to generate view instances. According to the given storage an appropriate view is
         * returned. When using the Host backend we return host view instances.
         */
        struct make_view_f {
            template <typename S, typename SI>
            auto operator()(data_store<S, SI> const &src) const GT_AUTO_RETURN(make_host_view(src));
        };

        template <uint_t Id>
        struct once_per_block {
            template <typename Left, typename Right>
            GT_FUNCTION static void assign(Left &l, Right const &r) {
                l = r;
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            template <typename LocalDomain, typename Grid, typename ReductionData, typename ExecutionInfo>
            GT_FUNCTION static void run(LocalDomain const &local_domain,
                Grid const &grid,
                ReductionData &reduction_data,
                const ExecutionInfo &execution_info) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data<ReductionData>::value), GT_INTERNAL_ERROR);

                strgrid::execute_kernel_functor_mic<RunFunctorArgs>(local_domain, grid, reduction_data)(execution_info);
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        using mss_fuse_esfs_strategy = std::true_type;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template <typename BackendIds>
        struct select_strategy {
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
            typedef strategy_from_id_mic<typename BackendIds::strategy_id_t> type;
        };

#ifdef ENABLE_METERS
        typedef timer_mic performance_meter_t;
#else
        typedef timer_dummy performance_meter_t;
#endif
    };

} // namespace gridtools
