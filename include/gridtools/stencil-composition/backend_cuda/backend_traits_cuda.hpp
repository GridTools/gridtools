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

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/quote.hpp>

#include "../../common/defs.hpp"
#include "../../storage/data_store.hpp"
#include "../../storage/data_store_field.hpp"
#include "../../storage/storage_cuda/data_field_view_helpers.hpp"
#include "../../storage/storage_cuda/data_view_helpers.hpp"

#include "../backend_traits_fwd.hpp"
#include "../grid_traits_fwd.hpp"
#include "../run_functor_arguments_fwd.hpp"
#include "execute_kernel_functor_cuda.hpp"
#include "iterate_domain_cache.hpp"
#include "run_esf_functor_cuda.hpp"
#include "strategy_cuda.hpp"

#ifdef ENABLE_METERS
#include "timer_cuda.hpp"
#else
#include "../timer_dummy.hpp"
#endif

/**@file
@brief type definitions and structures specific for the CUDA backend*/
namespace gridtools {

    /**forward declaration*/

    template <template <class> class IterateDomainBase, typename IterateDomainArguments>
    class iterate_domain_cuda;

    template <typename IterateDomainImpl>
    struct positional_iterate_domain;

    template <typename IterateDomainImpl>
    struct iterate_domain;

    /** @brief traits struct defining the types which are specific to the CUDA backend*/
    template <>
    struct backend_traits_from_id<enumtype::Cuda> {

        /** This is the functor used to generate view instances. According to the given storage (data_store,
           data_store_field) an appropriate view is returned. When using the CUDA backend we return device view
           instances.
        */
        struct make_view_f {
            template <typename S, typename SI>
            auto operator()(data_store<S, SI> const &src) const GT_AUTO_RETURN(make_device_view(src));
            template <typename S, uint_t... N>
            auto operator()(data_store_field<S, N...> const &src) const GT_AUTO_RETURN(make_field_device_view(src));
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
            template <typename LocalDomain, typename Grid, typename ReductionData, typename ExecutionInfo>
            static void run(
                LocalDomain &local_domain, const Grid &grid, ReductionData &reduction_data, ExecutionInfo &&) {
                typedef typename kernel_functor_executor<backend_ids_t, RunFunctorArgs>::type kernel_functor_executor_t;
                kernel_functor_executor_t(local_domain, grid)();
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        typedef std::true_type mss_fuse_esfs_strategy;

        // high level metafunction that contains the run_esf_functor corresponding to this backend
        typedef boost::mpl::quote2<run_esf_functor_cuda> run_esf_functor_h_t;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template <typename BackendIds>
        struct select_strategy {
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
            typedef strategy_from_id_cuda<BackendIds::s_strategy_id> type;
        };

        /**
         * @brief metafunction that returns the right iterate domain for this backend
         * (depending on whether the local domain is positional or not)
         * @tparam IterateDomainArguments the iterate domain arguments
         * @return the iterate domain type for this backend
         */
        template <typename IterateDomainArguments>
        struct select_iterate_domain {
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), GT_INTERNAL_ERROR);
            // indirection in order to avoid instantiation of both types of the eval_if
            template <typename _IterateDomainArguments>
            struct select_positional_iterate_domain {
// TODO to do this properly this should belong to a arch_grid_trait (i.e. a trait dispatching types depending
// on the comp architecture and the grid.
#ifdef STRUCTURED_GRIDS
                typedef iterate_domain_cuda<positional_iterate_domain, _IterateDomainArguments> type;
#else
                typedef iterate_domain_cuda<iterate_domain, _IterateDomainArguments> type;
#endif
            };

            template <typename _IterateDomainArguments>
            struct select_basic_iterate_domain {
                typedef iterate_domain_cuda<iterate_domain, _IterateDomainArguments> type;
            };

            typedef
                typename boost::mpl::eval_if<local_domain_is_stateful<typename IterateDomainArguments::local_domain_t>,
                    select_positional_iterate_domain<IterateDomainArguments>,
                    select_basic_iterate_domain<IterateDomainArguments>>::type type;
        };

        template <typename IterateDomainArguments>
        struct select_iterate_domain_cache {
            typedef iterate_domain_cache<IterateDomainArguments> type;
        };

#ifdef ENABLE_METERS
        typedef timer_cuda performance_meter_t;
#else
        typedef timer_dummy performance_meter_t;
#endif
    };

} // namespace gridtools
