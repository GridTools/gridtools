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
#include <boost/mpl/for_each.hpp>

#include "../backend_traits_fwd.hpp"
#include "../block_size.hpp"
#include "../empty_iterate_domain_cache.hpp"
#include "iterate_domain_mic.hpp"
#include "run_esf_functor_mic.hpp"
#include "strategy_mic.hpp"

#ifdef ENABLE_METERS
#include "timer_mic.hpp"
#else
#include "../timer_dummy.hpp"
#endif

/**@file
@brief type definitions and structures specific for the Mic backend
*/
namespace gridtools {
    namespace _impl_mic {
        /**forward declaration*/
        template < typename Arguments >
        struct run_functor_mic;
    }

    /**Traits struct, containing the types which are specific for the mic backend*/
    template <>
    struct backend_traits_from_id< enumtype::Mic > {

        /** This is the function used to extract a pointer out of a given storage info.
            In the case of Mic backend we have to return the CPU pointer.
        */
        template < typename StorageInfoPtr >
        static StorageInfoPtr extract_storage_info_ptr(StorageInfoPtr t) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_storage_info< typename boost::decay< decltype(*t) >::type >::value), GT_INTERNAL_ERROR);
            return t;
        }

        /** This is the functor used to generate view instances. According to the given storage (data_store,
           data_store_field) an appropriate view is returned. When using the Mic backend we return host view instances.
        */
        template < typename AggregatorType >
        struct instantiate_view {
            GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< AggregatorType >::value), GT_INTERNAL_ERROR);

            AggregatorType const &m_agg;
            instantiate_view(AggregatorType &agg) : m_agg(agg) {}

            template < typename ViewFusionMapElem,
                typename Arg = typename boost::fusion::result_of::first< ViewFusionMapElem >::type >
            arg_storage_pair< Arg, typename Arg::data_store_t > const &get_arg_storage_pair() const {
                GRIDTOOLS_STATIC_ASSERT((is_arg< Arg >::value), GT_INTERNAL_ERROR);
                return boost::fusion::deref(boost::fusion::find< arg_storage_pair< Arg, typename Arg::data_store_t > >(
                    m_agg.get_arg_storage_pairs()));
            }

            // specialization for creating view instance for data stores
            template < typename ViewFusionMapElem,
                typename Arg = typename boost::fusion::result_of::first< ViewFusionMapElem >::type >
            typename boost::enable_if< is_data_store< typename Arg::data_store_t >, void >::type operator()(
                ViewFusionMapElem &t) const {
                GRIDTOOLS_STATIC_ASSERT((is_arg< Arg >::value), GT_INTERNAL_ERROR);
                // make a view
                t = make_host_view(get_arg_storage_pair< ViewFusionMapElem >().m_value);
            }

            // specialization for creating view instance for data store fields
            template < typename ViewFusionMapElem,
                typename Arg = typename boost::fusion::result_of::first< ViewFusionMapElem >::type >
            typename boost::enable_if< is_data_store_field< typename Arg::data_store_t >, void >::type operator()(
                ViewFusionMapElem &t) const {
                GRIDTOOLS_STATIC_ASSERT((is_arg< Arg >::value), GT_INTERNAL_ERROR);
                // make a view
                t = make_field_host_view(get_arg_storage_pair< ViewFusionMapElem >().m_value);
            }
        };

        template < typename Arguments >
        struct execute_traits {
            typedef _impl_mic::run_functor_mic< Arguments > run_functor_t;
        };

        template < uint_t Id, typename BlockSize >
        struct once_per_block {
            GRIDTOOLS_STATIC_ASSERT((is_block_size< BlockSize >::value), "Error: wrong type");

            template < typename Left, typename Right >
            GT_FUNCTION // inline
                static void
                assign(Left &l, Right const &r) {
                l = (Left)r;
            }
        };

        /**
           Static method in order to calculate the field offset. In the iterate domain we store one pointer per
           storage. In addition to this each OpenMP thread stores an integer that indicates the offset of this
           pointer. For temporaries we use an oversized storage in order to have private halo
           regions for each thread. This method calculates the offset for temporaries and takes the private halo and
           alignment information into account.
        */
        template < typename LocalDomain, typename PEBlockSize, typename Arg, typename GridTraits, typename StorageInfo >
        static typename boost::enable_if_c< Arg::is_temporary, int >::type fields_offset(StorageInfo const *sinfo) {
            const int thread = omp_get_thread_num();
            const int total_threads = omp_get_max_threads();
            const int thread_offset =
                (sinfo->padded_total_length() - StorageInfo::get_initial_offset()) * thread / total_threads;
            return StorageInfo::get_initial_offset() + thread_offset;
        }

        /**
           Static method in order to calculate the field offset. In the iterate domain we store one pointer per
           storage in the shared memory. In addition to this each OpenMP thread stores an integer that indicates
           the offset of this pointer. This function computes the field offset for non temporary storages.
        */
        template < typename LocalDomain, typename PEBlockSize, typename Arg, typename GridTraits, typename StorageInfo >
        static typename boost::enable_if_c< !Arg::is_temporary, int >::type fields_offset(StorageInfo const *sinfo) {
            return StorageInfo::get_initial_offset();
        }

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template < typename RunFunctorArgs >
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArgs >::value), GT_INTERNAL_ERROR);
            template < typename LocalDomain, typename Grid, typename ReductionData, typename ExecutionInfo >
            static void run(LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const ExecutionInfo &execution_info) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), GT_INTERNAL_ERROR);

                using grid_traits_t = grid_traits_from_id< backend_ids_t::s_grid_type_id >;
                using arch_grid_traits_t =
                    typename grid_traits_t::template with_arch< backend_ids_t::s_backend_id >::type;
                using kernel_functor_executor_t =
                    typename arch_grid_traits_t::template kernel_functor_executor< RunFunctorArgs >::type;

                kernel_functor_executor_t(local_domain, grid, reduction_data)(execution_info);
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        struct mss_fuse_esfs_strategy {
            typedef boost::mpl::bool_< true > type;
            BOOST_STATIC_CONSTANT(bool, value = (type::value));
        };

        // high level metafunction that contains the run_esf_functor corresponding to this backend
        typedef boost::mpl::quote2< run_esf_functor_mic > run_esf_functor_h_t;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template < typename BackendIds >
        struct select_strategy {
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), GT_INTERNAL_ERROR);
            typedef strategy_from_id_mic< BackendIds::s_strategy_id > type;
        };

        template < enumtype::strategy StrategyId >
        struct get_block_size {
            typedef typename strategy_from_id_mic< StrategyId >::block_size_t type;
        };

        /**
         * @brief metafunction that returns the right iterate domain
         * (depending on whether the local domain is positional or not)
         * @param IterateDomainArguments the iterate domain arguments
         * @return the iterate domain type for this backend
         */
        template < typename IterateDomainArguments >
        struct select_iterate_domain {
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), GT_INTERNAL_ERROR);
// indirection in order to avoid instantiation of both types of the eval_if
#ifdef STRUCTURED_GRIDS
            template < typename _IterateDomainArguments >
            struct select_positional_iterate_domain {
                typedef iterate_domain_mic< _IterateDomainArguments > type;
            };
#endif

            template < typename _IterateDomainArguments >
            struct select_basic_iterate_domain {
                typedef iterate_domain_mic< _IterateDomainArguments > type;
            };

            typedef typename boost::mpl::eval_if<
                local_domain_is_stateful< typename IterateDomainArguments::local_domain_t >,
#ifdef STRUCTURED_GRIDS
                select_positional_iterate_domain< IterateDomainArguments >,
#else
                select_basic_iterate_domain< IterateDomainArguments >,
#endif
                select_basic_iterate_domain< IterateDomainArguments > >::type type;
        };

        template < typename IterateDomainArguments >
        struct select_iterate_domain_cache {
            typedef empty_iterate_domain_cache type;
        };

#ifdef ENABLE_METERS
        typedef timer_mic performance_meter_t;
#else
        typedef timer_dummy performance_meter_t;
#endif
    };

} // namespace gridtools
