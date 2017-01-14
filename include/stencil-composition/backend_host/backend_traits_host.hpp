/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "empty_iterate_domain_cache.hpp"
#include "iterate_domain_host.hpp"
#include "run_esf_functor_host.hpp"
#include "strategy_host.hpp"

#ifdef ENABLE_METERS
#include "timer_host.hpp"
#else
#include "../timer_dummy.hpp"
#endif

/**@file
@brief type definitions and structures specific for the Host backend
*/
namespace gridtools {
    namespace _impl_host {
        /**forward declaration*/
        template < typename Arguments >
        struct run_functor_host;
    }

    /**Traits struct, containing the types which are specific for the host backend*/
    template <>
    struct backend_traits_from_id< enumtype::Host > {

        template < typename T >
        static T extract_storage_info_ptr(T t) {
            return t;
        }

        template < typename AggregatorType >
        struct instantiate_view {

            AggregatorType &m_agg;
            instantiate_view(AggregatorType &agg) : m_agg(agg) {}

            template < typename T, typename Arg = typename boost::fusion::result_of::first< T >::type >
            arg_storage_pair< Arg, typename Arg::storage_t > get_arg_storage_pair() const {
                return boost::fusion::deref(boost::fusion::find< arg_storage_pair< Arg, typename Arg::storage_t > >(
                    m_agg.get_arg_storage_pairs()));
            }

            template < typename T, typename Arg = typename boost::fusion::result_of::first< T >::type >
            typename boost::enable_if< is_data_store< typename Arg::storage_t >, void >::type operator()(T &t) const {
                // make a view
                if(get_arg_storage_pair< T >().ptr.get())
                    t = make_host_view(*(get_arg_storage_pair< T >().ptr));
            }

            template < typename T, typename Arg = typename boost::fusion::result_of::first< T >::type >
            typename boost::enable_if< is_data_store_field< typename Arg::storage_t >, void >::type operator()(
                T &t) const {
                // make a view
                if(get_arg_storage_pair< T >().ptr.get())
                    t = make_field_host_view(*(get_arg_storage_pair< T >().ptr));
            }
        };

        template < typename Arguments >
        struct execute_traits {
            typedef _impl_host::run_functor_host< Arguments > run_functor_t;
        };

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the i-direction, in a 2D
            grid of threads.
        */
        static uint_t n_i_pes(uint_t = 0) {
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        }

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the j-direction, in a 2D
            grid of threads.
        */
        static uint_t n_j_pes(uint_t = 0) { return 1; }

        /** This is the function used by the specific backend
         *  that determines the i coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t processing_element_i() {
#ifdef _OPENMP
            return omp_get_thread_num();
#else
            return 0;
#endif
        }

        /** This is the function used by the specific backend
         *  that determines the j coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t processing_element_j() { return 0; }

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
           Static method in order to calculate the field offset. 

           We only use the offset in i at this stage. Why?
            __ __   Here we have two blocks. No matter what the halo is we want
           |  |  |  to end up in the left top position of each element,
           |__|__|  and therefore we don't consider the j offset at the moment.
                    The real offset is the offset calculated here + offset halo i + offset halo j!
        */
        template <typename LocalDomain, typename PEBlockSize, bool Tmp, typename MaxExtent, typename StorageInfo>
        static typename boost::enable_if_c<Tmp, int>::type 
        fields_offset(StorageInfo const* sinfo) {
            const uint_t i = processing_element_i();
            constexpr int diff_i_minus = boost::mpl::at_c<MaxExtent, 0>::type::value;
            constexpr int diff_i_plus = boost::mpl::at_c<MaxExtent, 1>::type::value;
            constexpr int diff_j_minus = boost::mpl::at_c<MaxExtent, 2>::type::value;
            constexpr int blocksize = (diff_i_minus + PEBlockSize::i_size_t::value + diff_i_plus);
            return  StorageInfo::get_initial_offset() + 
                sinfo->template stride<0>() * (blocksize * i - diff_i_minus) - 
                sinfo->template stride<1>() * diff_j_minus;
        }

        template <typename LocalDomain, typename PEBlockSize, bool Tmp, typename MaxExtent, typename StorageInfo>
        static typename boost::enable_if_c<!Tmp, int>::type 
        fields_offset(StorageInfo const* sinfo) {
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

            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArgs >::value), "Internal Error: wrong type");
            template < typename LocalDomain, typename Grid, typename ReductionData >
            static void run(LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const uint_t bi,
                const uint_t bj) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), "Internal Error: wrong type");

                // each strategy executes a different high level loop for a mss
                strategy_from_id_host< backend_ids_t::s_strategy_id >::template mss_loop<
                    RunFunctorArgs >::template run(local_domain, grid, reduction_data, bi, bj);
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        struct mss_fuse_esfs_strategy {
            typedef boost::mpl::bool_< false > type;
            BOOST_STATIC_CONSTANT(bool, value = (type::value));
        };

        // high level metafunction that contains the run_esf_functor corresponding to this backend
        typedef boost::mpl::quote2< run_esf_functor_host > run_esf_functor_h_t;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template < typename BackendIds >
        struct select_strategy {
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), "Error");
            typedef strategy_from_id_host< BackendIds::s_strategy_id > type;
        };

        template < enumtype::strategy StrategyId >
        struct get_block_size {
            typedef typename strategy_from_id_host< StrategyId >::block_size_t type;
        };

        /**
         * @brief metafunction that returns the right iterate domain
         * (depending on whether the local domain is positional or not)
         * @param IterateDomainArguments the iterate domain arguments
         * @return the iterate domain type for this backend
         */
        template < typename IterateDomainArguments >
        struct select_iterate_domain {
            GRIDTOOLS_STATIC_ASSERT(
                (is_iterate_domain_arguments< IterateDomainArguments >::value), "Internal Error: wrong type");
// indirection in order to avoid instantiation of both types of the eval_if
#ifdef STRUCTURED_GRIDS
            template < typename _IterateDomainArguments >
            struct select_positional_iterate_domain {
                typedef iterate_domain_host< positional_iterate_domain, _IterateDomainArguments > type;
            };
#endif

            template < typename _IterateDomainArguments >
            struct select_basic_iterate_domain {
                typedef iterate_domain_host< iterate_domain, _IterateDomainArguments > type;
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
        typedef timer_host performance_meter_t;
#else
        typedef timer_dummy performance_meter_t;
#endif
    };

} // namespace gridtools
