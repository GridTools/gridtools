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

#include "../../common/numerics.hpp"
#include "../backend_traits_fwd.hpp"
#include "../block_size.hpp"
#include "iterate_domain_cuda.hpp"
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
    namespace _impl_cuda {
        template < typename Arguments >
        struct run_functor_cuda;
    }

    /** @brief traits struct defining the types which are specific to the CUDA backend*/
    template <>
    struct backend_traits_from_id< enumtype::Cuda > {

        /** This is the function used to extract a pointer out of a given storage info.
            In the case of CUDA we have to retrieve the GPU pointer.
        */
        template < typename StorageInfoPtr >
        static StorageInfoPtr extract_storage_info_ptr(StorageInfoPtr t) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_storage_info< typename boost::decay< decltype(*t) >::type >::value), GT_INTERNAL_ERROR);
            return t->get_gpu_ptr();
        }

        /** This is the functor used to generate view instances. According to the given storage (data_store,
           data_store_field) an appropriate view is returned. When using the CUDA backend we return device view
           instances.
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
                t = make_device_view(get_arg_storage_pair< ViewFusionMapElem >().m_value);
            }

            // specialization for creating view instance for data store fields
            template < typename ViewFusionMapElem,
                typename Arg = typename boost::fusion::result_of::first< ViewFusionMapElem >::type >
            typename boost::enable_if< is_data_store_field< typename Arg::data_store_t >, void >::type operator()(
                ViewFusionMapElem &t) const {
                GRIDTOOLS_STATIC_ASSERT((is_arg< Arg >::value), GT_INTERNAL_ERROR);
                // make a view
                t = make_field_device_view(get_arg_storage_pair< ViewFusionMapElem >().m_value);
            }
        };

        template < typename Arguments >
        struct execute_traits {
            typedef _impl_cuda::run_functor_cuda< Arguments > run_functor_t;
        };

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of processing elements in the i-direction (i.e. cuda blocks
            in the CUDA backend), in a 2D grid of threads.
        */
        static uint_t n_i_pes(const uint_t i_size) {
            typedef typename strategy_from_id_cuda< enumtype::Block >::block_size_t block_size_t;
            return (i_size + block_size_t::i_size_t::value) / block_size_t::i_size_t::value;
        }

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of processing elements in the j-direction (i.e. cuda blocks
            in the CUDA backend), in a 2D grid of threads.
        */
        static uint_t n_j_pes(const uint_t j_size) {
            typedef typename strategy_from_id_cuda< enumtype::Block >::block_size_t block_size_t;
            return (j_size + block_size_t::j_size_t::value) / block_size_t::j_size_t::value;
        }

        /** This is the function used by the specific backend
         *  that determines the i coordinate of a processing element.
         *  In the case of CUDA, a processing element is equivalent to a CUDA block
         */
        GT_FUNCTION
        static uint_t processing_element_i() { return blockIdx.x; }

        /** This is the function used by the specific backend
         *  that determines the j coordinate of a processing element.
         *  In the case of CUDA, a processing element is equivalent to a CUDA block
         */
        GT_FUNCTION
        static uint_t processing_element_j() { return blockIdx.y; }

        /**
           @brief assigns the two given values using the given thread Id whithin the block
        */
        template < uint_t Id, typename BlockSize >
        struct once_per_block {
            GRIDTOOLS_STATIC_ASSERT((is_block_size< BlockSize >::value), GT_INTERNAL_ERROR);

            template < typename Left, typename Right >
            GT_FUNCTION static void assign(Left &l, Right const &r) {
                assert(blockDim.z == 1);
                const uint_t pe_elem = threadIdx.y * BlockSize::i_size_t::value + threadIdx.x;
                if (Id % (BlockSize::i_size_t::value * BlockSize::j_size_t::value) == pe_elem) {
                    l = (Left)r;
                }
            }
        };

        /**
           Static method in order to calculate the field offset. In the iterate domain we store one pointer per
           storage in the shared memory. In addition to this each CUDA thread stores an integer that indicates
           the offset of this pointer. For temporaries we use an oversized storage in order to have private halo
           regions for each block. This method calculates the offset for temporaries and takes the private halo and
           alignment information into account.
        */
        template < typename LocalDomain, typename PEBlockSize, typename Arg, typename GridTraits, typename StorageInfo >
        GT_FUNCTION static typename boost::enable_if_c< Arg::is_temporary, int >::type fields_offset(
            StorageInfo const *sinfo) {
            typedef GridTraits grid_traits_t;
            typedef typename LocalDomain::max_i_extent_t max_i_t;
            // halo in I and J direction
            constexpr int halo_i = StorageInfo::halo_t::template at< grid_traits_t::dim_i_t::value >();
            constexpr int halo_j = StorageInfo::halo_t::template at< grid_traits_t::dim_j_t::value >();
            // calculate the blocksize in I and J direction
            constexpr int block_size_i = 2 * max_i_t::value + PEBlockSize::i_size_t::value;
            constexpr int block_size_j = 2 * halo_j + PEBlockSize::j_size_t::value;

            // protect against div. by 0 and compute the distance between two blocks
            constexpr int diff_between_blocks =
                ((StorageInfo::alignment_t::value > 1)
                        ? _impl::static_ceil(static_cast< float >(block_size_i) / StorageInfo::alignment_t::value) *
                              StorageInfo::alignment_t::value
                        : block_size_i);

            // compute offset in I and J
            const uint_t i = processing_element_i() * diff_between_blocks + halo_i;
            const uint_t j = Arg::location_t::n_colors::value *
                             (diff_between_blocks * gridDim.x * processing_element_j() * block_size_j);
            // return field offset (Initial storage offset + Alignment correction value + I offset + J offset)
            return (int)StorageInfo::get_initial_offset() + i + j;
        }

        /**
           Static method in order to calculate the field offset. In the iterate domain we store one pointer per
           storage in the shared memory. In addition to this each CUDA thread stores an integer that indicates
           the offset of this pointer. This function computes the field offset for non temporary storages.
        */
        template < typename LocalDomain, typename PEBlockSize, typename Arg, typename GridTraits, typename StorageInfo >
        GT_FUNCTION static typename boost::enable_if_c< !Arg::is_temporary, int >::type fields_offset(
            StorageInfo const *sinfo) {
            return StorageInfo::get_initial_offset();
        }

        /**
         * @brief main execution of a mss.
         * @tparam RunFunctorArgs run functor arguments
         */
        template < typename RunFunctorArgs >
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArgs >::value), GT_INTERNAL_ERROR);
            template < typename LocalDomain, typename Grid, typename ReductionData >
            static void run(LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const uint_t bi,
                const uint_t bj) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);

                typedef grid_traits_from_id< backend_ids_t::s_grid_type_id > grid_traits_t;
                typedef
                    typename grid_traits_t::template with_arch< backend_ids_t::s_backend_id >::type arch_grid_traits_t;

                typedef typename arch_grid_traits_t::template kernel_functor_executor< RunFunctorArgs >::type
                    kernel_functor_executor_t;
                kernel_functor_executor_t(local_domain, grid, bi, bj)();
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
        typedef boost::mpl::quote2< run_esf_functor_cuda > run_esf_functor_h_t;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template < typename BackendIds >
        struct select_strategy {
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), GT_INTERNAL_ERROR);
            typedef strategy_from_id_cuda< BackendIds::s_strategy_id > type;
        };

        /**
         * @brief metafunction that returns the block size
         */
        template < enumtype::strategy StrategyId >
        struct get_block_size {
            GRIDTOOLS_STATIC_ASSERT(StrategyId == enumtype::Block, "For CUDA backend only Block strategy is supported");
            typedef typename strategy_from_id_cuda< StrategyId >::block_size_t type;
        };

        /**
         * @brief metafunction that returns the right iterate domain for this backend
         * (depending on whether the local domain is positional or not)
         * @tparam IterateDomainArguments the iterate domain arguments
         * @return the iterate domain type for this backend
         */
        template < typename IterateDomainArguments >
        struct select_iterate_domain {
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), GT_INTERNAL_ERROR);
            // indirection in order to avoid instantiation of both types of the eval_if
            template < typename _IterateDomainArguments >
            struct select_positional_iterate_domain {
// TODO to do this properly this should belong to a arch_grid_trait (i.e. a trait dispatching types depending
// on the comp architecture and the grid.
#ifdef STRUCTURED_GRIDS
                typedef iterate_domain_cuda< positional_iterate_domain, _IterateDomainArguments > type;
#else
                typedef iterate_domain_cuda< iterate_domain, _IterateDomainArguments > type;
#endif
            };

            template < typename _IterateDomainArguments >
            struct select_basic_iterate_domain {
                typedef iterate_domain_cuda< iterate_domain, _IterateDomainArguments > type;
            };

            typedef typename boost::mpl::eval_if<
                local_domain_is_stateful< typename IterateDomainArguments::local_domain_t >,
                select_positional_iterate_domain< IterateDomainArguments >,
                select_basic_iterate_domain< IterateDomainArguments > >::type type;
        };

        template < typename IterateDomainArguments >
        struct select_iterate_domain_cache {
            typedef iterate_domain_cache< IterateDomainArguments > type;
        };

#ifdef ENABLE_METERS
        typedef timer_cuda performance_meter_t;
#else
        typedef timer_dummy performance_meter_t;
#endif
    };

} // namespace gridtools
