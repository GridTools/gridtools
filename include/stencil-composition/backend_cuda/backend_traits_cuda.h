#pragma once
#include <boost/mpl/for_each.hpp>
#include "../backend_traits_fwd.h"
#include "execute_kernel_functor_cuda.h"
#include "run_esf_functor_cuda.h"
#include "../block_size.h"
#include "iterate_domain_cuda.h"
#include "strategy_cuda.h"

/**@file
@brief type definitions and structures specific for the CUDA backend*/
namespace gridtools{

    /**forward declaration*/
    namespace _impl_cuda{
        template <typename Arguments>
        struct run_functor_cuda;
    }

    /**forward declaration*/
    template<typename T>
    struct hybrid_pointer;

    /** @brief traits struct defining the types which are specific to the CUDA backend*/
    template<>
    struct backend_traits_from_id< enumtype::Cuda >
    {

        template <typename T>
        struct pointer
        {
            typedef hybrid_pointer<T> type;
        };

        template <typename ValueType, typename Layout, bool Temp=false, short_t SpaceDim=1 >
        struct storage_traits
        {
            typedef storage< base_storage<typename pointer<ValueType>::type, Layout, Temp, SpaceDim> > storage_t;
        };

        template <typename Arguments>
        struct execute_traits
        {
            typedef _impl_cuda::run_functor_cuda<Arguments> run_functor_t;
        };

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of processing elements in the i-direction (i.e. cuda blocks
            in the CUDA backend), in a 2D grid of threads.
        */
        static uint_t n_i_pes(const int i_size) {
            typedef typename strategy_from_id_cuda<enumtype::Block>::block_size_t block_size_t;
            return (i_size + block_size_t::i_size_t::value) / block_size_t::i_size_t::value;
       }

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of processing elements in the j-direction (i.e. cuda blocks
            in the CUDA backend), in a 2D grid of threads.
        */
        static uint_t n_j_pes(const uint_t j_size) {
            typedef typename strategy_from_id_cuda<enumtype::Block>::block_size_t block_size_t;
            return (j_size + block_size_t::j_size_t::value) / block_size_t::j_size_t::value;
        }

        /** This is the function used by the specific backend
         *  that determines the i coordinate of a processing element.
         *  In the case of CUDA, a processing element is equivalent to a CUDA block
         */
        GT_FUNCTION
        static uint_t processing_element_i() {
            return blockIdx.x;
        }

        /** This is the function used by the specific backend
         *  that determines the j coordinate of a processing element.
         *  In the case of CUDA, a processing element is equivalent to a CUDA block
         */
        GT_FUNCTION
        static uint_t processing_element_j() {
            return blockIdx.y;
        }

        /**
           @brief assigns the two given values using the given thread Id whithin the block
        */
        template <uint_t Id>
        struct once_per_block {
            template<typename Left, typename Right>
            GT_FUNCTION
            static void assign(Left& l, Right const& r){
                //TODOCOSUNA if there are more ID than threads in a block????
                if(threadIdx.x==Id)
                    {
                        l=r;
                    }
            }
        };

        /**
         * @brief main execution of a mss.
         * @tparam RunFunctorArgs run functor arguments
         * @tparam StrategyId id of the strategy (ignored for the CUDA backend as for the moment there is only one way
         *     scheduling the work)
         */
        template<typename RunFunctorArgs, enumtype::strategy StrategyId>
        struct mss_loop
        {
            BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value));
            template<typename LocalDomain, typename Coords>
            static void run(LocalDomain& local_domain, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                BOOST_STATIC_ASSERT((is_local_domain<LocalDomain>::value));
                BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));

                execute_kernel_functor_cuda<RunFunctorArgs>(local_domain, coords, bi, bj)();
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        struct mss_fuse_esfs_strategy
        {
            typedef boost::mpl::bool_<true> type;
            BOOST_STATIC_CONSTANT(bool, value=(type::value));
        };

        // high level metafunction that contains the run_esf_functor corresponding to this backend
        typedef boost::mpl::quote2<run_esf_functor_cuda> run_esf_functor_h_t;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template<enumtype::strategy StrategyId>
        struct select_strategy
        {
            typedef strategy_from_id_cuda<StrategyId> type;
        };

        /*
         * @brief metafunction that determines whether this backend requires redundant computations at halo points
         * of each block, given the strategy Id
         * @tparam StrategyId the strategy id
         * @return always true for CUDA
         */
        template<enumtype::strategy StrategyId>
        struct requires_temporary_redundant_halos
        {
            BOOST_STATIC_ASSERT(StrategyId==enumtype::Block);
            typedef boost::mpl::true_ type;
        };

        /**
         * @brief metafunction that returns the block size
         */
        template<enumtype::strategy StrategyId>
        struct get_block_size {
            GRIDTOOLS_STATIC_ASSERT(StrategyId == enumtype::Block,
                    "For CUDA backend only Block strategy is supported")
            typedef typename strategy_from_id_cuda<StrategyId>::block_size_t type;
        };

        /**
         * @brief metafunction that returns the right iterate domain for this backend
         * (depending on whether the local domain is positional or not)
         * @tparam LocalDomain the local domain
         * @return the iterate domain type for this backend
         */
        template <typename LocalDomain>
        struct select_iterate_domain {
            BOOST_STATIC_ASSERT((is_local_domain<LocalDomain>::value));
            //indirection in order to avoid instantiation of both types of the eval_if
            template<typename _LocalDomain>
            struct select_positional_iterate_domain
            {
                typedef iterate_domain_cuda<positional_iterate_domain, _LocalDomain> type;
            };

            template<typename _LocalDomain>
            struct select_basic_iterate_domain
            {
                typedef iterate_domain_cuda<iterate_domain, _LocalDomain> type;
            };

            typedef typename boost::mpl::eval_if<
                local_domain_is_stateful<LocalDomain>,
                select_positional_iterate_domain<LocalDomain>,
                select_basic_iterate_domain<LocalDomain>
            >::type type;
        };
    };

}//namespace gridtools
