#pragma once

#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/reverse.hpp>
#include "level.h"

#include "../common/meta_array.h"
#include "axis.h"
#include "backend_traits_cuda.h"
#include "backend_traits_host.h"
#include "execution_types.h"
#include "mss_metafunctions.h"
#include "mss_local_domain.h"
#include "mss.h"
/**
@file

\brief This class contains the traits which are used in backand.h
*/

namespace gridtools{
    /** enum defining the strategy policy for distributing the work. */

//    namespace _impl{

//forward declaration
    template<typename T>
    struct run_functor;

/**
@brief traits struct, specialized for the specific backends.
*/
    template<enumtype::backend Id>
    struct backend_traits_from_id{};

/**
@brief traits struct, specialized for the specific strategies
*/
    template<enumtype::strategy Strategy>
    struct strategy_from_id{};

    /** The following struct is defined here since the current version of NVCC does not accept local types to be used as template arguments of __global__ functions \todo move inside backend::run()*/
    template<
        typename FunctorList,
        typename LoopIntervals,
        typename FunctorsMap,
        typename RangeSizes,
        typename LocalDomainList,
        typename Coords,
        typename ExecutionEngine,
        enumtype::strategy StrategyId>
    struct run_functor_arguments
    {
        typedef FunctorList functor_list_t;
        typedef LoopIntervals loop_intervals_t;
        typedef FunctorsMap functors_map_t;
        typedef RangeSizes range_sizes_t;
        typedef LocalDomainList local_domain_list_t;
        typedef Coords coords_t;
        typedef ExecutionEngine execution_type_t;
        static const enumtype::strategy s_strategy_id=StrategyId;
    };

    template<typename T> struct is_run_functor_arguments : boost::mpl::false_{};

    template<
        typename FunctorList,
        typename LoopIntervals,
        typename FunctorsMap,
        typename RangeSizes,
        typename LocalDomainList,
        typename Coords,
        typename ExecutionEngine,
        enumtype::strategy StrategyId>
    struct is_run_functor_arguments<run_functor_arguments<FunctorList, LoopIntervals, FunctorsMap, RangeSizes, LocalDomainList, Coords, ExecutionEngine, StrategyId> > :
        boost::mpl::true_{};


/** @brief functor struct whose specializations are responsible of running the kernel
    The kernel contains the computational intensive loops on the backend. The fact that it is a functor (and not a templated method) allows for partial specialization (e.g. two backends may share the same strategy)
*/
    template< typename Backend >
    struct execute_kernel_functor
    {
        template< typename Traits >
        static void execute_kernel( const typename Traits::local_domain_t& local_domain, const Backend * f);
    };

/**
   @brief traits struct for the run_functor

   empty declaration
*/
    template <class Subclass>
    struct run_functor_traits{};

/**
   @brief traits struct for the run_functor
   Specialization for all backend classes.
   This struct defines a type for all the template arguments in the run_functor subclasses. It is required because in the run_functor class definition the 'Derived'
   template argument is an incomplete type (ans thus we can not access its template arguments).
   This struct also contains all the type definitions common to all backends.
 */
    template <
        typename Arguments,
        template < typename Argument > class Back
    >
    struct run_functor_traits< Back< Arguments > >
    {
        BOOST_STATIC_ASSERT((is_run_functor_arguments<Arguments>::value));
        typedef Arguments arguments_t;
        typedef typename Arguments::local_domain_list_t local_domain_list_t;
        typedef typename Arguments::coords_t coords_t;
        typedef typename Arguments::functor_list_t functor_list_t;
        typedef typename Arguments::loop_intervals_t loop_intervals_t;
        typedef typename Arguments::functors_map_t functors_map_t;
        typedef typename Arguments::range_sizes_t range_sizes_t;
        typedef Back<Arguments> backend_t;

/**
   @brief traits class to be used inside the functor \ref gridtools::_impl::execute_kernel_functor, which dependson an Index type.
*/
            template <typename Index>
            struct traits{
                typedef typename boost::mpl::at<range_sizes_t, Index>::type range_t;
                typedef typename boost::mpl::at<functor_list_t, Index>::type functor_t;
                typedef typename boost::fusion::result_of::value_at<local_domain_list_t, Index>::type local_domain_t;
                typedef typename boost::mpl::at<functors_map_t, Index>::type interval_map_t;
                typedef typename index_to_level<
                    typename boost::mpl::deref<
                        typename boost::mpl::find_if<
                            loop_intervals_t,
                            boost::mpl::has_key<interval_map_t, boost::mpl::_1>
                            >::type
                        >::type::first
                    >::type first_hit_t;

                typedef typename local_domain_t::iterate_domain_t iterate_domain_t;
            };
    };

    template<typename T> struct printz{BOOST_MPL_ASSERT_MSG((false), ZZZZZZZZZZ, (T));};

    /**
     * @brief functor that executes all the functors contained within the mss
     * @tparam TMssArray meta array containing all the mss descriptors
     * @tparam Coords coordinates
     * @tparam MssLocalDomainArray sequence of mss local domain (each contains the local domain list of a mss)
     * @tparam BackendId id of backend
     * @tparam StrategyId id of strategy
     */
    template<typename TMssArray, typename Coords, typename MssLocalDomainArray, enumtype::backend BackendId, enumtype::strategy StrategyId>
    struct mss_functor
    {
        BOOST_STATIC_ASSERT((is_sequence_of<MssLocalDomainArray, is_mss_local_domain>::value));
        BOOST_STATIC_ASSERT((is_meta_array_of<TMssArray, is_mss_descriptor>::value));
        BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));

        mss_functor(MssLocalDomainArray& local_domain_lists, const Coords& coords, const int block_idx, const int block_idy) :
            m_local_domain_lists(local_domain_lists), m_coords(coords), m_block_idx(block_idx), m_block_idy(block_idy) {}

        /**
         * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
         */
        template <typename Index>
        void operator()(Index const& ) const
        {
            typedef typename boost::fusion::result_of::value_at<MssLocalDomainArray, Index>::type mss_local_domain_t;
            BOOST_STATIC_ASSERT((is_mss_local_domain<mss_local_domain_t>::value));
            BOOST_STATIC_ASSERT((Index::value < boost::mpl::size<typename TMssArray::elements>::value));
            typedef typename boost::mpl::at<typename TMssArray::elements, Index>::type MssType;
            typedef typename mss_local_domain_t::LocalDomainList local_domain_list_t;

            local_domain_list_t& local_domain_list = (local_domain_list_t&)boost::fusion::at<Index>(m_local_domain_lists).local_domain_list;

            typedef typename MssType::execution_engine_t ExecutionEngine;

            typedef typename mss_loop_intervals<MssType, Coords>::type LoopIntervals; // List of intervals on which functors are defined
            //wrapping all the template arguments in a single container
            typedef typename boost::mpl::if_<typename boost::mpl::bool_< ExecutionEngine::type::iteration==enumtype::forward >::type, LoopIntervals, typename boost::mpl::reverse<LoopIntervals>::type >::type oriented_loop_intervals_t;
                        // List of functors to execute (in order)
            typedef typename MssType::functors_list functors_list_t;
            // computed range sizes to know where to compute functot at<i>
            typedef typename MssType::range_sizes range_sizes;
            // Map between interval and actual arguments to pass to Do methods
            typedef typename mss_functor_do_method_lookup_maps<MssType, Coords>::type FunctorsMap;

            // compute the struct with all the type arguments for the run functor
            typedef run_functor_arguments<functors_list_t, oriented_loop_intervals_t, FunctorsMap, range_sizes, local_domain_list_t, Coords, ExecutionEngine, StrategyId> run_functor_args_t;

            typedef backend_traits_from_id< BackendId > backend_traits_t;

            typedef typename backend_traits_t::template execute_traits< run_functor_args_t >::run_functor_t run_functor_t;

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<functors_list_t>::type::value> iter_range;

            //execute all the functors of the mss
            strategy_from_id<StrategyId>::template mss_loop<run_functor_args_t, BackendId>::template run(local_domain_list, m_coords, m_block_idx, m_block_idy);
        }
    private:
        MssLocalDomainArray& m_local_domain_lists;
        const Coords& m_coords;
        const uint_t m_block_idx, m_block_idy;
    };

/**
   @brief specialization for the \ref gridtools::_impl::Naive strategy
   A single loop spans all three directions, i, j and k
*/
    template<>
    struct strategy_from_id< enumtype::Naive>
    {
        static const uint_t BI=0;
        static const uint_t BJ=0;
        static const uint_t BK=0;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam TMssArray a meta array with all the mss descriptors
         * @tparam BackendId id of the backend
         */
        template<typename TMssArray, enumtype::backend BackendId>
        struct fused_mss_loop
        {

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename TMssArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Coords>
            static void run(LocalDomainListArray& local_domain_lists, const Coords& coords)
            {
                typedef backend_traits_from_id< BackendId > backend_traits;
                backend_traits::template for_each<iter_range> (mss_functor<TMssArray, Coords, LocalDomainListArray, BackendId, enumtype::Naive> (local_domain_lists, coords,0,0));
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         * @tparam BackendId id of the backend
         */
        template<typename RunFunctorArgs, enumtype::backend BackendId>
        struct mss_loop
        {
            BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value));
            template<typename LocalDomainList, typename Coords>
            static void run(LocalDomainList& local_domain_list, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));
                typedef backend_traits_from_id< BackendId > backend_traits_t;

                typedef typename backend_traits_t::template execute_traits< RunFunctorArgs >::run_functor_t run_functor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;

                typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<functor_list_t>::type::value> iter_range;

                backend_traits_t::template for_each< iter_range >(run_functor_t(local_domain_list, coords));
            }
        };

        //with the naive algorithms, the temporary storages are like the non temporary ones
        template <enumtype::backend Backend, typename ValueType, typename LayoutType , uint_t BI, uint_t BJ, uint_t IMinus, uint_t JMinus, uint_t IPlus, uint_t JPlus>
        struct tmp
        {
            typedef storage<base_storage<Backend, ValueType, LayoutType, true> > host_storage_t;
        };

    };

//forward declaration
    template< enumtype::backend A,typename B,typename C,uint_t D,uint_t E,uint_t F,uint_t G,uint_t H,uint_t I >
    struct host_tmp_storage;

/**
   @brief specialization for the \ref gridtools::_impl::Block strategy
   The loops over i and j are split according to the values of BI and BJ
*/
    template<>
    struct strategy_from_id <enumtype::Block>
    {
        static const uint_t BI=2;
        static const uint_t BJ=2;
        static const uint_t BK=0;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam TMssArray a meta array with all the mss descriptors
         * @tparam BackendId id of the backend
         */
        template<typename TMssArray, enumtype::backend BackendId>
        struct fused_mss_loop
        {
            BOOST_STATIC_ASSERT((is_meta_array_of<TMssArray, is_mss_descriptor>::value));
            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename TMssArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Coords>
            static void run(LocalDomainListArray& local_domain_lists, const Coords& coords)
            {
                BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));
                typedef backend_traits_from_id<BackendId> backend_traits;

                //TODO consider the largest ij range of all mss to compute number of blocks?
                uint_t n = coords.i_high_bound() - coords.i_low_bound();
                uint_t m = coords.j_high_bound() - coords.j_low_bound();

                uint_t NBI = n/BI;
                uint_t NBJ = m/BJ;
                for (uint_t bi = 0; bi <= NBI; ++bi) {
                    for (uint_t bj = 0; bj <= NBJ; ++bj) {
                        backend_traits::template for_each<iter_range> (mss_functor<TMssArray, Coords, LocalDomainListArray, BackendId, enumtype::Block> (local_domain_lists, coords,bi,bj));
                    }
                }
            }

        };

        /**
         * @brief main execution of a mss for a given IJ block. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         * @tparam BackendId id of the backend
         */
        template<typename RunFunctorArgs, enumtype::backend BackendId>
        struct mss_loop
        {
            BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value));
            template<typename LocalDomainList, typename Coords>
            static void run(LocalDomainList& local_domain_list, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                typedef backend_traits_from_id< BackendId > backend_traits_t;

                typedef typename backend_traits_t::template execute_traits< RunFunctorArgs >::run_functor_t run_functor_t;
                typedef typename RunFunctorArgs::functor_list_t functor_list_t;

                typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<functor_list_t>::type::value> iter_range;

                typedef typename boost::mpl::at<typename RunFunctorArgs::range_sizes_t, typename boost::mpl::back<iter_range>::type >::type range_t;

                uint_t n = coords.i_high_bound() + range_t::iplus::value - coords.i_low_bound() + range_t::iminus::value;
                uint_t m = coords.j_high_bound() + range_t::jplus::value - coords.j_low_bound() + range_t::jminus::value;

                uint_t NBI = n/BI;
                uint_t NBJ = m/BJ;

                uint_t _starti = bi*BI+coords.i_low_bound();
                uint_t _startj = bj*BJ+coords.j_low_bound();

                uint_t block_size_i = BI;
                uint_t block_size_j = BJ;

                if(bi == NBI && bj == NBJ)
                {
                    block_size_i = n-NBI*bi;
                    block_size_j = m-NBJ*bj;
                }
                else if(bi == NBI)
                {
                    block_size_i = n-NBI*bi;
                }
                else if(bj == NBJ)
                {
                    block_size_j = m-NBJ*bj;
                }

                backend_traits_t::template for_each< iter_range >(run_functor_t(local_domain_list, coords, _starti, _startj, block_size_i, block_size_j, bi, bj));
            }
        };

        template <enumtype::backend Backend, typename ValueType, typename LayoutType , uint_t BI, uint_t BJ, uint_t IMinus, uint_t JMinus, uint_t IPlus, uint_t JPlus>
        struct tmp
        {
            typedef host_tmp_storage <  Backend, ValueType, LayoutType, BI, BJ, IMinus, JMinus, IPlus, JPlus> host_storage_t;
        };

    };

    template <enumtype::backend, uint_t Id>
    struct once_per_block{
    };
//    }//namespace _impl
}//namespace gridtools
