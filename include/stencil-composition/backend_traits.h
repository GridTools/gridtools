#pragma once

#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include "level.h"

#include "backend_traits_cuda.h"
#include "backend_traits_host.h"
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
        struct backend_from_id
        {
        };

/**
@brief traits struct, specialized for the specific strategies
*/
        template<enumtype::strategy Strategy>
        struct strategy_from_id
        {
        };

/**
   @brief wasted code because of the lack of constexpr
   its specializations, given the backend subclass of \ref gridtools::_impl::run_functor, returns the corresponding enum of type \ref gridtools::_impl::BACKEND .
*/
        template <class RunFunctor>
        struct backend_type
        {};

	/** The following struct is defined here since the current version of NVCC does not accept local types to be used as template arguments of __global__ functions \todo move inside backend::run()*/
	template<typename FunctorList, typename LoopIntervals, typename FunctorsMap, typename RangeSizes, typename LocalDomainList, typename Coords, typename ExecutionEngine>
        struct arguments
        {
            typedef FunctorList functor_list_t;
            typedef LoopIntervals loop_intervals_t;
            typedef FunctorsMap functors_map_t;
            typedef RangeSizes range_sizes_t;
            typedef LocalDomainList domain_list_t;
            typedef Coords coords_t;
            typedef ExecutionEngine execution_type_t;
        };

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
            typedef Arguments arguments_t;
            typedef typename Arguments::functor_list_t functor_list_t;
            typedef typename Arguments::loop_intervals_t loop_intervals_t;
            typedef typename Arguments::functors_map_t functors_map_t;
            typedef typename Arguments::range_sizes_t range_sizes_t;
            typedef typename Arguments::domain_list_t domain_list_t;
            typedef typename Arguments::coords_t coords_t;
            typedef Back<Arguments> backend_t;

/**
   @brief traits class to be used inside the functor \ref gridtools::_impl::execute_kernel_functor, which dependson an Index type.
*/
            template <typename Index>
            struct traits{
                typedef typename boost::mpl::at<range_sizes_t, Index>::type range_t;
                typedef typename boost::mpl::at<functor_list_t, Index>::type functor_t;
                typedef typename boost::fusion::result_of::value_at<domain_list_t, Index>::type local_domain_t;
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

            template<typename Backend>
            struct loop
            {
                typedef typename run_functor_traits<Backend>::arguments_t arguments_t;
                typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename arguments_t::functor_list_t>::type::value> iter_range;
                typedef typename arguments_t::domain_list_t domain_list_t;
                typedef typename arguments_t::coords_t coords_t;
                //typedef typename arguments_t::local_domain_t local_domain_t;

                static void run_loop( domain_list_t& local_domain_list, const coords_t& coords)
                    {
                        typedef backend_from_id< backend_type< Backend >::s_backend > backend_traits;

                        backend_traits::template for_each< iter_range >(Backend (local_domain_list, coords));
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

            template< typename Backend >
            struct loop
            {
                typedef typename run_functor_traits<Backend>::arguments_t arguments_t;
                typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename arguments_t::functor_list_t>::type::value> iter_range;
                typedef typename arguments_t::domain_list_t domain_list_t;
                typedef typename arguments_t::coords_t coords_t;

                static void run_loop(domain_list_t& local_domain_list, coords_t const& coords)
                    {
                        typedef backend_from_id< backend_type< Backend >::s_backend > backend_traits;

                        typedef typename boost::mpl::at<typename arguments_t::range_sizes_t, typename boost::mpl::back<iter_range>::type >::type range_t;
                        uint_t n = coords.i_high_bound() + range_t::iplus::value - (coords.i_low_bound() + range_t::iminus::value);
                        uint_t m = coords.j_high_bound() + range_t::jplus::value - (coords.j_low_bound() + range_t::jminus::value);

                        uint_t NBI = n/BI;
                        uint_t NBJ = m/BJ;
                        {
                            //internal blocks
                            for (uint_t bi = 0; bi < NBI; ++bi) {
                                for (uint_t bj = 0; bj < NBJ; ++bj) {
                                    uint_t _starti = bi*BI+coords.i_low_bound();
                                    uint_t _startj = bj*BJ+coords.j_low_bound();
                                    backend_traits::template for_each<iter_range>( Backend (local_domain_list,coords, _starti, _startj, BI-1, BJ-1, bi, bj));
                                }
                            }

                            //last block row
                            for (uint_t bj = 0; bj < NBJ; ++bj) {
                                uint_t _starti = NBI*BI+coords.i_low_bound();
                                uint_t _startj = bj*BJ+coords.j_low_bound();
                                backend_traits::template for_each<iter_range>(Backend (local_domain_list,coords,_starti,_startj, n-NBI*BI, BJ-1, NBI, bj));
                            }

                            //last block column
                            for (uint_t bi = 0; bi < NBI; ++bi) {
                                uint_t _starti = bi*BI+coords.i_low_bound();
                                uint_t _startj = NBJ*BJ+coords.j_low_bound();
                                backend_traits::template for_each<iter_range>(Backend (local_domain_list,coords,_starti,_startj,BI-1, m-NBJ*BJ, bi, NBJ));
                            }

                            //last single block entry
                            {
                                uint_t _starti = NBI*BI+coords.i_low_bound();
                                uint_t _startj = NBJ*BJ+coords.j_low_bound();
                                backend_traits::template for_each<iter_range>( Backend (local_domain_list,coords,_starti,_startj,n-NBI*BI,m-NBJ*BJ, NBI, NBJ));
                            }
                        }
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
