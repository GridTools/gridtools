#pragma once

#include "backend_traits_cuda.h"
#include "backend_traits_naive.h"
#include <boost/fusion/include/value_at.hpp>
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
   @brief template arguments container
   the only purpose of this struct is to collect template arguments in one single types container, in order to lighten the notation
*/
        template <
            typename FunctorList,
            typename LoopIntervals,
            typename FunctorsMap,
            typename RangeSizes,
            typename DomainList,
            typename Coords>
        struct template_argument_traits
        {
            typedef FunctorList functor_list_t;
            typedef LoopIntervals loop_intervals_t;
            typedef FunctorsMap functors_map_t;
            typedef RangeSizes range_sizes_t;
            typedef DomainList domain_list_t;
            typedef Coords coords_t;

        };



/**
   @brief wasted code because of the lack of constexpr
   its specializations, given the backend subclass of \ref gridtools::_impl::run_functor, returns the corresponding enum of type \ref gridtools::_impl::BACKEND .
*/
        template <class RunFunctor>
        struct backend_type
        {};

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
            static const int BI=0;
            static const int BJ=0;
            static const int BK=0;

            template<typename Backend>
            struct loop
            {
                typedef typename run_functor_traits<Backend>::arguments_t arguments_t;
                typedef boost::mpl::range_c<int, 0, boost::mpl::size<typename arguments_t::functor_list_t>::type::value> iter_range;
                typedef typename arguments_t::domain_list_t domain_list_t;
                typedef typename arguments_t::coords_t coords_t;
                //typedef typename arguments_t::local_domain_t local_domain_t;

                static void runLoop( domain_list_t& local_domain_list, const coords_t& coords)
                    {
                        typedef backend_from_id< backend_type< Backend >::s_backend > backend_traits;

                        backend_traits::template for_each< iter_range >(Backend(local_domain_list, coords));
                    }
            };

            //with the naive algorithms, the temporary storages are like the non temporary ones
            template <enumtype::backend Backend, typename ValueType, typename LayoutType , int BI, int BJ, int IMinus, int JMinus, int IPlus, int JPlus>
            struct tmp
                {
                    typedef base_storage<Backend, ValueType, LayoutType, true> host_storage_t;
                };

        };


/**
   @brief specialization for the \ref gridtools::_impl::Block strategy
   The loops over i and j are split according to the values of BI and BJ
*/
        template<>
        struct strategy_from_id <enumtype::Block>
        {
            static const int BI=2;
            static const int BJ=2;
            static const int BK=0;

            template< typename Backend >
            struct loop
            {
                typedef typename run_functor_traits<Backend>::arguments_t arguments_t;
                typedef boost::mpl::range_c<int, 0, boost::mpl::size<typename arguments_t::functor_list_t>::type::value> iter_range;
                typedef typename arguments_t::domain_list_t domain_list_t;
                typedef typename arguments_t::coords_t coords_t;

                static void runLoop(domain_list_t& local_domain_list, coords_t const& coords)
                    {
                        typedef backend_from_id< backend_type< Backend >::s_backend > backend_traits;

                        typedef typename boost::mpl::at<typename arguments_t::range_sizes_t, typename boost::mpl::back<iter_range>::type >::type range_t;
                        int n = coords.i_high_bound() + range_t::iplus::value - (coords.i_low_bound() + range_t::iminus::value);
                        int m = coords.j_high_bound() + range_t::jplus::value - (coords.j_low_bound() + range_t::jminus::value);

                        int NBI = n/BI;
                        int NBJ = m/BJ;
                        {
                            for (int bi = 0; bi < NBI; ++bi) {
                                for (int bj = 0; bj < NBJ; ++bj) {
                                    int _starti = bi*BI+coords.i_low_bound();
                                    int _startj = bj*BJ+coords.j_low_bound();
                                    backend_traits::template for_each<iter_range>( Backend (local_domain_list,coords, _starti, _startj, BI, BJ, bi, bj));
                                }
                            }

                            for (int bj = 0; bj < NBJ; ++bj) {
                                int _starti = NBI*BI+coords.i_low_bound();
                                int _startj = bj*BJ+coords.j_low_bound();
                                backend_traits::template for_each<iter_range>(Backend (local_domain_list,coords,_starti,_startj, n-NBI*BI, BJ, NBI, bj));
                            }

                            for (int bi = 0; bi < NBI; ++bi) {
                                int _starti = bi*BI+coords.i_low_bound();
                                int _startj = NBJ*BJ+coords.j_low_bound();
                                backend_traits::template for_each<iter_range>(Backend (local_domain_list,coords,_starti,_startj,BI, m-NBJ*BJ, bi, NBJ));
                            }

                            {
                                int _starti = NBI*BI+coords.i_low_bound();
                                int _startj = NBJ*BJ+coords.j_low_bound();
                                backend_traits::template for_each<iter_range>( Backend (local_domain_list,coords,_starti,_startj,n-NBI*BI,m-NBJ*BJ, NBI, NBJ));
                            }
                        }
                    }
            };

            template <enumtype::backend Backend, typename ValueType, typename LayoutType , int BI, int BJ, int IMinus, int JMinus, int IPlus, int JPlus>
            struct tmp
                {
                    typedef host_tmp_storage<  Backend, ValueType, LayoutType, BI, BJ, IMinus, JMinus, IPlus, JPlus> host_storage_t;
                };

        };
//    }//namespace _impl
}//namespace gridtools
