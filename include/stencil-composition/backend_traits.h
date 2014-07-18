#pragma once

/**
@file

\brief This class contains the traits which are used in backand.h
*/

namespace gridtools{
    namespace _impl{

        template<typename T>
        struct run_functor;

        enum STRATEGY  {Naive, Block};
        enum BACKEND  {Cuda, Host};

        template<BACKEND Id>
        struct backend_from_id
        {
        };

        template<STRATEGY Strategy>
        struct strategy_from_id
        {
        };

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




        template <BACKEND Backend>
	    struct cout{
            template <typename T>
            void operator <<(T t);
	    };


//wasted code because of the lack of constexpr
        template <class RunFunctor>
        struct backend_type
        {};


    template< typename Backend >
    struct execute_kernel_functor
    {
        template< typename Traits >
        static void execute_kernel( const typename Traits::local_domain_type& local_domain, const Backend * f);
    };

/**
   @brief traits struct for the run_functor

   This struct defines a type for all the template arguments in the run_functor subclasses. It is required because in the run_functor class definition the 'Derived'
   template argument is an incomplete type (ans thus we can not access its template arguments).
   This struct also contains all the type definitions common to all backends.
*/
        template <class Subclass>
        struct run_functor_traits{};

        template <
            typename Arguments,
            template < typename Arguments > class Back
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

            template <typename Index>
            struct traits{
                typedef typename boost::mpl::at<range_sizes_t, Index>::type range_type;
                typedef typename boost::mpl::at<functor_list_t, Index>::type functor_type;
                typedef typename boost::fusion::result_of::value_at<domain_list_t, Index>::type local_domain_type;
                typedef typename boost::mpl::at<functors_map_t, Index>::type interval_map;
                typedef typename index_to_level<
                    typename boost::mpl::deref<
                        typename boost::mpl::find_if<
                            loop_intervals_t,
                            boost::mpl::has_key<interval_map, boost::mpl::_1>
                            >::type
                        >::type::first
                    >::type first_hit;

                typedef typename local_domain_type::iterate_domain_type iterate_domain_type;
            };
        };


        template<>
        struct strategy_from_id< Naive>
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

                static void runLoop( domain_list_t local_domain_list, coords_t coords)
                    {
                        typedef backend_from_id< backend_type< Backend >::m_backend > backend_traits;

                        backend_traits::template for_each< iter_range >(_impl::run_functor< Backend >(local_domain_list, coords));
                    }
            };
        };


        template<>
        struct strategy_from_id <Block>
        {
            static const int BI=4;
            static const int BJ=4;
            static const int BK=0;

            template< typename Backend >
            struct loop
            {
                typedef typename run_functor_traits<Backend>::arguments_t arguments_t;
                typedef boost::mpl::range_c<int, 0, boost::mpl::size<typename arguments_t::functor_list_t>::type::value> iter_range;
                typedef typename arguments_t::domain_list_t domain_list_t;
                typedef typename arguments_t::coords_t coords_t;

                static void runLoop(domain_list_t local_domain_list, coords_t coords)
                    {
                        typedef backend_from_id< backend_type< Backend >::m_backend > backend_traits;
                        backend_traits::template for_each<iter_range>(_impl::run_functor< Backend >(local_domain_list,coords));

                        typedef typename boost::mpl::at<typename arguments_t::range_sizes_t, typename boost::mpl::back<iter_range>::type >::type range_type;
                        int n = coords.i_high_bound() + range_type::iplus::value - (coords.i_low_bound() + range_type::iminus::value);
                        int m = coords.j_high_bound() + range_type::jplus::value - (coords.j_low_bound() + range_type::jminus::value);

                        int NBI = n/BI;
                        int NBJ = m/BJ;
                        {
                            for (int bi = 0; bi < NBI; ++bi) {
                                for (int bj = 0; bj < NBJ; ++bj) {
                                    int starti = bi*BI+coords.i_low_bound();
                                    int startj = bj*BJ+coords.j_low_bound();
                                    backend_traits::template for_each<iter_range>(_impl::run_functor< Backend > (local_domain_list,coords, starti, startj, BI, BJ));
                                }
                            }

                            for (int bj = 0; bj < NBJ; ++bj) {
                                int starti = NBI*BI+coords.i_low_bound();
                                int startj = bj*BJ+coords.j_low_bound();
                                backend_traits::template for_each<iter_range>(_impl::run_functor< Backend > (local_domain_list,coords,starti,startj, n-NBI*BI, BJ));
                            }

                            for (int bi = 0; bi < NBI; ++bi) {
                                int starti = bi*BI+coords.i_low_bound();
                                int startj = NBJ*BJ+coords.j_low_bound();
                                backend_traits::template for_each<iter_range>(_impl::run_functor<Backend> (local_domain_list,coords,starti,startj,BI, n-NBJ*BJ));
                            }

                            int starti = NBI*BI+coords.i_low_bound();
                            int startj = NBJ*BJ+coords.j_low_bound();
                            backend_traits::template for_each<iter_range>(_impl::run_functor < Backend > (local_domain_list,coords,starti,startj,n-NBI*BI,n-NBJ*BJ));
                        }
                    }
            };
        };
    }//namespace _impl
}//namespace gridtools
