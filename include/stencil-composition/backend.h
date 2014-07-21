#pragma once

#include "backend_traits.h"


/**
   @file

*/

namespace gridtools {


    namespace _impl {



/**
   \brief "base" struct for all the backend
   This class implements static polimorphism by means of the CRTP pattern. It contains all what is common for all the backends.
*/
        template < typename Derived >
	    struct run_functor {

            typedef Derived derived_t;
            typedef run_functor_traits<Derived> derived_traits;

            typename derived_traits::coords_t const & m_coords;
            typename derived_traits::domain_list_t & m_domain_list;
            int m_starti, m_startj, m_BI, m_BJ;

            // Block strategy
            explicit run_functor(typename derived_traits::domain_list_t& dom_list, typename derived_traits::coords_t const& coords, int i, int j, int bi, int bj)
                :
                  m_domain_list(dom_list)
                , m_coords(coords)
                , m_starti(i)
                , m_startj(j)
                , m_BI(bi)
                , m_BJ(bj)
                {}

            // Naive strategy
            explicit run_functor(typename derived_traits::domain_list_t& dom_list, typename derived_traits::coords_t const& coords)
                :
                  m_domain_list(dom_list)
                , m_coords(coords)
                , m_starti(coords.i_low_bound())
                , m_startj(coords.i_low_bound())
                , m_BI(coords.i_high_bound()-coords.i_low_bound())
                , m_BJ(coords.i_high_bound()-coords.i_low_bound())
                {}


            /**
             * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
             */
            template <typename Index>
            void operator()(Index const& ) const {

#ifndef NDEBUG
                static const BACKEND backend_t = backend_type<derived_t>::m_backend;

                typedef typename derived_traits::template traits<Index>::range_type range_type;
//\todo a generic cout is still on the way (have to implement all the '<<' operators)
                cout< backend_t >() << "Functor " <<  typename derived_traits::template traits<Index>::functor_type() << "\n";
                cout< backend_t >() << "I loop " << m_coords.i_low_bound() + range_type::iminus::value << " -> "
                                    << (m_coords.i_high_bound() + range_type::iplus::value) << "\n";
                cout< backend_t >() << "J loop " << m_coords.j_low_bound() + range_type::jminus::value << " -> "
                                    << m_coords.j_high_bound() + range_type::jplus::value << "\n";
                cout< backend_t >() <<  " ******************** " /*<< first_hit()*/ << "\n";
                cout< backend_t >() << " ******************** " /*<< coords.template value_at<first_hit>()*/ << "\n";
#endif


                typename derived_traits::template traits<Index>::local_domain_type& local_domain = boost::fusion::at<Index>(m_domain_list);
                typedef execute_kernel_functor<  derived_t > functor_type;
                functor_type::template execute_kernel< typename derived_traits::template traits<Index> >(local_domain, static_cast<const derived_t*>(this));

            }
        };
    }//namespace _impl



/**this struct contains the 'run' method for all backends, with a policy determining the specific type. Each backend contains a traits class for the specific case.*/
    template< _impl::BACKEND BackendType, _impl::STRATEGY StrategyType >
    struct backend: public heap_allocated_temps<backend<BackendType, StrategyType > >
    {
        typedef _impl::backend_from_id <BackendType> backend_traits;
        typedef _impl::strategy_from_id <StrategyType> strategy_traits;
        static const _impl::STRATEGY m_strategy_id=StrategyType;
        static const _impl::BACKEND m_backend_id =BackendType;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef typename backend_traits::template storage_traits<ValueType, Layout>::storage_type type;
        };

        template <typename ValueType, typename Layout>
        struct temporary_storage_type {
            typedef temporary< typename backend_traits::template storage_traits<ValueType, Layout>::storage_type > type;
        };


        /**
         * \brief calls the \ref gridtools::run_functor for each functor in the FunctorList.
         * the loop over the functors list is unrolled at compile-time using the for_each construct.
         * \tparam FunctorList  List of functors to execute (in order)
         * \tparam range_sizes computed range sizes to know where to compute functot at<i>
         * \tparam LoopIntervals List of intervals on which functors are defined
         * \tparam FunctorsMap Map between interval and actual arguments to pass to Do methods
         * \tparam Domain Domain class (not really useful maybe)
         * \tparam Coords Coordinate class with domain sizes and splitter coordinates
         * \tparam LocalDomainList List of local domain to be pbassed to functor at<i>
         */
        template <typename FunctorList, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename LoopIntervals, // List of intervals on which functors are defined
                  typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
                  typename Domain, // Domain class (not really useful maybe)
                  typename Coords, // Coordinate class with domain sizes and splitter coordinates
                  typename LocalDomainList> // List of local domain to be pbassed to functor at<i>
        static void run(Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {

            typedef boost::mpl::range_c<int, 0, boost::mpl::size<FunctorList>::type::value> iter_range;
            typedef _impl::template_argument_traits< FunctorList, LoopIntervals, FunctorsMap, range_sizes, LocalDomainList, Coords > arguments;
            typedef typename backend_traits::template execute_traits< arguments >::backend_t backend_t;

            _impl::strategy_from_id< m_strategy_id >::template loop< backend_t >::runLoop(local_domain_list, coords);
        }
    };


} // namespace gridtools
