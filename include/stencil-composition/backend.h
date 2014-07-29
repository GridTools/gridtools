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
            typedef run_functor_traits<Derived> derived_traits_t;

            typename derived_traits_t::coords_t const & m_coords;
            typename derived_traits_t::domain_list_t & m_domain_list;
            int m_starti, m_startj, m_BI, m_BJ;

            // Block strategy
            explicit run_functor(typename derived_traits_t::domain_list_t& dom_list, typename derived_traits_t::coords_t const& coords, int i, int j, int bi, int bj)
                :
                  m_domain_list(dom_list)
                , m_coords(coords)
                , m_starti(i)
                , m_startj(j)
                , m_BI(bi)
                , m_BJ(bj)
                {}

            // Naive strategy
            explicit run_functor(typename derived_traits_t::domain_list_t& dom_list, typename derived_traits_t::coords_t const& coords)
                :
                  m_domain_list(dom_list)
                , m_coords(coords)
                , m_starti(coords.i_low_bound())
                , m_startj(coords.j_low_bound())
                , m_BI(coords.i_high_bound()-coords.i_low_bound())
                , m_BJ(coords.j_high_bound()-coords.j_low_bound())
                {}


            /**
             * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
             */
            template <typename Index>
            void operator()(Index const& ) const {

                typename derived_traits_t::template traits<Index>::local_domain_t& local_domain = boost::fusion::at<Index>(m_domain_list);
                typedef execute_kernel_functor<  derived_t > functor_t;
                functor_t::template execute_kernel< typename derived_traits_t::template traits<Index> >(local_domain, static_cast<const derived_t*>(this));

            }
        };
    }//namespace _impl



/**this struct contains the 'run' method for all backends, with a policy determining the specific type. Each backend contains a traits class for the specific case.*/
    template< enumtype::backend BackendType, enumtype::strategy StrategyType >
    struct backend: public heap_allocated_temps<backend<BackendType, StrategyType > >
    {
        typedef _impl::backend_from_id <BackendType> backend_traits_t;
        typedef _impl::strategy_from_id <StrategyType> strategy_traits_t;
        static const enumtype::strategy s_strategy_id=StrategyType;
        static const enumtype::backend s_backend_id =BackendType;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef typename backend_traits_t::template storage_traits<ValueType, Layout>::storage_t type;
        };

        template <typename ValueType, typename Layout>
        struct temporary_storage_type {
            typedef temporary< typename backend_traits_t::template storage_traits<ValueType, Layout>::storage_t > type;
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
        static void run(Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {// TODO: I would swap the arguments coords and local_domain_list here, for consistency
            //wrapping all the template arguments in a single container
            typedef _impl::template_argument_traits< FunctorList, LoopIntervals, FunctorsMap, range_sizes, LocalDomainList, Coords > arguments_t;
            typedef typename backend_traits_t::template execute_traits< arguments_t >::backend_t backend_t;
            _impl::strategy_from_id< s_strategy_id >::template loop< backend_t >::runLoop(local_domain_list, coords);
        }
    };


} // namespace gridtools
