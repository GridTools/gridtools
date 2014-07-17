#pragma once

#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/find_if.hpp>
#include "../storage/storage.h"
#include "basic_token_execution.h"
#include "heap_allocated_temps.h"

#include "backend.h"

namespace gridtools {
    namespace _impl{
        template <>
        struct cout<_impl::Host>
        {
            template <typename T>
            const cout& operator << (T arg) const {std::cout<<arg; return *this;}
        };

    }//namespace _impl

    namespace _impl_host {


        template < typename Arguments >
        struct run_functor_host : public _impl::run_functor < run_functor_host< Arguments > >
        {
            //\todo move to the base class

            // useful if we can use constexpr
            // static const _impl::BACKEND m_backend=_impl::Host;
            // static const _impl::BACKEND backend() {return m_backend;} //constexpr

            typedef typename Arguments::coords_t coords_t;
            typedef typename Arguments::domain_list_t domain_list_t;
            coords_t const &coords;
            domain_list_t &domain_list;

            explicit run_functor_host(domain_list_t & domain_list, coords_t const& coords)
                : coords(coords)
                , domain_list(domain_list)
                {}
        };

    } // namespace _impl_host


/** Partial specialization: naive implementation for the host backend (2 policies specify strategy and backend)*/


    struct backend_host: public heap_allocated_temps<backend_host> {
        static const int BI = 0;
        static const int BJ = 0;
        static const int BK = 0;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef storage<ValueType, Layout> type;
        };

        template <typename ValueType, typename Layout>
        struct temporary_storage_type {
            typedef temporary<storage<ValueType, Layout> > type;
        };

//* \todo redundant template arguments
        template < typename Arguments, typename Domain, typename Coords, typename LocalDomainList > // FunctorList, // List of functors to execute (in order)
        // typename range_sizes, // computed range sizes to know where to compute functot at<i>
        // typename LoopIntervals, // List of intervals on which functors are defined
        // typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
        // typename Domain, // Domain class (not really useful maybe)
        // typename Coords, // Coordinate class with domain sizes and splitter coordinates
        // typename LocalDomainList> // List of local domain to be pbassed to functor at<i>
        static void run( Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {

            typedef boost::mpl::range_c<int, 0, boost::mpl::size<typename Arguments::functor_list_t>::type::value> iter_range;

            gridtools::for_each<iter_range>(_impl::run_functor<_impl_host::run_functor_host
                                            < Arguments >
                                            >
                                            (local_domain_list,coords));
        }
    };


    namespace _impl{

        template <typename Arguments >
        struct execute_kernel_functor < _impl::Naive, _impl_host::run_functor_host< Arguments > >
        {
            typedef _impl_host::run_functor_host< Arguments > backend_t;
            template< typename Traits >
            static void execute_kernel( const typename Traits::local_domain_type& local_domain, const backend_t * f )
                {
                    typedef typename Arguments::coords_t coords_t;
                    typedef typename Arguments::loop_intervals_t loop_intervals_t;
                    typedef typename Traits::range_type range_type;
                    typedef typename Traits::functor_type functor_type;
                    typedef typename Traits::local_domain_type  local_domain_type;
                    typedef typename Traits::interval_map interval_map;
                    typedef typename Traits::iterate_domain_type iterate_domain_type;

                    for (int i = f->coords.i_low_bound() + range_type::iminus::value;
                         i < f->coords.i_high_bound() + range_type::iplus::value;
                         ++i)
                        for (int j = f->coords.j_low_bound() + range_type::jminus::value;
                             j < f->coords.j_high_bound() + range_type::jplus::value;
                             ++j)
                        {
                            iterate_domain_type it_domain(local_domain, i,j, f->coords.template value_at<typename Traits::first_hit>());

                            gridtools::for_each<loop_intervals_t>
                                (_impl::run_f_on_interval
                                 <functor_type,
                                 interval_map,
                                 iterate_domain_type,
                                 coords_t>
                                 (it_domain,f->coords)
                                    );
                        }
                }

        };


//wasted code because of the lack of constexpr
        template <typename Arguments >
        struct backend_type< _impl_host::run_functor_host< Arguments > >
        {
            static const BACKEND m_backend=Host;
        };

        /**Traits struct, containing the types which are specific for the host backend*/
        template<>
        struct backend_from_id<Host>
        {
            template <typename ValueType, typename Layout>
            struct storage_traits{
                typedef storage<ValueType, Layout> storage_type;
            };

            template <typename Arguments>
            struct execute_traits
            {
                typedef _impl_host::run_functor_host< Arguments > run_functor;
            };

            //function alias (pre C++11, std::bind or std::mem_fn)
            template<
                typename Sequence
                , typename F
                >
            //unnecessary copies/indirections if the compiler is not smart (std::forward)
            inline static void for_each(F f)
                {
                    gridtools::for_each<Sequence>(f);
                }
        };

    }

} // namespace gridtools
