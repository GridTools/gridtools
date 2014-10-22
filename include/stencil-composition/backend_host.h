#pragma once

#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/find_if.hpp>

#include "../storage/storage.h"
#include "execution_policy.h"
#include "heap_allocated_temps.h"
#include "backend.h"

#include<boost/mpl/print.hpp>

/**
   @file
   @brief Implements the stencil operations for the host backend
 */

namespace gridtools {

    namespace _impl_host {

/** @brief Host backend
    Derived class of the CRTP pattern defined in \ref gridtools::_impl::run_functor */
        template < typename Arguments >
        struct run_functor_host : public _impl::run_functor < run_functor_host< Arguments > >
        {

            typedef _impl::run_functor < run_functor_host < Arguments > > super;
            explicit run_functor_host(typename Arguments::domain_list_t& domain_list,  typename Arguments::coords_t const& coords)
                : super(domain_list, coords)
            {}

            explicit run_functor_host(typename Arguments::domain_list_t& domain_list,  typename Arguments::coords_t const& coords, uint_t i, uint_t j, uint_t bi, uint_t bj, uint_t blki, uint_t blkj)
                : super(domain_list, coords, i, j, bi, bj, blki, blkj)
            {}

        };
    }

    // namespace _impl{

/** @brief Partial specialization: naive and block implementation for the host backend */
    template <typename Arguments >
    struct execute_kernel_functor < _impl_host::run_functor_host< Arguments > >
    {
        typedef _impl_host::run_functor_host< Arguments > backend_t;

        template <typename FunctorType, typename IntervalMapType, typename IterateDomainType, typename CoordsType>
        struct extra_arguments{
            typedef FunctorType functor_t;
            typedef IntervalMapType interval_map_t;
            typedef IterateDomainType local_domain_t;
            typedef CoordsType coords_t;};

/**
   @brief core of the kernel execution
   \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
*/
            template< typename Traits >
            static void execute_kernel( typename Traits::local_domain_t& local_domain, const backend_t * f )
                {
                    typedef typename Arguments::coords_t coords_type;
                    typedef typename Arguments::loop_intervals_t loop_intervals_t;
                    typedef typename Traits::range_t range_t;
                    typedef typename Traits::functor_t functor_type;
                    typedef typename Traits::local_domain_t  local_domain_t;
                    typedef typename Traits::interval_map_t interval_map_type;
                    typedef typename Traits::iterate_domain_t iterate_domain_type;
                    typedef typename Arguments::execution_type_t execution_type_t;

#ifndef NDEBUG
                    // TODO a generic cout is still on the way (have to implement all the '<<' operators)
                    std::cout << "Functor " <<  functor_type() << "\n";
                    std::cout << "I loop " << f->m_starti  + range_t::iminus::value << " -> "
                              << f->m_starti + f->m_BI + range_t::iplus::value << "\n";
                    std::cout << "J loop " << f->m_startj + range_t::jminus::value << " -> "
                              << f->m_startj + f->m_BJ + range_t::jplus::value << "\n";
                    std::cout <<  " ******************** " << typename Traits::first_hit_t() << "\n";
                    std::cout << " ******************** " << f->m_coords.template value_at<typename Traits::first_hit_t>() << "\n";
#endif


                    for (uint_t i = f->m_starti + range_t::iminus::value;
                         i < f->m_starti + f->m_BI + range_t::iplus::value;
                         ++i)
		      {
			for_each<local_domain.local_args>(increment<0>);
                        for (uint_t j = f->m_startj + range_t::jminus::value;
                             j < f->m_startj + f->m_BJ + range_t::jplus::value;
                             ++j)
                            {
			      for_each<local_domain.local_args>(increment<1>());
			      //#ifndef NDEBUG
			      //std::cout << "Move to : " << i << ", " << j << std::endl;
			      //#endif

                                /** setting an iterator to the address of the current i,j entry to be accessed */
                                iterate_domain_type it_domain(local_domain, i,j, f->m_coords.template value_at<typename Traits::first_hit_t>(), f->blk_idx_i, f->blk_idx_j );


                                //local structs can be passed as template arguments in C++11 (would improve readability)
                                // struct extra_arguments{
                                //     typedef functor_type functor_t;
                                //     typedef interval_map_type interval_map_t;
                                //     typedef iterate_domain_type local_domain_t;
                                //     typedef coords_type coords_t;};

                                /** run the iteration on the k dimension */
                                gridtools::for_each< loop_intervals_t >
                                    (_impl::run_f_on_interval
                                     <
                                     execution_type_t,
                                     extra_arguments<functor_type, interval_map_type, iterate_domain_type, coords_type>
                                     >
                                     (it_domain,f->m_coords)
                                        );
                            }
		      }
                }

        };


/**
   @brief given the backend \ref gridtools::_impl_host::run_functor_host returns the backend ID gridtools::enumtype::Host
   wasted code because of the lack of constexpr
*/
        template <typename Arguments >
        struct backend_type< _impl_host::run_functor_host< Arguments > >
        {
            static const enumtype::backend s_backend=enumtype::Host;
        };

    // } //namespace _impl

} // namespace gridtools
